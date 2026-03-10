"""
Optimized MoE — PyTorch only, no custom CUDA/Triton.

Optimizations vs kernel01:
  Same Level 1/2/3 optimizations, plus:
  Level 4: Functional decomposition into focused sub-functions
  Level 5: torch.compile on expert_ffn inner loop (fuses chunk/silu/mul → single kernel)

Sub-functions:
  _dequant_hidden       — FP8 block-scale dequant for hidden states
  _dequant_weight       — FP8 block-scale dequant for a single [N, K] weight tensor
  _routing              — DeepSeek-V3 no-aux two-level routing
  _build_permutation    — sort (token, expert) pairs, compute counts/offsets/slots
  _build_padded_input   — vectorized scatter into [E, max_Tk, H]
  _expert_ffn           — dequant + GEMM1 + SwiGLU + GEMM2 for one expert (compiled)
  _weighted_scatter     — vectorized unpad + weighted scatter_add → [T, H]

torch.compile boundary:
  _expert_ffn_inner is compiled separately so chunk/silu/mul are fused into a single
  kernel. The dequant (repeat_interleave) sits outside the compiled region because
  its data-dependent shapes prevent static compilation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

BLOCK = 128  # FP8 block-scale granularity


# ---------------------------------------------------------------------------
# Compiled inner FFN — chunk/silu/mul/matmul fused by torch.compile
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True)
def _expert_ffn_inner(
    x_e: torch.Tensor,   # [1, max_Tk, H]
    w13: torch.Tensor,   # [2I, H]  already dequanted f32
    w2:  torch.Tensor,   # [H, I]   already dequanted f32
) -> torch.Tensor:       # [max_Tk, H]
    """GEMM1 → SwiGLU → GEMM2 for a single expert, fused by torch.compile."""
    g1          = x_e.matmul(w13.t())           # [1, max_Tk, 2I]
    gate, up    = g1.chunk(2, dim=-1)           # each [1, max_Tk, I]
    c           = F.silu(up) * gate             # [1, max_Tk, I]  fused
    return c.matmul(w2.t()).squeeze(0)          # [max_Tk, H]


# ---------------------------------------------------------------------------
# 1) Dequantization helpers
# ---------------------------------------------------------------------------

def _dequant_hidden(
    hidden_states:       torch.Tensor,   # [T, H] fp8 e4m3
    hidden_states_scale: torch.Tensor,   # [H//128, T] float
) -> torch.Tensor:                       # [T, H] float32
    """
    Dequantize hidden states.
    Scale layout [H//128, T] is transposed vs logical [T, H//128]:
    for each H-block iteration in the GEMM K-loop, all T token scales
    are contiguous → single coalesced load per tile.
    """
    T  = hidden_states.shape[0]
    H  = hidden_states.shape[1]
    nb = H // BLOCK

    scale_expanded = (
        hidden_states_scale.float()          # [H//128, T]
        .permute(1, 0)                       # [T, H//128]
        .unsqueeze(-1)
        .expand(T, nb, BLOCK)               # [T, H//128, 128]
        .reshape(T, H)
    )
    return hidden_states.float() * scale_expanded   # [T, H]


def _dequant_weight(
    weight: torch.Tensor,   # [N, K] fp8
    scale:  torch.Tensor,   # [N//128, K//128] float
) -> torch.Tensor:          # [N, K] float32
    """
    Dequantize a single 2-D weight matrix with block scales.
    Scale layout [N_blocks, K_blocks] matches standard GEMM tile order.
    """
    return weight.float() * (
        scale.float()
        .repeat_interleave(BLOCK, dim=0)    # [N, K//128]
        .repeat_interleave(BLOCK, dim=1)    # [N, K]
    )


# ---------------------------------------------------------------------------
# 2) Routing
# ---------------------------------------------------------------------------

def _routing(
    routing_logits: torch.Tensor,   # [T, E_global] float
    routing_bias:   torch.Tensor,   # [E_global] float
    routed_scaling_factor: float,
    top_k:      int = 8,
    n_group:    int = 8,
    topk_group: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-V3 no-aux two-level routing.

    Returns:
        topk_idx  [T, top_k]  — global expert indices selected per token
        weights   [T, E]      — normalized routing weights (bias-free sigmoid)
    """
    T, E_global = routing_logits.shape
    group_size  = E_global // n_group

    s           = torch.sigmoid(routing_logits.float())        # [T, E]
    s_with_bias = s + routing_bias.float()

    # Coarse: top-topk_group groups by sum-of-top2 score
    group_scores = (
        s_with_bias.view(T, n_group, group_size)
        .topk(k=2, dim=2).values
        .sum(dim=2)                                            # [T, n_group]
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_scores.topk(topk_group, dim=1).indices, 1.0)

    score_mask = (
        group_mask.unsqueeze(2)
        .expand(T, n_group, group_size)
        .reshape(T, E_global)
    )

    # Fine: global top-k within selected groups
    pruned   = s_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    topk_idx = pruned.topk(k=top_k, dim=1).indices            # [T, top_k]

    # Routing weights: unbiased sigmoid, normalized, scaled
    mask    = torch.zeros_like(s).scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor

    return topk_idx, weights


# ---------------------------------------------------------------------------
# 3) Token permutation
# ---------------------------------------------------------------------------

def _build_permutation(
    topk_idx:     torch.Tensor,   # [T, top_k]
    local_start:  int,
    e_local:      int,
    top_k:        int,
    device:       torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort all (token, expert) pairs by local expert index.

    Returns:
        sorted_token_idx   [M]   — token indices in expert-sorted order
        sorted_expert_idx  [M]   — local expert indices (0..E_local-1)
        expert_counts      [E]   — tokens per expert
        expert_offsets     [E]   — cumulative start of each expert's segment
        slot_idx           [M]   — position of each pair within its expert's segment
    """
    T = topk_idx.shape[0]

    flat_expert = topk_idx.reshape(-1)                         # [T*top_k]
    flat_token  = (
        torch.arange(T, device=device)
        .unsqueeze(1).expand(T, top_k).reshape(-1)
    )

    local_mask        = (flat_expert >= local_start) & (flat_expert < local_start + e_local)
    local_token_idx   = flat_token[local_mask]
    local_expert_idx  = flat_expert[local_mask] - local_start

    order             = local_expert_idx.argsort(stable=True)
    sorted_token_idx  = local_token_idx[order]
    sorted_expert_idx = local_expert_idx[order]

    expert_counts  = sorted_expert_idx.bincount(minlength=e_local)
    expert_offsets = F.pad(expert_counts[:-1].cumsum(0), (1, 0))

    # Slot index: position of token i within its expert's segment
    slot_idx = (
        torch.arange(sorted_token_idx.shape[0], device=device)
        - expert_offsets[sorted_expert_idx]
    )

    return sorted_token_idx, sorted_expert_idx, expert_counts, expert_offsets, slot_idx


# ---------------------------------------------------------------------------
# 4) Build padded input
# ---------------------------------------------------------------------------

def _build_padded_input(
    A:                 torch.Tensor,   # [T, H] f32
    sorted_token_idx:  torch.Tensor,   # [M]
    sorted_expert_idx: torch.Tensor,   # [M]
    slot_idx:          torch.Tensor,   # [M]
    e_local:           int,
    max_tk:            int,
) -> torch.Tensor:                     # [E, max_Tk, H]
    """
    Gather token hidden states and scatter into [E, max_Tk, H] without a Python loop.
    Uses (expert_idx, slot_idx) as 2-D indices for a single vectorized write.
    """
    H        = A.shape[1]
    X_perm   = A[sorted_token_idx]                             # [M, H]
    X_padded = torch.zeros(e_local, max_tk, H, device=A.device, dtype=A.dtype)
    X_padded[sorted_expert_idx, slot_idx] = X_perm            # single scatter
    return X_padded


# ---------------------------------------------------------------------------
# 5) Per-expert FFN  (dequant outside compile, matmul/activation inside)
# ---------------------------------------------------------------------------

def _expert_ffn(
    X_padded:            torch.Tensor,   # [E, max_Tk, H]
    gemm1_weights:       torch.Tensor,   # [E, 2I, H] fp8
    gemm1_weights_scale: torch.Tensor,   # [E, 2I//128, H//128] float
    gemm2_weights:       torch.Tensor,   # [E, H, I] fp8
    gemm2_weights_scale: torch.Tensor,   # [E, H//128, I//128] float
    expert_counts:       torch.Tensor,   # [E] long
) -> torch.Tensor:                       # [E, max_Tk, H]
    """
    For each local expert:
      1. Lazy dequant weight (only if expert has tokens) — stays in Python, not compiled
         so repeat_interleave's data-dependent shapes don't block torch.compile.
      2. Call _expert_ffn_inner (compiled) — GEMM1 + SwiGLU + GEMM2 fused.

    Peak weight memory: ~2 experts in flight (current + next iteration).
    """
    E_local  = X_padded.shape[0]
    max_tk   = X_padded.shape[1]
    H        = X_padded.shape[2]
    device   = X_padded.device

    O_padded = torch.empty(E_local, max_tk, H, device=device, dtype=torch.float32)
    counts   = expert_counts.tolist()

    for le in range(E_local):
        if counts[le] == 0:
            O_padded[le].zero_()
            continue

        # Dequant outside compile: repeat_interleave has data-dependent output shape
        w13 = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le])  # [2I, H]
        w2  = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le])  # [H, I]

        # Compiled: chunk + silu + mul + two matmuls fused
        O_padded[le] = _expert_ffn_inner(
            X_padded[le].unsqueeze(0),   # [1, max_Tk, H]
            w13, w2,
        )
        # w13, w2 ref-count → 0 here; CPython releases immediately → ~200MB peak

    return O_padded


# ---------------------------------------------------------------------------
# 6) Weighted scatter-add
# ---------------------------------------------------------------------------

def _weighted_scatter(
    O_padded:          torch.Tensor,   # [E, max_Tk, H]
    sorted_token_idx:  torch.Tensor,   # [M]
    sorted_expert_idx: torch.Tensor,   # [M]
    slot_idx:          torch.Tensor,   # [M]
    weights:           torch.Tensor,   # [T, E_global]
    local_start:       int,
    T:                 int,
) -> torch.Tensor:                     # [T, H] f32
    """
    Unpad O_padded → [M, H] via a single gather, apply routing weights,
    then scatter_add back to [T, H]. No Python loop over experts.
    """
    H = O_padded.shape[2]
    device = O_padded.device

    # Single gather: (expert_idx, slot_idx) → [M, H]
    O_perm = O_padded[sorted_expert_idx, slot_idx]             # [M, H]

    sorted_weights = weights[
        sorted_token_idx,
        sorted_expert_idx + local_start,
    ]                                                          # [M]

    O_weighted = O_perm * sorted_weights.unsqueeze(1)          # [M, H]

    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    output.scatter_add_(
        0,
        sorted_token_idx.unsqueeze(1).expand(-1, H),
        O_weighted,
    )
    return output


# ---------------------------------------------------------------------------
# Top-level kernel
# ---------------------------------------------------------------------------

@torch.no_grad()
def kernel(
    routing_logits:       torch.Tensor,   # [T, E_global] float
    routing_bias:         torch.Tensor,   # [E_global] float
    hidden_states:        torch.Tensor,   # [T, H] fp8 e4m3
    hidden_states_scale:  torch.Tensor,   # [H//128, T] float
    gemm1_weights:        torch.Tensor,   # [E_local, 2I, H] fp8
    gemm1_weights_scale:  torch.Tensor,   # [E_local, 2I//128, H//128] float
    gemm2_weights:        torch.Tensor,   # [E_local, H, I] fp8
    gemm2_weights_scale:  torch.Tensor,   # [E_local, H//128, I//128] float
    local_expert_offset:  int,
    routed_scaling_factor: float,
) -> torch.Tensor:                        # [T, H] bfloat16
    E_local     = gemm1_weights.shape[0]
    T           = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device      = hidden_states.device

    # 1. Dequant hidden states
    A = _dequant_hidden(hidden_states, hidden_states_scale)    # [T, H] f32

    # 2. Routing
    topk_idx, weights = _routing(
        routing_logits, routing_bias, routed_scaling_factor,
    )

    # 3. Permutation
    (sorted_token_idx, sorted_expert_idx,
     expert_counts, expert_offsets, slot_idx) = _build_permutation(
        topk_idx, local_start, E_local, topk_idx.shape[1], device,
    )

    M_total = sorted_token_idx.shape[0]
    if M_total == 0:
        return torch.zeros(T, hidden_states.shape[1],
                           dtype=torch.bfloat16, device=device)

    max_tk = int(expert_counts.max())

    # 4. Build padded input [E, max_Tk, H]
    X_padded = _build_padded_input(
        A, sorted_token_idx, sorted_expert_idx, slot_idx, E_local, max_tk,
    )

    # 5. Per-expert FFN (lazy dequant + compiled inner loop)
    O_padded = _expert_ffn(
        X_padded,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        expert_counts,
    )

    # 6. Weighted scatter-add
    output = _weighted_scatter(
        O_padded, sorted_token_idx, sorted_expert_idx,
        slot_idx, weights, local_start, T,
    )

    return output.to(torch.bfloat16)