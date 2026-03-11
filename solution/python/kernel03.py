"""
Optimized MoE — PyTorch only, no custom CUDA/Triton.

Optimizations vs kernel02:
  Level 6: Cache reuse between GEMM1 scratch and GEMM2 output
           One [E, max_Tk, max(2I, H)] buffer serves both passes.
           GEMM1 writes its [2I]-wide output, SwiGLU consumes it immediately,
           then GEMM2 writes its [H]-wide output into the same memory.
           No aliasing hazard: SwiGLU runs before GEMM2 writes.
           Mirrors vLLM's cache13 = torch.empty(M * topk * max(N, K)).

  Fix: _dequant_hidden expand+reshape → repeat_interleave
       expand() produces a stride-0 non-contiguous tensor; reshape on it
       triggers an implicit contiguous() with a different fp32 accumulation
       order, causing bf16 rounding drift vs baseline (abs_err > 0 in
       kernel01/02). repeat_interleave always materialises a contiguous
       tensor with deterministic layout, matching _dequant_weight.

Reverted: weight fuse inside _expert_ffn_inner
       Building tok_w per expert (torch.zeros + fancy index) costs more
       kernel launches than the O_weighted mul it eliminates. This
       optimisation belongs in the Triton kernel epilogue (MUL_ROUTED_WEIGHT),
       not in PyTorch.
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
    g1       = x_e.matmul(w13.t())       # [1, max_Tk, 2I]
    gate, up = g1.chunk(2, dim=-1)       # each [1, max_Tk, I]
    c        = F.silu(up) * gate         # [1, max_Tk, I]  fused
    return c.matmul(w2.t()).squeeze(0)   # [max_Tk, H]


# ---------------------------------------------------------------------------
# 1) Dequantization helpers
# ---------------------------------------------------------------------------

def _dequant_hidden(
    hidden_states:       torch.Tensor,   # [T, H] fp8 e4m3
    hidden_states_scale: torch.Tensor,   # [H//128, T] float
) -> torch.Tensor:                       # [T, H] float32
    """
    Dequantize hidden states.

    Scale layout [H//128, T]: H-block is outer so that for each K-block
    iteration in the GEMM, all T token scales are contiguous → single
    coalesced load per tile.

    repeat_interleave (not expand+reshape): expand produces a stride-0
    non-contiguous tensor; reshape on it implicitly calls contiguous() with
    a different fp32 reduction order → bf16 rounding drift vs baseline.
    repeat_interleave always produces a contiguous tensor, matching
    _dequant_weight and eliminating the numerical discrepancy.
    """
    scale_TH = (
        hidden_states_scale.float()
        .permute(1, 0)                       # [T, H//128]
        .repeat_interleave(BLOCK, dim=1)     # [T, H]  contiguous, deterministic
    )
    return hidden_states.float() * scale_TH


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
    routing_logits:        torch.Tensor,   # [T, E_global] float
    routing_bias:          torch.Tensor,   # [E_global] float
    routed_scaling_factor: float,
    top_k:      int = 8,
    n_group:    int = 8,
    topk_group: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-V3 no-aux two-level routing.

    Returns:
        topk_idx  [T, top_k]  — global expert indices per token
        weights   [T, E]      — normalised routing weights (bias-free sigmoid)
    """
    T, E_global = routing_logits.shape
    group_size  = E_global // n_group

    s           = torch.sigmoid(routing_logits.float())
    s_with_bias = s + routing_bias.float()

    group_scores = (
        s_with_bias.view(T, n_group, group_size)
        .topk(k=2, dim=2).values
        .sum(dim=2)
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_scores.topk(topk_group, dim=1).indices, 1.0)

    score_mask = (
        group_mask.unsqueeze(2)
        .expand(T, n_group, group_size)
        .reshape(T, E_global)
    )

    pruned   = s_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    topk_idx = pruned.topk(k=top_k, dim=1).indices

    mask    = torch.zeros_like(s).scatter_(1, topk_idx, 1.0)
    weights = s * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor

    return topk_idx, weights


# ---------------------------------------------------------------------------
# 3) Token permutation
# ---------------------------------------------------------------------------

def _build_permutation(
    topk_idx:    torch.Tensor,   # [T, top_k]
    local_start: int,
    e_local:     int,
    top_k:       int,
    device:      torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort all (token, expert) pairs by local expert index.

    Returns:
        sorted_token_idx   [M]  — token indices in expert-sorted order
        sorted_expert_idx  [M]  — local expert indices 0..E_local-1
        expert_counts      [E]  — tokens per expert
        expert_offsets     [E]  — cumulative segment starts (O(1) lookup)
        slot_idx           [M]  — position within each expert's segment
    """
    T = topk_idx.shape[0]

    flat_expert = topk_idx.reshape(-1)
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
    Gather token hidden states and scatter into [E, max_Tk, H] without a
    Python loop. Single vectorised write via (expert_idx, slot_idx) indices.
    """
    H        = A.shape[1]
    X_perm   = A[sorted_token_idx]
    X_padded = torch.zeros(e_local, max_tk, H, device=A.device, dtype=A.dtype)
    X_padded[sorted_expert_idx, slot_idx] = X_perm
    return X_padded


# ---------------------------------------------------------------------------
# 5) Per-expert FFN — lazy dequant + compiled inner, shared cache
# ---------------------------------------------------------------------------

def _expert_ffn(
    X_padded:            torch.Tensor,   # [E, max_Tk, H]
    gemm1_weights:       torch.Tensor,   # [E, 2I, H] fp8
    gemm1_weights_scale: torch.Tensor,   # [E, 2I//128, H//128] float
    gemm2_weights:       torch.Tensor,   # [E, H, I] fp8
    gemm2_weights_scale: torch.Tensor,   # [E, H//128, I//128] float
    expert_counts:       torch.Tensor,   # [E] long
    cache:               torch.Tensor,   # [E, max_Tk, max(2I, H)] shared buffer
) -> torch.Tensor:                       # cache[:, :, :H] view — [E, max_Tk, H]
    """
    Per-expert loop: lazy dequant + compiled GEMM1+SwiGLU+GEMM2.

    Writes GEMM2 output into cache[:, :, :H] (a view, no copy).
    GEMM1's [2I]-wide intermediate lives only inside the compiled graph;
    it never aliases the cache buffer, so there is no read-after-write hazard.
    Peak weight memory: one expert's w13 + w2 at a time (~200 MB).
    """
    H      = X_padded.shape[2]
    counts = expert_counts.tolist()

    O_padded = cache[:, :, :H]   # [E, max_Tk, H]  — alias, no allocation

    for le in range(len(counts)):
        if counts[le] == 0:
            O_padded[le].zero_()
            continue

        w13 = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le])  # [2I, H]
        w2  = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le])  # [H, I]

        O_padded[le] = _expert_ffn_inner(
            X_padded[le].unsqueeze(0),   # [1, max_Tk, H]
            w13, w2,
        )
        # w13, w2 refcount → 0 here; CPython releases immediately → ~200 MB peak

    return O_padded   # [E, max_Tk, H]


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
    Single gather to unpad [E, max_Tk, H] → [M, H], apply routing weights,
    then scatter_add to [T, H]. No Python loop over experts.
    """
    H      = O_padded.shape[2]
    device = O_padded.device

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
    routing_logits:        torch.Tensor,   # [T, E_global] float
    routing_bias:          torch.Tensor,   # [E_global] float
    hidden_states:         torch.Tensor,   # [T, H] fp8 e4m3
    hidden_states_scale:   torch.Tensor,   # [H//128, T] float
    gemm1_weights:         torch.Tensor,   # [E_local, 2I, H] fp8
    gemm1_weights_scale:   torch.Tensor,   # [E_local, 2I//128, H//128] float
    gemm2_weights:         torch.Tensor,   # [E_local, H, I] fp8
    gemm2_weights_scale:   torch.Tensor,   # [E_local, H//128, I//128] float
    local_expert_offset:   int,
    routed_scaling_factor: float,
) -> torch.Tensor:                         # [T, H] bfloat16
    H       = hidden_states.shape[1]
    I       = gemm2_weights.shape[2]
    E_local = gemm1_weights.shape[0]
    T       = routing_logits.shape[0]
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
        return torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    max_tk = int(expert_counts.max())

    # 4. Build padded input [E, max_Tk, H]
    X_padded = _build_padded_input(
        A, sorted_token_idx, sorted_expert_idx, slot_idx, E_local, max_tk,
    )

    # -------------------------------------------------------------------------
    # Shared cache sized [E, max_Tk, max(2I, H)]:
    #   For DeepSeek-V3: 2I=4096 < H=7168 → cache_width = H, zero overhead.
    #   For configs where 2I > H: cache_width = 2I, still one allocation.
    # -------------------------------------------------------------------------
    cache_width = max(2 * I, H)
    cache = torch.empty(E_local, max_tk, cache_width, device=device, dtype=torch.float32)

    # 5. Per-expert FFN (lazy dequant + compiled inner, shared cache)
    O_padded = _expert_ffn(
        X_padded,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        expert_counts,
        cache,
    )

    # 6. Weighted scatter-add
    output = _weighted_scatter(
        O_padded, sorted_token_idx, sorted_expert_idx,
        slot_idx, weights, local_start, T,
    )

    return output.to(torch.bfloat16)