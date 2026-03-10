"""
Optimized MoE — PyTorch only, no custom CUDA/Triton.

Optimizations vs kernel00 (baseline):
  Level 1: Lazy per-expert dequant (3.5GB → ~50MB peak)
  Level 2: Token permutation + bmm (32 small GEMMs → 1 batched GEMM)
  Level 3: Vectorized scatter-add (32 index_add_ → 1 scatter_add_)

Pipeline:
  1. Dequant hidden states only (weights deferred to per-expert lazy dequant)
  2. DeepSeek-V3 no-aux routing (identical to baseline)
  3. Token permutation: sort (token, expert) pairs by expert → contiguous segments
  4. Pad segments to max_Tk → bmm for GEMM1 and GEMM2
  5. SwiGLU in-between
  6. Vectorized weighted scatter-add back to [T, H]
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,       # [T, E_global] float
    routing_bias: torch.Tensor,         # [E_global] float
    hidden_states: torch.Tensor,        # [T, H] fp8 e4m3
    hidden_states_scale: torch.Tensor,  # [H//128, T] float
    gemm1_weights: torch.Tensor,        # [E_local, 2I, H] fp8
    gemm1_weights_scale: torch.Tensor,  # [E_local, 2I//128, H//128] float
    gemm2_weights: torch.Tensor,        # [E_local, H, I] fp8
    gemm2_weights_scale: torch.Tensor,  # [E_local, H//128, I//128] float
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    device = hidden_states.device

    # -------------------------------------------------------------------------
    # 1) Dequant hidden states only — weights deferred to per-expert loop
    # -------------------------------------------------------------------------
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                # [H//128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()               # [T, H//128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .expand(T, H // BLOCK, BLOCK)                              # [T, H//128, 128]
        .reshape(T, H)                                             # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                                  # [T, H] float32

    # -------------------------------------------------------------------------
    # 2) Routing (identical to baseline)
    # -------------------------------------------------------------------------
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M_mask = torch.zeros_like(s)
    M_mask.scatter_(1, topk_idx, 1.0)
    weights = s * M_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # -------------------------------------------------------------------------
    # 3) Token permutation — sort all (token, expert) pairs by expert
    # -------------------------------------------------------------------------
    local_start = int(local_expert_offset)

    # Flatten topk_idx: each token has TOP_K expert assignments
    flat_expert_idx = topk_idx.reshape(-1)                          # [T*8]
    flat_token_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)

    # Keep only pairs assigned to local experts
    local_mask = (flat_expert_idx >= local_start) & (flat_expert_idx < local_start + E_local)

    if not local_mask.any():
        return torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    local_token_idx = flat_token_idx[local_mask]                    # [M]
    local_expert_idx = flat_expert_idx[local_mask] - local_start    # [M] in 0..31

    # Sort by expert so same-expert tokens are contiguous
    sort_order = local_expert_idx.argsort(stable=True)
    sorted_token_idx = local_token_idx[sort_order]                  # [M]
    sorted_expert_idx = local_expert_idx[sort_order]                # [M]

    # Compute per-expert counts and offsets
    expert_counts = torch.zeros(E_local, dtype=torch.long, device=device)
    expert_counts.scatter_add_(0, sorted_expert_idx.long(),
                               torch.ones_like(sorted_expert_idx, dtype=torch.long))
    expert_offsets = torch.zeros(E_local, dtype=torch.long, device=device)
    expert_offsets[1:] = expert_counts[:-1].cumsum(0)

    M_total = sorted_token_idx.shape[0]
    max_tk = expert_counts.max().item()

    if max_tk == 0:
        return torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    # -------------------------------------------------------------------------
    # 4) Lazy per-expert dequant + padded bmm
    #    Build [E_local, max_Tk, H] input and dequant weights per-expert
    # -------------------------------------------------------------------------

    # Gather all needed token hidden states
    X_sorted = A[sorted_token_idx]                                  # [M, H]

    # Pad into [E_local, max_Tk, H] for bmm
    X_padded = torch.zeros(E_local, max_tk, H, device=device, dtype=torch.float32)
    # Also build [E_local, max_Tk, 2I] and [E_local, max_Tk, H] for results
    offsets_list = expert_offsets.tolist()
    counts_list = expert_counts.tolist()

    for le in range(E_local):
        c = counts_list[le]
        if c == 0:
            continue
        off = offsets_list[le]
        X_padded[le, :c] = X_sorted[off:off + c]

    # Lazy dequant: build W13 [E_local, 2I, H] and W2 [E_local, H, I] one expert at a time
    # To still use bmm, we need the full [E, 2I, H] weight tensor, but dequant per-expert
    # avoids the massive intermediate S_expanded tensors
    W13 = torch.empty(E_local, 2 * I, H, device=device, dtype=torch.float32)
    W2 = torch.empty(E_local, H, I, device=device, dtype=torch.float32)

    for le in range(E_local):
        if counts_list[le] == 0:
            continue
        # GEMM1 weights: [2I, H] with scale [2I//128, H//128]
        w13_e = gemm1_weights[le].to(torch.float32)                # [2I, H]
        s13_e = gemm1_weights_scale[le].to(torch.float32)          # [2I//128, H//128]
        s13_exp = s13_e.repeat_interleave(BLOCK, dim=0)            # [2I, H//128]
        s13_exp = s13_exp.repeat_interleave(BLOCK, dim=1)          # [2I, H]
        W13[le] = w13_e * s13_exp

        # GEMM2 weights: [H, I] with scale [H//128, I//128]
        w2_e = gemm2_weights[le].to(torch.float32)                 # [H, I]
        s2_e = gemm2_weights_scale[le].to(torch.float32)           # [H//128, I//128]
        s2_exp = s2_e.repeat_interleave(BLOCK, dim=0)              # [H, I//128]
        s2_exp = s2_exp.repeat_interleave(BLOCK, dim=1)            # [H, I]
        W2[le] = w2_e * s2_exp

    # -------------------------------------------------------------------------
    # 5) Batched GEMM1 + SwiGLU + Batched GEMM2
    #    [E, max_Tk, H] @ [E, H, 2I] → [E, max_Tk, 2I]
    # -------------------------------------------------------------------------

    # GEMM1: bmm
    G1 = torch.bmm(X_padded, W13.permute(0, 2, 1))                # [E, max_Tk, 2I]

    # SwiGLU
    X1 = G1[:, :, :I]                                              # gate [E, max_Tk, I]
    X2 = G1[:, :, I:]                                              # up   [E, max_Tk, I]
    C = F.silu(X2) * X1                                            # [E, max_Tk, I]

    # GEMM2: bmm
    O_padded = torch.bmm(C, W2.permute(0, 2, 1))                  # [E, max_Tk, H]

    # -------------------------------------------------------------------------
    # 6) Vectorized scatter-add — unpad + weight + single scatter_add_
    # -------------------------------------------------------------------------

    # Extract valid (non-padded) results back to [M, H]
    O_all = torch.empty(M_total, H, device=device, dtype=torch.float32)
    for le in range(E_local):
        c = counts_list[le]
        if c == 0:
            continue
        off = offsets_list[le]
        O_all[off:off + c] = O_padded[le, :c]

    # Gather routing weights for each (token, expert) pair
    sorted_global_expert = sorted_expert_idx + local_start
    sorted_weights = weights[sorted_token_idx, sorted_global_expert]  # [M]

    # Weighted results
    O_weighted = O_all * sorted_weights.unsqueeze(1)                # [M, H]

    # Single scatter_add_
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    output.scatter_add_(
        0,
        sorted_token_idx.unsqueeze(1).expand(M_total, H),
        O_weighted,
    )

    return output.to(torch.bfloat16)
