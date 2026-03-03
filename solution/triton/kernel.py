"""
Triton fused MoE kernel for moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.

Architecture:
  1. Routing (PyTorch) — sigmoid, group-topk, global top-8, weight norm
  2. Token permutation — sort tokens by expert for coalesced access
  3. GEMM1 fused with FP8 block-scale dequant: [Tk, H] x [2I, H]^T -> [Tk, 2I]
  4. SwiGLU activation
  5. GEMM2 fused with FP8 block-scale dequant: [Tk, I] x [H, I]^T -> [Tk, H]
  6. Weighted scatter-add back to output
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton GEMM kernel with FP8 block-scale dequantization
# ============================================================================
@triton.jit
def _gemm_fp8_blockscale(
    # Pointers
    A_ptr, A_scale_ptr,   # activation [M, K] fp8, scale [K//BS, M] (transposed!)
    B_ptr, B_scale_ptr,   # weight [N, K] fp8, scale [N//BS, K//BS]
    C_ptr,                # output [M, N] fp32
    # Dims
    M, N, K,
    # Strides for A [M, K]
    stride_am, stride_ak,
    # Strides for A_scale [K//BS, M] (transposed layout)
    stride_as0, stride_as1,
    # Strides for B [N, K]
    stride_bn, stride_bk,
    # Strides for B_scale [N//BS, K//BS]
    stride_bs0, stride_bs1,
    # Strides for C [M, N]
    stride_cm, stride_cn,
    # Block scale size
    BLOCK_SCALE: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled GEMM: C[m,n] = sum_k dequant(A[m,k]) * dequant(B[n,k])

    FP8 block-scale dequant is fused: after accumulating a BLOCK_K tile in FP8,
    we multiply by the appropriate scales before adding to the FP32 accumulator.

    BLOCK_K must equal BLOCK_SCALE (128) for correct scale indexing.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for ki in range(num_k_tiles):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile [BLOCK_N, BLOCK_K] -> we need [BLOCK_K, BLOCK_N] for matmul
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Do the matmul in fp8 -> fp32: A[M,K] @ B[N,K]^T = A[M,K] @ B^T[K,N]
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # We need a @ b^T
        a_fp32 = a.to(tl.float32)
        b_fp32 = b.to(tl.float32)

        # Block scale indices: ki-th K-block
        k_block_idx = ki  # since BLOCK_K == BLOCK_SCALE

        # A_scale: [K//BS, M] transposed layout. For token m, k-block ki: A_scale[ki, m]
        a_scale_ptrs = A_scale_ptr + k_block_idx * stride_as0 + offs_m * stride_as1
        a_scale_mask = offs_m < M
        a_scale = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)  # [BLOCK_M]

        # B_scale: [N//BS, K//BS]. For n-block pid_n, k-block ki
        # But offs_n can span multiple scale blocks if BLOCK_N > BLOCK_SCALE
        # Since BLOCK_N <= BLOCK_SCALE typically, each n in the tile maps to n_block = n // BLOCK_SCALE
        n_block_indices = offs_n // BLOCK_SCALE  # [BLOCK_N]
        b_scale_ptrs = B_scale_ptr + n_block_indices * stride_bs0 + k_block_idx * stride_bs1
        b_scale_mask = offs_n < N
        b_scale = tl.load(b_scale_ptrs, mask=b_scale_mask, other=1.0)  # [BLOCK_N]

        # Dequantize: A_deq[m,k] = A_fp8[m,k] * a_scale[m]
        #             B_deq[n,k] = B_fp8[n,k] * b_scale[n]
        # C += A_deq @ B_deq^T = (A_fp8 * a_scale[:, None]) @ (B_fp8 * b_scale[:, None])^T
        #    = (a_scale[:, None]) * (A_fp8 @ B_fp8^T) * (b_scale[None, :])  -- but this is WRONG
        #      because scales are per-K-block, not per-element across K
        # Since BLOCK_K == BLOCK_SCALE, each a_scale/b_scale is constant across the K-tile.
        # So: tile_result[m, n] = a_scale[m] * (sum_k A_fp8[m,k] * B_fp8[n,k]) * b_scale[n]
        tile = tl.dot(a_fp32, tl.trans(b_fp32))  # [BLOCK_M, BLOCK_N]
        tile = tile * a_scale[:, None] * b_scale[None, :]
        acc += tile

    # Store C [BLOCK_M, BLOCK_N]
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


# ============================================================================
# SwiGLU activation kernel
# ============================================================================
@triton.jit
def _swiglu(
    X_ptr,   # input [M, 2*I], contains [gate, up] concatenated
    O_ptr,   # output [M, I]
    M, I_val,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I_val)

    # Load gate (first I columns) and up (last I columns)
    gate_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_i[None, :] * stride_xk
    up_ptrs = X_ptr + offs_m[:, None] * stride_xm + (offs_i[None, :] + I_val) * stride_xk

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU: silu(up) * gate, where silu(x) = x * sigmoid(x)
    silu_up = up * tl.sigmoid(up)
    out = silu_up * gate

    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_i[None, :] * stride_ok
    tl.store(o_ptrs, out, mask=mask)


# ============================================================================
# Scatter-add kernel: output[token_idx[i]] += result[i] * weight[i]
# ============================================================================
@triton.jit
def _scatter_add_weighted(
    Result_ptr,     # [total_tokens, H] fp32 — per-expert results
    Weight_ptr,     # [total_tokens] fp32 — routing weights
    TokenIdx_ptr,   # [total_tokens] int64 — original token indices
    Output_ptr,     # [T, H] fp32 — accumulated output
    total_tokens, H_val,
    stride_rm, stride_rh,
    stride_om, stride_oh,
    BLOCK_H: tl.constexpr,
):
    """One program per (token_in_result, h_block)."""
    pid_tok = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_tok >= total_tokens:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H_val

    # Load result row
    r_ptrs = Result_ptr + pid_tok * stride_rm + offs_h * stride_rh
    r = tl.load(r_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    # Load weight
    w = tl.load(Weight_ptr + pid_tok).to(tl.float32)

    # Load target token index
    tok_idx = tl.load(TokenIdx_ptr + pid_tok)

    # Atomic add to output
    o_ptrs = Output_ptr + tok_idx * stride_om + offs_h * stride_oh
    tl.atomic_add(o_ptrs, r * w, mask=mask_h)


# ============================================================================
# Routing (PyTorch — small relative to GEMMs, not worth a Triton kernel yet)
# ============================================================================
def deepseek_v3_routing(
    routing_logits: torch.Tensor,  # [T, E_global]
    routing_bias: torch.Tensor,    # [E_global]
    routed_scaling_factor: float,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
):
    """
    Returns:
        topk_idx: [T, top_k] int64 — selected expert indices
        topk_weights: [T, top_k] float32 — normalized routing weights
    """
    T, E = routing_logits.shape
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E // n_group
    s_wb_grouped = s_with_bias.view(T, n_group, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)  # [T, n_group]

    _, group_idx = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, n_group, group_size).reshape(T, E)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=top_k, dim=1, largest=True, sorted=False)

    # Weights from s (no bias), normalized
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # Gather per-topk weights
    topk_weights = torch.gather(weights, 1, topk_idx)  # [T, top_k]

    return topk_idx, topk_weights


# ============================================================================
# Main entry point
# ============================================================================
@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    H = 7168
    I = 2048
    BLOCK = 128
    E_LOCAL = gemm1_weights.shape[0]
    E_GLOBAL = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8

    device = hidden_states.device

    # ------- 1. Routing -------
    topk_idx, topk_weights = deepseek_v3_routing(
        routing_logits, routing_bias, routed_scaling_factor,
    )  # [T, 8], [T, 8]

    # ------- 2. Build per-expert token lists -------
    local_start = int(local_expert_offset)
    local_end = local_start + E_LOCAL

    # Flatten topk_idx to find which (token, slot) pairs map to local experts
    flat_expert = topk_idx.reshape(-1)           # [T*TOP_K]
    flat_token = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    flat_weight_idx = torch.arange(T * TOP_K, device=device)

    # Filter to local experts
    local_mask = (flat_expert >= local_start) & (flat_expert < local_end)
    local_expert_flat = flat_expert[local_mask]        # global expert ids
    local_token_flat = flat_token[local_mask]          # original token indices
    local_weight_flat = topk_weights.reshape(-1)[local_mask]  # routing weights
    local_expert_local = local_expert_flat - local_start  # local expert indices [0, E_LOCAL)

    if local_expert_flat.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    # Sort by local expert for grouped processing
    sort_idx = torch.argsort(local_expert_local, stable=True)
    sorted_expert = local_expert_local[sort_idx]
    sorted_token = local_token_flat[sort_idx]
    sorted_weight = local_weight_flat[sort_idx]

    # Find expert boundaries
    expert_counts = torch.zeros(E_LOCAL, dtype=torch.int64, device=device)
    expert_counts.scatter_add_(0, sorted_expert.long(), torch.ones_like(sorted_expert, dtype=torch.int64))
    expert_offsets = torch.cumsum(expert_counts, dim=0)
    expert_starts = expert_offsets - expert_counts  # start index for each expert

    total_tokens = sorted_token.numel()

    # ------- 3. Dequant hidden_states and gather -------
    # hidden_states: [T, H] fp8, hidden_states_scale: [H//BLOCK, T] float32
    # Dequant: A[t, h] = hidden_states[t, h] * scale[h // BLOCK, t]
    # We do this in PyTorch then gather the needed tokens
    A_fp32 = hidden_states.float()  # [T, H]
    A_scale = hidden_states_scale.float()  # [H//BLOCK, T]
    A_scale_expanded = A_scale.t().unsqueeze(-1).expand(T, H // BLOCK, BLOCK).reshape(T, H)
    A_deq = A_fp32 * A_scale_expanded  # [T, H]

    # Gather tokens needed by local experts
    A_gathered = A_deq[sorted_token]  # [total_tokens, H]

    # ------- 4. Per-expert GEMM1 -> SwiGLU -> GEMM2 -------
    # Pre-dequantize weights (simpler first version — fused version later)
    # GEMM1 weights: [E_LOCAL, 2I, H] fp8, scale [E_LOCAL, 2I//BLOCK, H//BLOCK]
    W1_fp32 = gemm1_weights.float()
    S1 = gemm1_weights_scale.float()
    S1_exp = S1.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    W1 = W1_fp32 * S1_exp  # [E_LOCAL, 2I, H]

    # GEMM2 weights: [E_LOCAL, H, I] fp8, scale [E_LOCAL, H//BLOCK, I//BLOCK]
    W2_fp32 = gemm2_weights.float()
    S2 = gemm2_weights_scale.float()
    S2_exp = S2.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    W2 = W2_fp32 * S2_exp  # [E_LOCAL, H, I]

    # Process each expert
    result_buf = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        start = expert_starts[le].item()
        count = expert_counts[le].item()
        if count == 0:
            continue
        end = start + count

        A_e = A_gathered[start:end]  # [count, H]
        W1_e = W1[le]               # [2I, H]
        W2_e = W2[le]               # [H, I]

        # GEMM1
        G1 = A_e @ W1_e.t()  # [count, 2I]

        # SwiGLU
        gate = G1[:, :I]
        up = G1[:, I:]
        silu_up = up * torch.sigmoid(up)
        C = silu_up * gate  # [count, I]

        # GEMM2
        O = C @ W2_e.t()  # [count, H]

        result_buf[start:end] = O

    # ------- 5. Weighted scatter-add -------
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Use Triton scatter-add for atomic accumulation
    BLOCK_H = 128
    grid_scatter = (total_tokens, triton.cdiv(H, BLOCK_H))
    _scatter_add_weighted[grid_scatter](
        result_buf, sorted_weight, sorted_token, output,
        total_tokens, H,
        result_buf.stride(0), result_buf.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_H=BLOCK_H,
    )

    return output.to(torch.bfloat16)