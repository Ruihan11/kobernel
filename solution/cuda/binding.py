"""
Python binding for CUDA MoE kernel.

Compiles kernel.cu via torch.utils.cpp_extension.load() and provides the
kernel() entry point. Routing and weight dequant run in PyTorch; SwiGLU,
scatter-add, and activation dequant+gather use custom CUDA kernels.

Test:
    python tests/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py \
        --ref "solution/python/kernel00.py::kernel" \
        --test "solution/cuda/binding.py::kernel" \
        --device cuda --T 1 4 16 64 256
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# ============================================================================
# Compile kernel.cu
# ============================================================================
_cuda_ops = None
_KERNEL_DIR = Path(__file__).parent


def _get_ops():
    global _cuda_ops
    if _cuda_ops is None:
        _cuda_ops = load(
            name="moe_cuda_ops",
            sources=[str(_KERNEL_DIR / "kernel.cu")],
            verbose=(os.environ.get("MOE_CUDA_VERBOSE", "0") == "1"),
        )
    return _cuda_ops


# ============================================================================
# DeepSeek-V3 routing (PyTorch)
# ============================================================================
def deepseek_v3_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
):
    T, E = routing_logits.shape
    logits = routing_logits.float()
    bias = routing_bias.float().reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E // n_group
    s_wb_grouped = s_with_bias.view(T, n_group, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, n_group, group_size).reshape(T, E)
    )

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=top_k, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    topk_weights = torch.gather(weights, 1, topk_idx)
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
    output: torch.Tensor = None,
):
    H = 7168
    I = 2048
    BLOCK = 128
    E_LOCAL = gemm1_weights.shape[0]
    T = routing_logits.shape[0]
    TOP_K = 8

    device = hidden_states.device
    ops = _get_ops()

    # ------- 1. Routing -------
    topk_idx, topk_weights = deepseek_v3_routing(
        routing_logits, routing_bias, routed_scaling_factor,
    )

    # ------- 2. Build per-expert token lists -------
    local_start = int(local_expert_offset)
    local_end = local_start + E_LOCAL

    flat_expert = topk_idx.reshape(-1)
    flat_token = (
        torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    )

    local_mask = (flat_expert >= local_start) & (flat_expert < local_end)
    local_expert_flat = flat_expert[local_mask]
    local_token_flat = flat_token[local_mask]
    local_weight_flat = topk_weights.reshape(-1)[local_mask]
    local_expert_local = local_expert_flat - local_start

    if local_expert_flat.numel() == 0:
        result = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
        if output is not None:
            output.copy_(result)
            return
        return result

    # Sort by expert for grouped processing
    sort_idx = torch.argsort(local_expert_local, stable=True)
    sorted_expert = local_expert_local[sort_idx]
    sorted_token = local_token_flat[sort_idx]
    sorted_weight = local_weight_flat[sort_idx]

    # Expert boundaries
    expert_counts = torch.zeros(E_LOCAL, dtype=torch.int64, device=device)
    expert_counts.scatter_add_(
        0, sorted_expert.long(),
        torch.ones_like(sorted_expert, dtype=torch.int64),
    )
    expert_starts = torch.cumsum(expert_counts, dim=0) - expert_counts
    total_tokens = sorted_token.numel()

    # ------- 3. Fused FP8 dequant + gather (CUDA) -------
    A_gathered = ops.dequant_gather_forward(
        hidden_states, hidden_states_scale.float(),
        sorted_token, H, T, BLOCK,
    )  # [total_tokens, H] float32

    # ------- 4. Pre-dequantize weights (PyTorch) -------
    W1_fp32 = gemm1_weights.float()
    S1 = gemm1_weights_scale.float()
    S1_exp = S1.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    W1 = W1_fp32 * S1_exp  # [E_LOCAL, 2I, H]

    W2_fp32 = gemm2_weights.float()
    S2 = gemm2_weights_scale.float()
    S2_exp = S2.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    W2 = W2_fp32 * S2_exp  # [E_LOCAL, H, I]

    # ------- 5. Per-expert GEMM1 -> SwiGLU (CUDA) -> GEMM2 -------
    result_buf = torch.empty((total_tokens, H), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        start = expert_starts[le].item()
        count = expert_counts[le].item()
        if count == 0:
            continue
        end = start + count

        A_e = A_gathered[start:end]   # [count, H]
        W1_e = W1[le]                 # [2I, H]
        W2_e = W2[le]                 # [H, I]

        # GEMM1: cuBLAS
        G1 = A_e @ W1_e.t()  # [count, 2I]

        # SwiGLU: CUDA kernel
        C = ops.swiglu_forward(G1, I)  # [count, I]

        # GEMM2: cuBLAS
        result_buf[start:end] = C @ W2_e.t()  # [count, H]

    # ------- 6. Weighted scatter-add (CUDA) -------
    out_buf = torch.zeros((T, H), dtype=torch.float32, device=device)
    ops.scatter_add_weighted_forward(result_buf, sorted_weight, sorted_token, out_buf)

    result = out_buf.to(torch.bfloat16)
    if output is not None:
        output.copy_(result)
        return
    return result