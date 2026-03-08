"""
Reference MoE implementation for moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.

This is the correctness ground truth. It is NOT optimized — it pre-dequantizes all
weights upfront (allocating ~3.5 GB) and runs a Python for-loop over 32 experts
(~200 kernel launches total). Use it only for correctness validation.

Pipeline:
  1. FP8 block-scale dequantization of hidden states + all expert weights
  2. DeepSeek-V3 no-aux routing: sigmoid → group-top2 → top-4 groups → global top-8
  3. Per-expert: GEMM1 [Tk,H]@[H,2I] → SwiGLU → GEMM2 [Tk,I]@[I,H]
  4. Weighted scatter-add: output[token_i] += routing_weight * expert_output
"""

import torch


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,       # [T, E_global] float — router scores before sigmoid
    routing_bias: torch.Tensor,         # [E_global] float — per-expert bias (load balancing)
    hidden_states: torch.Tensor,        # [T, H] fp8 e4m3 — input token embeddings
    hidden_states_scale: torch.Tensor,  # [H//128, T] float — block scales (H-dim is outer!)
    gemm1_weights: torch.Tensor,        # [E_local, 2I, H] fp8 — gate+up projections packed
    gemm1_weights_scale: torch.Tensor,  # [E_local, 2I//128, H//128] float
    gemm2_weights: torch.Tensor,        # [E_local, H, I] fp8 — down projection
    gemm2_weights_scale: torch.Tensor,  # [E_local, H//128, I//128] float
    local_expert_offset: int,           # first global expert index on this GPU rank
    routed_scaling_factor: float,       # multiplier applied to normalized routing weights
):
    """
    Returns [T, H]x bfloat16 — MoE FFN output, with contributions from all local experts
    accumulated via weighted scatter-add.

    Multi-GPU note: 256 global experts are split across ranks. This kernel computes only
    the E_local=32 experts owned by this rank (experts [local_expert_offset, +32)).
    Callers are responsible for cross-rank AllReduce after all ranks return.
    """

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]

    BLOCK = 128  # FP8 block-scale granularity: one scale per 128 consecutive elements
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168"
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8        # experts selected per token globally
    N_GROUP = 8      # number of expert groups (256 experts / 8 groups = 32 per group)
    TOPK_GROUP = 4   # groups kept per token before selecting top-8 experts

    # Block counts (used in shape assertions)
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # -------------------------------------------------------------------------
    # 1) FP8 block-scale dequantization
    #
    # Each FP8 value is stored as: float_val ≈ fp8_val * scale
    # where scale is shared across a block of 128 consecutive elements.
    #
    # Hidden state scale layout: [H//128, T] — H-block index is the OUTER dim.
    # This is transposed vs. the logical [T, H//128] layout. The reason is GEMM
    # tile access: for a fixed K-block (K = H here), all T token scales are
    # contiguous in memory → single coalesced load per tile iteration.
    # -------------------------------------------------------------------------

    # Dequantize hidden states: expand scale from [H//128, T] → [T, H]
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                # [H//128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()               # [T, H//128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                       # [T, H//128, 128]
        .reshape(T, H)                                             # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                                  # [T, H] float32

    # Dequantize GEMM1 weights (gate + up projections packed): [E, 2I, H]
    # Scale layout: [E, (2I)//128, H//128] — standard [N//128, K//128] tile order
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)      # [E, 2I, H//128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded                                  # [E, 2I, H] float32

    # Dequantize GEMM2 weights (down projection): [E, H, I]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)        # [E, H, I//128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)    # [E, H, I]
    W2 = W2_fp32 * S2_expanded                                     # [E, H, I] float32

    # -------------------------------------------------------------------------
    # 2) DeepSeek-V3 no-aux routing
    #
    # Two-level selection: first pick top-4 groups (coarse), then top-8 experts
    # within those groups (fine). Bias is added during selection to balance load
    # across experts, but is NOT used when computing the final routing weights
    # (weights should reflect true relevance, not load-balancing adjustments).
    # -------------------------------------------------------------------------

    logits = routing_logits.to(torch.float32)                      # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [E_global]

    # Step 2a: sigmoid scores (bounded, positive) + per-expert bias
    s = 1.0 / (1.0 + torch.exp(-logits))                          # [T, E] ∈ (0,1)
    s_with_bias = s + bias                                         # [T, E]

    # Step 2b: group-level coarse selection
    # Split 256 experts into 8 groups of 32; score each group by sum of its top-2
    group_size = E_global // N_GROUP                               # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)        # [T, 8, 32]
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                            # [T, 8]

    # Keep only top-4 groups; zero out experts in the other 4 groups
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)                    # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    # Expand group mask to expert level: [T, 8] → [T, 8, 32] → [T, 256]
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    # Step 2c: global top-8 within the 128 kept experts (4 groups × 32)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    # topk_idx: [T, 8] — global expert indices selected for each token

    # Step 2d: compute routing weights from unbiased sigmoid scores
    # Normalize over the 8 selected experts, then scale
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)                                   # binary mask [T, E]
    weights = s * M                                                 # zero out non-selected
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor      # [T, E] normalized

    # -------------------------------------------------------------------------
    # 3) Per-expert: GEMM1 → SwiGLU → GEMM2
    #
    # For each local expert, find tokens assigned to it, run the FFN, and
    # accumulate weighted results into the output buffer.
    #
    # SwiGLU: W13 packs gate (W1) and up-projection (W3) into [2I, H].
    #   GEMM1 output [Tk, 2I] is split: first I cols = gate, last I cols = up.
    #   output = silu(up) * gate  where silu(x) = x·σ(x)
    # -------------------------------------------------------------------------

    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le  # global expert index
        if ge < 0 or ge >= E_global:
            continue

        # Find tokens that selected this expert (anywhere in their top-8)
        sel_mask_per_token = (topk_idx == ge).any(dim=1)           # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]

        # Gather inputs for this expert's tokens
        A_e = A.index_select(0, token_idx)                         # [Tk, H]
        W13_e = W13[le]                                            # [2I, H]
        W2_e = W2[le]                                              # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())                                 # [Tk, 2I]

        # SwiGLU activation: split gate/up, apply silu(up) * gate
        X1 = G1[:, :I]                                             # gate [Tk, I]
        X2 = G1[:, I:]                                             # up   [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))                     # silu(up) = x·σ(x)
        C = silu_X2 * X1                                           # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = C.matmul(W2_e.t())                                     # [Tk, H]

        # Weighted scatter-add: output[token_i] += routing_weight * expert_output
        # Each token may appear in up to 8 experts, so we accumulate (not overwrite)
        w_tok = weights.index_select(0, token_idx)[:, ge]          # [Tk] scalar weight per token
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)