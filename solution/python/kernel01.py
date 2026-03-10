"""
Fused MoE kernel — Python entry point with inline CUDA JIT compilation.

language = "python" in config.toml, so flashinfer-bench treats this as pure Python.
On first call, we JIT-compile the CUDA kernels via torch.utils.cpp_extension.
If compilation fails (missing nvcc, wrong arch), falls back to optimized PyTorch.
"""

import torch
import torch.nn.functional as F
import os
import tempfile

# =============================================================================
# Inline CUDA source
# =============================================================================
_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

// FP8 E4M3 -> float
__device__ __forceinline__ float fp8_to_f32(__nv_fp8_e4m3 v) {
    return float(v);
}

__device__ __forceinline__ float load_fp8(const void* ptr, int idx) {
    __nv_fp8_e4m3 v;
    reinterpret_cast<const unsigned char*>(ptr)[0];
    unsigned char raw = ((const unsigned char*)ptr)[idx];
    memcpy(&v, &raw, 1);
    return float(v);
}

// =========================================================================
// Kernel 1: Fused GEMM1 + SwiGLU
// Each thread computes one element of the SwiGLU output [total_tokens, I]
// Grid: (ceil(I/BN), ceil(total_tokens/BM))
// Block: (BN, BM) where BN=32, BM=4
// =========================================================================
constexpr int BN1 = 32;
constexpr int BM1 = 4;
constexpr int H_DIM = 7168;
constexpr int I_DIM = 2048;
constexpr int TWO_I = 4096;
constexpr int BS = 128;  // block scale granularity

__global__ void gemm1_swiglu_kernel(
    const void*    __restrict__ hidden_fp8,      // [T_orig, H] fp8
    const float*   __restrict__ hidden_scale,    // [H/128, T_orig]
    const void*    __restrict__ w1_fp8,          // [E, 2I, H] fp8
    const float*   __restrict__ w1_scale,        // [E, 2I/128, H/128]
    const int64_t* __restrict__ sorted_token,    // [total]
    const int32_t* __restrict__ expert_starts,   // [E]
    const int32_t* __restrict__ expert_counts,   // [E]
    float*         __restrict__ out,             // [total, I]
    int T_orig, int total, int E_local
) {
    int col = blockIdx.x * BN1 + threadIdx.x;  // output col in [0, I)
    int row = blockIdx.y * BM1 + threadIdx.y;  // row in sorted token list
    
    if (col >= I_DIM || row >= total) return;
    
    // Find expert for this row
    int expert_id = 0;
    for (int e = 0; e < E_local; e++) {
        if (row < expert_starts[e] + expert_counts[e]) {
            expert_id = e;
            break;
        }
        if (e + 1 < E_local && row < expert_starts[e + 1]) {
            expert_id = e;
            break;
        }
    }
    // Simpler: linear scan
    expert_id = 0;
    {
        int acc = 0;
        for (int e = 0; e < E_local; e++) {
            acc += expert_counts[e];
            if (row < acc) { expert_id = e; break; }
        }
    }
    
    int token_id = sorted_token[row];
    
    // gate column = col, up column = col + I
    int gate_col = col;
    int up_col = col + I_DIM;
    
    int num_k_blocks = H_DIM / BS;  // 56
    int num_n_blocks = TWO_I / BS;  // 32
    
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    
    for (int kb = 0; kb < num_k_blocks; kb++) {
        float a_sc = hidden_scale[kb * T_orig + token_id];
        float w_sc_gate = w1_scale[expert_id * num_n_blocks * num_k_blocks + (gate_col / BS) * num_k_blocks + kb];
        float w_sc_up   = w1_scale[expert_id * num_n_blocks * num_k_blocks + (up_col / BS) * num_k_blocks + kb];
        
        float dot_gate = 0.0f, dot_up = 0.0f;
        int k0 = kb * BS;
        
        #pragma unroll 4
        for (int kk = 0; kk < BS; kk++) {
            int k = k0 + kk;
            float a = load_fp8(hidden_fp8, token_id * H_DIM + k);
            float wg = load_fp8(w1_fp8, (size_t)expert_id * TWO_I * H_DIM + (size_t)gate_col * H_DIM + k);
            float wu = load_fp8(w1_fp8, (size_t)expert_id * TWO_I * H_DIM + (size_t)up_col * H_DIM + k);
            dot_gate += a * wg;
            dot_up += a * wu;
        }
        
        acc_gate += dot_gate * a_sc * w_sc_gate;
        acc_up   += dot_up * a_sc * w_sc_up;
    }
    
    // SwiGLU: silu(up) * gate
    float silu_up = acc_up / (1.0f + expf(-acc_up));
    out[row * I_DIM + col] = silu_up * acc_gate;
}

// =========================================================================
// Kernel 2: Fused GEMM2 + Weighted Scatter-Add
// Each thread computes one element of [total_tokens, H], then atomic adds
// Grid: (ceil(H/BN), ceil(total/BM))
// Block: (BN, BM)
// =========================================================================
constexpr int BN2 = 32;
constexpr int BM2 = 4;

__global__ void gemm2_scatter_kernel(
    const float*   __restrict__ swiglu_in,       // [total, I] fp32
    const void*    __restrict__ w2_fp8,          // [E, H, I] fp8
    const float*   __restrict__ w2_scale,        // [E, H/128, I/128]
    const int64_t* __restrict__ sorted_token,
    const float*   __restrict__ sorted_weight,   // [total]
    const int32_t* __restrict__ expert_starts,
    const int32_t* __restrict__ expert_counts,
    float*         __restrict__ output,          // [T_orig, H]
    int total, int E_local
) {
    int col = blockIdx.x * BN2 + threadIdx.x;  // H dim
    int row = blockIdx.y * BM2 + threadIdx.y;
    
    if (col >= H_DIM || row >= total) return;
    
    // Find expert
    int expert_id = 0;
    {
        int acc = 0;
        for (int e = 0; e < E_local; e++) {
            acc += expert_counts[e];
            if (row < acc) { expert_id = e; break; }
        }
    }
    
    int token_id = sorted_token[row];
    float rw = sorted_weight[row];
    
    int num_k_blocks = I_DIM / BS;  // 16
    int num_n_blocks = H_DIM / BS;  // 56
    
    float acc = 0.0f;
    
    for (int kb = 0; kb < num_k_blocks; kb++) {
        float w_sc = w2_scale[expert_id * num_n_blocks * num_k_blocks + (col / BS) * num_k_blocks + kb];
        
        float dot = 0.0f;
        int k0 = kb * BS;
        
        #pragma unroll 4
        for (int kk = 0; kk < BS; kk++) {
            int k = k0 + kk;
            float a = swiglu_in[row * I_DIM + k];
            float w = load_fp8(w2_fp8, (size_t)expert_id * H_DIM * I_DIM + (size_t)col * I_DIM + k);
            dot += a * w;
        }
        acc += dot * w_sc;
    }
    
    atomicAdd(&output[token_id * H_DIM + col], acc * rw);
}

// =========================================================================
// Entry point
// =========================================================================
torch::Tensor moe_cuda(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights,
    torch::Tensor gemm1_weights_scale,
    torch::Tensor gemm2_weights,
    torch::Tensor gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
    const int T = hidden_states.size(0);
    const int E_LOCAL = 32;
    const int E_GLOBAL = 256;
    const int TOP_K = 8;
    auto dev = hidden_states.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    // ---- Routing (PyTorch) ----
    auto s = torch::sigmoid(routing_logits.to(torch::kFloat32));
    auto sb = s + routing_bias.to(torch::kFloat32).view({-1});
    auto gr = sb.view({T, 8, 32});
    auto t2 = std::get<0>(gr.topk(2, 2));
    auto gs = t2.sum(2);
    auto gi = std::get<1>(gs.topk(4, 1));
    auto gm = torch::zeros_like(gs).scatter_(1, gi, 1.0);
    auto sm = gm.unsqueeze(2).expand({T, 8, 32}).reshape({T, E_GLOBAL});
    auto pr = sb.masked_fill(sm == 0, -1e38f);
    auto ti = std::get<1>(pr.topk(TOP_K, 1));
    auto mm = torch::zeros_like(s).scatter_(1, ti, 1.0);
    auto wt = s * mm;
    wt = (wt / (wt.sum(1, true) + 1e-20)) * routed_scaling_factor;
    
    // ---- Permutation ----
    int ls = (int)local_expert_offset;
    auto fe = ti.reshape({-1});
    auto ft = torch::arange(T, torch::kLong, dev).unsqueeze(1).expand({T, TOP_K}).reshape({-1});
    auto lm = (fe >= ls) & (fe < ls + E_LOCAL);
    
    if (!lm.any().item<bool>())
        return torch::zeros({T, H_DIM}, torch::kBFloat16, dev);
    
    auto le = fe.index({lm}) - ls;
    auto lt = ft.index({lm});
    auto ge = fe.index({lm});
    auto lw = wt.index({lt, ge});
    
    auto si = torch::argsort(le, true);
    auto se = le.index({si});
    auto st = lt.index({si}).to(torch::kInt64);
    auto sw = lw.index({si}).to(torch::kFloat32);
    
    auto ec = torch::zeros({E_LOCAL}, torch::kInt32, dev);
    ec.scatter_add_(0, se.to(torch::kInt64), torch::ones(se.numel(), torch::kInt32, dev));
    auto es = (torch::cumsum(ec, 0) - ec).to(torch::kInt32);
    
    int total = st.numel();
    
    // ---- Kernel 1 ----
    auto swiglu_buf = torch::empty({total, I_DIM}, torch::kFloat32, dev);
    {
        dim3 grid((I_DIM + BN1 - 1) / BN1, (total + BM1 - 1) / BM1);
        dim3 block(BN1, BM1);
        gemm1_swiglu_kernel<<<grid, block, 0, stream>>>(
            hidden_states.data_ptr(),
            hidden_states_scale.data_ptr<float>(),
            gemm1_weights.data_ptr(),
            gemm1_weights_scale.data_ptr<float>(),
            st.data_ptr<int64_t>(),
            es.data_ptr<int32_t>(),
            ec.data_ptr<int32_t>(),
            swiglu_buf.data_ptr<float>(),
            T, total, E_LOCAL
        );
    }
    
    // ---- Kernel 2 ----
    auto output = torch::zeros({T, H_DIM}, torch::kFloat32, dev);
    {
        dim3 grid((H_DIM + BN2 - 1) / BN2, (total + BM2 - 1) / BM2);
        dim3 block(BN2, BM2);
        gemm2_scatter_kernel<<<grid, block, 0, stream>>>(
            swiglu_buf.data_ptr<float>(),
            gemm2_weights.data_ptr(),
            gemm2_weights_scale.data_ptr<float>(),
            st.data_ptr<int64_t>(),
            sw.data_ptr<float>(),
            es.data_ptr<int32_t>(),
            ec.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            total, E_LOCAL
        );
    }
    
    return output.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &moe_cuda);
}
'''

# =============================================================================
# JIT compile CUDA extension
# =============================================================================
_cuda_ext = None

def _get_cuda_ext():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext
    
    try:
        from torch.utils.cpp_extension import load_inline
        _cuda_ext = load_inline(
            name="moe_fused",
            cpp_sources="",
            cuda_sources=[_CUDA_SRC],
            functions=["kernel"],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    except Exception as e:
        print(f"[moe_kernel] CUDA JIT compilation failed: {e}")
        print("[moe_kernel] Falling back to PyTorch implementation")
        _cuda_ext = None
    
    return _cuda_ext


# =============================================================================
# PyTorch fallback (if CUDA compile fails)
# =============================================================================
def _pytorch_fallback(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    H, I, BLOCK = 7168, 2048, 128
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    # Dequant
    A = hidden_states.float() * hidden_states_scale.float().t().repeat_interleave(BLOCK, 1)
    W13 = gemm1_weights.float() * gemm1_weights_scale.float().repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    W2 = gemm2_weights.float() * gemm2_weights_scale.float().repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)

    # Routing
    s = torch.sigmoid(routing_logits.float())
    sb = s + routing_bias.float().view(-1)
    gr = sb.view(T, 8, 32)
    gs = gr.topk(2, 2).values.sum(2)
    gi = gs.topk(4, 1).indices
    gm = torch.zeros_like(gs).scatter_(1, gi, 1.0)
    sm = gm.unsqueeze(2).expand(T, 8, 32).reshape(T, E_global)
    pr = sb.masked_fill(sm == 0, torch.finfo(torch.float32).min)
    ti = pr.topk(8, 1).indices
    mm = torch.zeros_like(s).scatter_(1, ti, 1.0)
    wt = s * mm
    wt = (wt / (wt.sum(1, True) + 1e-20)) * routed_scaling_factor

    # Permute
    ls = int(local_expert_offset)
    fe = ti.reshape(-1)
    ft = torch.arange(T, device=device).unsqueeze(1).expand(T, 8).reshape(-1)
    lm = (fe >= ls) & (fe < ls + E_local)
    if not lm.any():
        return torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    le = fe[lm] - ls
    lt = ft[lm]
    lw = wt[lt, fe[lm]]
    si = torch.argsort(le, stable=True)
    se, st_t, sw = le[si], lt[si], lw[si]
    ec = torch.zeros(E_local, dtype=torch.long, device=device)
    ec.scatter_add_(0, se.long(), torch.ones_like(se, dtype=torch.long))
    es = (ec.cumsum(0) - ec)
    
    A_s = A[st_t]
    total = st_t.numel()
    result = torch.empty(total, H, dtype=torch.float32, device=device)
    starts = es.tolist()
    counts = ec.tolist()
    for e in range(E_local):
        c = counts[e]
        if c == 0: continue
        s_ = starts[e]
        sl = slice(s_, s_ + c)
        G1 = A_s[sl] @ W13[e].t()
        result[sl] = (F.silu(G1[:, I:]) * G1[:, :I]) @ W2[e].t()
    
    output = torch.zeros(T, H, dtype=torch.float32, device=device)
    output.index_add_(0, st_t, result * sw.unsqueeze(1))
    return output.to(torch.bfloat16)


# =============================================================================
# Entry point
# =============================================================================
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
    ext = _get_cuda_ext()
    
    if ext is not None:
        return ext.kernel(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
        )
    else:
        return _pytorch_fallback(
            routing_logits, routing_bias,
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
        )