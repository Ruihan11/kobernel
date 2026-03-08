/*
 * Optimized CUDA kernels for fused MoE on B200/Blackwell.
 *
 * Key optimizations over baseline:
 *   P0: On-the-fly per-expert weight dequant (eliminates ~3.5GB intermediate)
 *   P1: Expert loop in C++ (eliminates Python loop overhead, better GPU pipelining)
 *   GEMMs via cuBLAS with reusable float32 weight buffers.
 *
 * Compiled via torch.utils.cpp_extension.load() from binding.py.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

// ========================= FP8 E4M3 helper =========================

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    int sign      = (bits >> 7) & 1;
    int exp_bits  = (bits >> 3) & 0xF;
    int mant_bits = bits & 0x7;

    float val;
    if (exp_bits == 0) {
        // Subnormal: (-1)^s * 2^(-6) * (mant / 8)
        val = ldexpf((float)mant_bits / 8.0f, -6);
    } else if (exp_bits == 0xF && mant_bits == 0x7) {
        val = nanf("");
    } else {
        // Normal: (-1)^s * 2^(exp-7) * (1 + mant/8)
        val = ldexpf(1.0f + (float)mant_bits / 8.0f, exp_bits - 7);
    }
    if (sign) val = -val;
    return val;
}

// ========================= CUDA Kernels =========================

// Fused FP8 dequant + token gather for hidden states.
// Avoids materializing the full [T, H] dequantized activation tensor.
// output[n, h] = fp8_data[token_idx[n], h] * scale[h // BS, token_idx[n]]
__global__ void dequant_gather_kernel(
    const uint8_t* __restrict__ fp8_data,  // [T, H] raw fp8 bytes
    const float* __restrict__ scale,       // [H//BS, T]
    const int64_t* __restrict__ token_idx, // [N]
    float* __restrict__ output,            // [N, H]
    int N, int H, int T, int BS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) return;

    int n = idx / H;
    int h = idx % H;
    int tok = (int)token_idx[n];
    int h_block = h / BS;

    float val = fp8_e4m3_to_float(fp8_data[tok * H + h]);
    float s = scale[h_block * T + tok];  // [H//BS, T] layout
    output[idx] = val * s;
}

// Dequant one expert's weight matrix: [N_out, N_in] fp8 -> float32
// scale layout: [N_out//BS, N_in//BS], row-major
__global__ void dequant_weight_kernel(
    const uint8_t* __restrict__ fp8_data,  // [N_out, N_in] fp8
    const float* __restrict__ scale,       // [N_out//BS, N_in//BS]
    float* __restrict__ output,            // [N_out, N_in] float32
    int N_out, int N_in, int scale_cols, int BS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_out * N_in) return;

    int row = idx / N_in;
    int col = idx % N_in;

    float val = fp8_e4m3_to_float(fp8_data[idx]);
    float s = scale[(row / BS) * scale_cols + (col / BS)];
    output[idx] = val * s;
}

// SwiGLU: output[m, i] = silu(input[m, I+i]) * input[m, i]
// Single fused kernel avoids materializing silu intermediate.
__global__ void swiglu_kernel(
    const float* __restrict__ input,   // [M, 2*I]
    float* __restrict__ output,        // [M, I]
    int M, int I
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * I) return;

    int m = idx / I;
    int i = idx % I;
    int base = m * 2 * I;

    float gate = input[base + i];
    float up   = input[base + I + i];
    float silu_up = up / (1.0f + expf(-up));
    output[idx] = silu_up * gate;
}

// Weighted scatter-add: output[token_idx[t], h] += result[t, h] * weight[t]
// 2D grid: (total_tokens, ceil(H/256)); uses atomicAdd for multi-expert overlap.
__global__ void scatter_add_weighted_kernel(
    const float* __restrict__ result,      // [total_tokens, H]
    const float* __restrict__ weights,     // [total_tokens]
    const int64_t* __restrict__ token_idx, // [total_tokens]
    float* __restrict__ output,            // [T, H]
    int total_tokens, int H
) {
    int tok = blockIdx.x;
    int h = blockIdx.y * blockDim.x + threadIdx.x;

    if (tok >= total_tokens || h >= H) return;

    float val = result[tok * H + h] * weights[tok];
    int out_row = (int)token_idx[tok];
    atomicAdd(&output[out_row * H + h], val);
}

// ========================= C++ entry points =========================

torch::Tensor dequant_gather_forward(
    torch::Tensor fp8_data,    // [T, H] fp8_e4m3fn
    torch::Tensor scale,       // [H//BS, T] float32
    torch::Tensor token_idx,   // [N] int64
    int64_t H, int64_t T, int64_t BS
) {
    int N = token_idx.size(0);
    auto output = torch::empty({N, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(fp8_data.device()));

    if (N == 0) return output;

    int total = N * (int)H;
    int block = 256;
    int grid = (total + block - 1) / block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dequant_gather_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)fp8_data.data_ptr(),
        scale.data_ptr<float>(),
        token_idx.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N, (int)H, (int)T, (int)BS
    );
    return output;
}

// Grouped MoE forward: runs all experts in a single C++ call.
// Replaces the Python expert loop + weight pre-dequant.
//
// Memory savings: ~3.5GB (all experts pre-dequant) -> ~177MB (2 reusable buffers)
// Launch overhead: Python loop (32 iters * GIL) -> C++ loop (direct cuBLAS dispatch)
torch::Tensor grouped_moe_forward(
    torch::Tensor A_gathered,      // [total_tokens, H] float32
    torch::Tensor gemm1_weights,   // [E_LOCAL, 2*I, H] fp8
    torch::Tensor gemm1_scales,    // [E_LOCAL, 2*I/128, H/128] float32
    torch::Tensor gemm2_weights,   // [E_LOCAL, H, I] fp8
    torch::Tensor gemm2_scales,    // [E_LOCAL, H/128, I/128] float32
    torch::Tensor sorted_token,    // [total_tokens] int64
    torch::Tensor sorted_weight,   // [total_tokens] float32
    torch::Tensor expert_starts,   // [E_LOCAL] int64
    torch::Tensor expert_counts,   // [E_LOCAL] int64
    int64_t T, int64_t H, int64_t I_val
) {
    auto device = A_gathered.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    int E_LOCAL = gemm1_weights.size(0);
    int total_tokens = A_gathered.size(0);
    int BS = 128;
    int block_threads = 256;

    // Copy expert routing info to host (single sync point)
    auto starts_cpu = expert_starts.cpu();
    auto counts_cpu = expert_counts.cpu();
    auto* starts_ptr = starts_cpu.data_ptr<int64_t>();
    auto* counts_ptr = counts_cpu.data_ptr<int64_t>();

    int max_count = 0;
    for (int e = 0; e < E_LOCAL; e++) {
        max_count = std::max(max_count, (int)counts_ptr[e]);
    }

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    if (max_count == 0 || total_tokens == 0) {
        return torch::zeros({T, H}, opts_f32);
    }

    // Reusable buffers (allocated once, reused per expert)
    // W1_buf: [2I, H] = [4096, 7168] = 29.4M floats = 117.4 MB
    // W2_buf: [H, I]  = [7168, 2048] = 14.7M floats =  58.7 MB
    // Total reusable: ~177 MB vs ~3.5 GB for all experts pre-dequant
    auto W1_buf = torch::empty({2 * I_val, H}, opts_f32);
    auto W2_buf = torch::empty({H, I_val}, opts_f32);
    auto G1_buf = torch::empty({(int64_t)max_count, 2 * I_val}, opts_f32);
    auto swiglu_buf = torch::empty({(int64_t)max_count, I_val}, opts_f32);
    auto result_buf = torch::empty({(int64_t)total_tokens, H}, opts_f32);
    auto output = torch::zeros({T, H}, opts_f32);

    // Weight data pointers and per-expert strides
    const uint8_t* w1_base = (const uint8_t*)gemm1_weights.data_ptr();
    const float* s1_base = gemm1_scales.data_ptr<float>();
    const uint8_t* w2_base = (const uint8_t*)gemm2_weights.data_ptr();
    const float* s2_base = gemm2_scales.data_ptr<float>();

    int64_t w1_expert_bytes = 2 * I_val * H;            // fp8 = 1 byte each
    int64_t s1_expert_elems = (2 * I_val / BS) * (H / BS);
    int64_t w2_expert_bytes = H * I_val;
    int64_t s2_expert_elems = (H / BS) * (I_val / BS);

    int s1_cols = (int)(H / BS);       // scale columns for W1: H//128
    int s2_cols = (int)(I_val / BS);   // scale columns for W2: I//128

    float alpha = 1.0f, beta = 0.0f;

    for (int e = 0; e < E_LOCAL; e++) {
        int start = (int)starts_ptr[e];
        int count = (int)counts_ptr[e];
        if (count == 0) continue;

        // --- Dequant GEMM1 weights for expert e: [2I, H] fp8 -> float32 ---
        {
            int w1_total = (int)(2 * I_val * H);
            int w1_grid = (w1_total + block_threads - 1) / block_threads;
            dequant_weight_kernel<<<w1_grid, block_threads, 0, stream>>>(
                w1_base + e * w1_expert_bytes,
                s1_base + e * s1_expert_elems,
                W1_buf.data_ptr<float>(),
                (int)(2 * I_val), (int)H, s1_cols, BS
            );
        }

        // --- GEMM1: [count, H] @ [2I, H]^T -> [count, 2I] ---
        // Row-major C = A @ W^T  =>  col-major: C_cm = W_cm^T @ A_cm
        // W_cm is [H, 2I] with ld=H; op=T gives [2I, H]
        // A_cm is [H, count] with ld=H; op=N
        // C_cm is [2I, count] with ld=2I
        {
            float* A_ptr = A_gathered.data_ptr<float>() + (int64_t)start * H;
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)(2 * I_val),   // m
                count,              // n
                (int)H,             // k
                &alpha,
                W1_buf.data_ptr<float>(), (int)H,    // "A" = W1_cm, lda = H
                A_ptr, (int)H,                        // "B" = A_cm, ldb = H
                &beta,
                G1_buf.data_ptr<float>(), (int)(2 * I_val)  // C, ldc = 2I
            );
        }

        // --- SwiGLU: silu(up) * gate ---
        {
            int sg_total = count * (int)I_val;
            int sg_grid = (sg_total + block_threads - 1) / block_threads;
            swiglu_kernel<<<sg_grid, block_threads, 0, stream>>>(
                G1_buf.data_ptr<float>(),
                swiglu_buf.data_ptr<float>(),
                count, (int)I_val
            );
        }

        // --- Dequant GEMM2 weights for expert e: [H, I] fp8 -> float32 ---
        {
            int w2_total = (int)(H * I_val);
            int w2_grid = (w2_total + block_threads - 1) / block_threads;
            dequant_weight_kernel<<<w2_grid, block_threads, 0, stream>>>(
                w2_base + e * w2_expert_bytes,
                s2_base + e * s2_expert_elems,
                W2_buf.data_ptr<float>(),
                (int)H, (int)I_val, s2_cols, BS
            );
        }

        // --- GEMM2: [count, I] @ [H, I]^T -> [count, H] ---
        // Row-major C = S @ W2^T  =>  col-major: C_cm = W2_cm^T @ S_cm
        // W2_cm is [I, H] with ld=I; op=T gives [H, I]
        // S_cm is [I, count] with ld=I; op=N
        // C_cm is [H, count] with ld=H
        {
            float* result_ptr = result_buf.data_ptr<float>() + (int64_t)start * H;
            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)H,             // m
                count,              // n
                (int)I_val,         // k
                &alpha,
                W2_buf.data_ptr<float>(), (int)I_val, // "A" = W2_cm, lda = I
                swiglu_buf.data_ptr<float>(), (int)I_val,  // "B" = S_cm, ldb = I
                &beta,
                result_ptr, (int)H                    // C, ldc = H
            );
        }
    }

    // --- Weighted scatter-add: output[tok, :] += weight * result ---
    if (total_tokens > 0) {
        dim3 sa_grid(total_tokens, ((int)H + block_threads - 1) / block_threads);
        scatter_add_weighted_kernel<<<sa_grid, block_threads, 0, stream>>>(
            result_buf.data_ptr<float>(),
            sorted_weight.data_ptr<float>(),
            sorted_token.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            total_tokens, (int)H
        );
    }

    return output;
}

// ========================= pybind11 module =========================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_gather_forward", &dequant_gather_forward,
          "Fused FP8-E4M3 dequant + token gather (CUDA)");
    m.def("grouped_moe_forward", &grouped_moe_forward,
          "Grouped MoE: all experts in one C++ call with on-the-fly weight dequant");
}