/*
 * CUDA kernels for fused MoE: SwiGLU, weighted scatter-add, FP8 dequant+gather.
 *
 * Compiled via torch.utils.cpp_extension.load() from binding.py.
 * Exposes three Python-callable functions through pybind11.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ===========================================================================
// SwiGLU: output[m, i] = silu(input[m, I+i]) * input[m, i]
// Single fused kernel avoids materializing silu intermediate.
// ===========================================================================
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

torch::Tensor swiglu_forward(torch::Tensor input, int64_t I_val) {
    int M = input.size(0);
    auto output = torch::empty({M, I_val}, input.options());

    int total = M * (int)I_val;
    int block = 256;
    int grid = (total + block - 1) / block;

    swiglu_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        M, (int)I_val
    );
    return output;
}

// ===========================================================================
// Weighted scatter-add: output[token_idx[t], h] += result[t, h] * weight[t]
// 2D grid: (total_tokens, ceil(H/256)); uses atomicAdd for multi-expert overlap.
// ===========================================================================
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

void scatter_add_weighted_forward(
    torch::Tensor result, torch::Tensor weights,
    torch::Tensor token_idx, torch::Tensor output
) {
    int total_tokens = result.size(0);
    int H = result.size(1);

    int block = 256;
    dim3 grid(total_tokens, (H + block - 1) / block);

    scatter_add_weighted_kernel<<<grid, block>>>(
        result.data_ptr<float>(), weights.data_ptr<float>(),
        token_idx.data_ptr<int64_t>(), output.data_ptr<float>(),
        total_tokens, H
    );
}

// ===========================================================================
// Fused FP8-E4M3 dequant + gather:
//   output[n, h] = fp8_data[token_idx[n], h] * scale[h // BS, token_idx[n]]
// Avoids materializing the full [T, H] dequantized activation tensor.
// ===========================================================================

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

torch::Tensor dequant_gather_forward(
    torch::Tensor fp8_data,    // [T, H] fp8_e4m3fn
    torch::Tensor scale,       // [H//BS, T] float32
    torch::Tensor token_idx,   // [N] int64
    int64_t H, int64_t T, int64_t BS
) {
    int N = token_idx.size(0);
    auto output = torch::empty({N, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(fp8_data.device()));

    int total = N * (int)H;
    int block = 256;
    int grid = (total + block - 1) / block;

    dequant_gather_kernel<<<grid, block>>>(
        (const uint8_t*)fp8_data.data_ptr(),
        scale.data_ptr<float>(),
        token_idx.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N, (int)H, (int)T, (int)BS
    );
    return output;
}

// ===========================================================================
// pybind11 module
// ===========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward,
          "SwiGLU activation (CUDA)");
    m.def("scatter_add_weighted_forward", &scatter_add_weighted_forward,
          "Weighted scatter-add with atomicAdd (CUDA)");
    m.def("dequant_gather_forward", &dequant_gather_forward,
          "Fused FP8-E4M3 dequant + token gather (CUDA)");
}
