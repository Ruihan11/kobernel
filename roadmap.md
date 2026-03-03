# Fused MoE Kernel Optimization Roadmap

## Project Overview

Competition: FlashInfer AI Kernel Generation Contest @ MLSys 2026
Track: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
Target: NVIDIA B200 (Blackwell)

The kernel implements a fused Mixture-of-Experts layer with:
- DeepSeek-V3 no-aux routing (sigmoid → group-top2 → top-8 global)
- FP8 E4M3 block-scale dequantization (128-element blocks)
- GEMM1: `[T, 7168] × [4096, 7168]^T → [T, 4096]`
- SwiGLU activation
- GEMM2: `[T, 2048] × [7168, 2048]^T → [T, 7168]`
- Weighted scatter-add accumulation

---

## Workflow

### 1. Edit kernel

- Reference (ground truth): `solution/python/kernel.py::kernel`
- Triton implementation: `solution/triton/kernel.py::kernel`
- CUDA implementation: `solution/cuda/kernel.cu` + `solution/cuda/binding.py`

### 2. Test correctness

```bash
# Reference self-check
python tests/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py \
    --ref "solution/python/kernel.py::kernel" \
    --test "solution/python/kernel.py::kernel"

# Triton vs reference
python tests/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py \
    --ref "solution/python/kernel.py::kernel" \
    --test "solution/triton/kernel.py::kernel" \
    --device cuda

# Sweep across batch sizes
python tests/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py \
    --ref "solution/python/kernel.py::kernel" \
    --test "solution/triton/kernel.py::kernel" \
    --device cuda --T 1 4 16 64 256
```

Output format:
```
T=   1  ✅  max_abs=0.0000e+00  cos=1.00000000  close=100.0%
T=  16  ✅  max_abs=3.2100e-03  cos=0.99998712  close=99.8%
T=  64  ❌  max_abs=1.5000e-01  cos=0.99812000  close=94.2%
```

Tolerances: `atol=1e-2`, cosine ≥ 0.999 (FP8 quantization noise is expected).

### 3. Profile on B200

```bash
# Profile reference kernel
modal run scripts/run_modal_profiling.py \
    --function run-python-kernel --tokens 16 --print-rows 20

# Profile triton kernel
modal run scripts/run_modal_profiling.py \
    --function run-triton-kernel --tokens 16 --print-rows 20

# Deploy TensorBoard for trace visualization
modal deploy scripts/run_modal_profiling.py
```

Traces are saved to Modal Volume `moe-profiling-traces` and also downloaded locally to `/tmp/`. Open in [Perfetto UI](https://ui.perfetto.dev) or TensorBoard.

What to look for in traces:
- **GPU idle gaps** between kernel launches → Python overhead, need fusion
- **Kernel launch count** → fewer is better, each launch costs ~5-15μs
- **Memory allocation events** → intermediate tensors that should live in registers/smem
- **GEMM utilization** → small M (few tokens per expert) underutilizes tensor cores

### 4. Benchmark on B200

```bash
# Official benchmark
python scripts/pack_solution.py
modal run scripts/run_modal.py
```

### 5. Iterate

Edit → Test → Profile → Benchmark → Repeat.

---

## Optimization Roadmap

| | Phase | Tool | What | Key Win |
|---|-------|------|------|---------|
| ✅ | 0. Baseline | PyTorch | Reference impl, 200+ kernel launches | Correctness ground truth |
| ✅ | 1. Skeleton | Triton + PyTorch | Token sort by expert, Triton scatter-add, still `torch.matmul` | Structured permutation |
| ⬜ | 2. Fused FP8 GEMM | Triton | Fuse block-scale dequant into GEMM K-loop (`BLOCK_K=128=BLOCK_SCALE`) | Eliminate pre-dequant tensors (~2GB) |
| ⬜ | 3. Fuse SwiGLU | Triton | Compute SwiGLU in registers in GEMM1 epilogue | Half mem traffic between GEMM1→GEMM2 |
| ⬜ | 4. Single grid | Triton | One kernel grid over all experts, binary-search expert boundaries | 32×2 launches → 2 launches |
| ⬜ | 5. Fuse routing | Triton | Triton kernel for sigmoid→group-topk→global-topk→permutation | Remove PyTorch routing overhead |
| ⬜ | 6. HW-specific | CUDA/CUTLASS/CuTe | TMA, warp specialization, persistent kernel, GEMM2 epilogue fused scatter-add | Peak Blackwell utilization |
| ⬜ | 7. Tune | Triton/CUDA | Autotune tile sizes, pad expert batches, L2 cache reorder, stream overlap | Squeeze last 10-30% |

---
