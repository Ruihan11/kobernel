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

## Fused MoE: What Fusion Actually Means

A naive MoE pass goes HBM → op → HBM → op → HBM for every intermediate. Fusion keeps data in registers or shared memory across op boundaries, eliminating round-trips:

| Fusion Boundary | What Stays On-Chip | HBM Traffic Eliminated |
|---|---|---|
| FP8 dequant → GEMM | Scale × tile result in registers | Pre-dequantized weight tensors (~2GB) |
| GEMM1 epilogue → SwiGLU | Output tile before store | GEMM1 output buffer |
| GEMM2 epilogue → scatter-add | Output row before store | GEMM2 output buffer |
| All experts in one grid | Expert metadata, token indices | 32×2 kernel launch overhead |

The GEMM epilogue is the natural fusion point: each output tile is computed entirely in registers before the store instruction. Elementwise ops (SwiGLU, scale, accumulate) applied *before* the store are "free" in terms of memory traffic.

## Triton vs CUDA for This Project

**Triton teaches**: tile-based thinking, memory hierarchy intuition, how fused algorithms are structured — all directly transferable.

**Triton hides**: explicit shared memory, warp-level primitives (`__shfl_sync`, WMMA), TMA (Tensor Memory Accelerator), WGMMA (warpgroup MMA on Hopper/Blackwell), CTA cluster programming, pipeline staging.

**For B200, the highest-value hardware features are:**
- **TMA** — hardware fetches tiles into smem asynchronously while compute runs
- **WGMMA** — warpgroup MMA operating on smem tiles directly (what makes H100/B200 GEMM fast)
- **Persistent warp specialization** — producer warps load while consumer warps compute, pipelined

These are only accessible via CUDA/PTX. Use Triton to validate the algorithm and understand fusion structure, then move to CUTLASS/CuTe for peak B200 utilization.

## Optimization Roadmap

| | Phase | Tool | What | Key Win |
|---|-------|------|------|---------|
| ✅ | 0. Baseline | PyTorch | Reference impl, 200+ kernel launches | Correctness ground truth |
| ✅ | 1. Skeleton | Triton + PyTorch | Token sort by expert, Triton scatter-add, still `torch.matmul` | Structured permutation |
| ⬜ | 2. Fused FP8 GEMM | Triton | Wire up `_gemm_fp8_blockscale`; pass FP8 tensors to `tl.dot` directly (not .to(fp32)); eliminate Python expert loop + pre-dequant | Eliminate ~2GB intermediate tensors; use hardware FP8 tensor cores |
| ⬜ | 3. Fuse SwiGLU epilogue | Triton | Compute SwiGLU in registers inside GEMM1 epilogue before store | Eliminate GEMM1 output buffer, halve mem traffic GEMM1→GEMM2 |
| ⬜ | 4. Single grid + fuse scatter-add | Triton | One kernel grid over all experts (keyed by expert_id × tile_m × tile_n); fuse weighted scatter-add into GEMM2 epilogue | 32×2 launches → 2 launches; eliminate GEMM2 output buffer |
| ⬜ | 5. Fuse routing | Triton | Triton kernel for sigmoid→group-topk→global-topk→permutation | Remove PyTorch routing overhead |
| ⬜ | 6. CUTLASS grouped GEMM | CUDA/CUTLASS 3.x | Drop in CUTLASS grouped GEMM with TMA + WGMMA; fuse SwiGLU + scatter-add in custom epilogue | Near-peak Blackwell utilization without full custom CUDA |
| ⬜ | 7. Custom CUDA kernel | CUDA/CuTe | Persistent warp-specialized kernel; producer warps do TMA loads, consumer warps do WGMMA; fused epilogue with scatter-add | Full control, peak throughput |
| ⬜ | 8. Tune | CUDA | Autotune tile sizes, pad expert batches to BLOCK_M, L2 cache reorder, stream overlap | Last 10-30% |

## Expert Token Batching Problem

With T≤256 tokens and 32 experts (top-8 routing), average tokens per expert ≈ 64 for T=256, but can be 0 for T=1. Small M severely underutilizes tensor cores. Mitigations:
- Pad expert batches to minimum `BLOCK_M` (wastes compute, fills pipeline)
- Stream-K decomposition (CUTLASS feature): splits K dimension across SMs for load balancing
- Group small experts together into one tile (complex but avoids padding waste)

---
