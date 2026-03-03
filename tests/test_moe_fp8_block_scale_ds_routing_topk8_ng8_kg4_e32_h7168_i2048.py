"""
Correctness test for moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.

Usage:
    python tests/test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.py \
        --ref "solution/python/kernel.py::kernel" \
        --test "solution/python/kernel.py::kernel"
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Geometry
H, I, E_GLOBAL, E_LOCAL, BLOCK = 7168, 2048, 256, 32, 128


def load_fn(entry: str):
    """Load 'path/to/file.py::func_name' -> callable."""
    path, func = entry.rsplit("::", 1)
    spec = importlib.util.spec_from_file_location("mod", Path(path).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def quantize_fp8_1d(x: torch.Tensor, block: int = BLOCK):
    """Per-block along last dim. For activations [T, H] -> scale [T, H//B]."""
    shape = x.shape
    n = shape[-1] // block
    blocked = x.reshape(*shape[:-1], n, block)
    amax = blocked.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (amax / fp8_max).squeeze(-1)
    fp8 = (blocked / amax * fp8_max).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).reshape(shape)
    return fp8, scale.float()


def quantize_fp8_2d(x: torch.Tensor, block: int = BLOCK):
    """
    Per-block on last two dims. For weights [E, R, C] -> scale [E, R//B, C//B].
    Each 128x128 tile gets one scale factor.
    """
    *batch, rows, cols = x.shape
    br, bc = rows // block, cols // block
    blocked = x.reshape(*batch, br, block, bc, block)
    amax = blocked.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (amax / fp8_max).squeeze(-1).squeeze(-2)
    fp8 = (blocked / amax * fp8_max).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    fp8 = fp8.reshape(*batch, rows, cols)
    return fp8, scale.float()


def make_inputs(T: int, seed: int = 42, device: str = "cpu"):
    """Generate random inputs matching competition spec."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)

    routing_logits = r(T, E_GLOBAL) * 1.5
    routing_bias = r(E_GLOBAL) * 0.1

    # hidden_states: 1D block scale along H, then transpose scale to [H//B, T]
    h_fp8, h_scale = quantize_fp8_1d(r(T, H) * 0.5)
    h_scale = h_scale.t().contiguous()  # [H//128, T]

    # weights: 2D block scale on [rows, cols] per expert
    # gemm1: [E, 2I, H] -> scale [E, 2I//128, H//128]
    w1_fp8, w1_scale = quantize_fp8_2d(r(E_LOCAL, 2 * I, H) * (2 / H) ** 0.5)
    # gemm2: [E, H, I] -> scale [E, H//128, I//128]
    w2_fp8, w2_scale = quantize_fp8_2d(r(E_LOCAL, H, I) * (2 / I) ** 0.5)

    local_expert_offset = (E_GLOBAL - E_LOCAL) // 2  # 112
    routed_scaling_factor = 2.5

    args = (
        routing_logits, routing_bias,
        h_fp8, h_scale,
        w1_fp8, w1_scale,
        w2_fp8, w2_scale,
        local_expert_offset, routed_scaling_factor,
    )
    if device != "cpu":
        args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
    return args


def compare(ref: torch.Tensor, test: torch.Tensor, atol=1e-2, cosine_thr=0.999):
    r, t = ref.float().cpu(), test.float().cpu()
    assert r.shape == t.shape, f"shape mismatch {r.shape} vs {t.shape}"
    max_abs = (r - t).abs().max().item()
    cos = F.cosine_similarity(r.reshape(1, -1), t.reshape(1, -1)).item()
    frac_ok = torch.isclose(r, t, atol=atol, rtol=atol).float().mean().item()
    passed = cos >= cosine_thr and frac_ok >= 0.99
    info = f"max_abs={max_abs:.4e}  cos={cos:.8f}  close={frac_ok*100:.1f}%"
    return passed, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference entry: path.py::func")
    parser.add_argument("--test", required=True, help="Test entry: path.py::func")
    parser.add_argument("--T", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--cosine", type=float, default=0.999)
    args = parser.parse_args()

    ref_fn = load_fn(args.ref)
    test_fn = load_fn(args.test)
    all_passed = True

    for T in args.T:
        inputs = make_inputs(T, seed=args.seed, device=args.device)
        ref_out = ref_fn(*inputs)
        test_out = test_fn(*inputs)
        passed, info = compare(ref_out, test_out, atol=args.atol, cosine_thr=args.cosine)
        status = "✅" if passed else "❌"
        print(f"T={T:>4}  {status}  {info}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()