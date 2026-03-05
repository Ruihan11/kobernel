# python kernel00

```
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 11.402 ms | 1.02x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload e05c6c03...: PASSED | 10.833 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 6230e838...: PASSED | 13.687 ms | 1.02x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 8f1ff9f1...: PASSED | 16.073 ms | 0.97x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 1a4c6ba1...: PASSED | 20.699 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload a7c2bcfd...: PASSED | 12.318 ms | 1.02x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 2e69caee...: PASSED | 11.248 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 8cba5890...: PASSED | 12.199 ms | 1.00x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 5e8dc11c...: PASSED | 44.716 ms | 1.00x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 58a34f27...: PASSED | 35.538 ms | 1.00x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 5eadab1e...: PASSED | 13.484 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload eedc63b2...: PASSED | 13.341 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload e626d3e6...: PASSED | 14.980 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 74d7ff04...: PASSED | 14.534 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 4822167c...: PASSED | 14.598 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 81955b1e...: PASSED | 14.170 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload 76010cb4...: PASSED | 13.922 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload fc378037...: PASSED | 14.221 ms | 1.01x speedup | abs_err=0.00e+00, rel_err=0.00e+00
  Workload f7d6ac7c...: PASSED | 12.910 ms | 1.02x speedup | abs_err=0.00e+00, rel_err=0.00e+00
```

# triton kernel01
```
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 11.585 ms | 1.04x speedup | abs_err=2.56e+02, rel_err=3.95e-03
  Workload e05c6c03...: PASSED | 11.276 ms | 0.99x speedup | abs_err=3.20e+01, rel_err=6.13e-03
  Workload 6230e838...: PASSED | 13.379 ms | 1.10x speedup | abs_err=1.60e+01, rel_err=1.39e-02
  Workload 8f1ff9f1...: PASSED | 14.959 ms | 1.08x speedup | abs_err=1.02e+03, rel_err=6.99e-03
  Workload 1a4c6ba1...: PASSED | 19.430 ms | 1.08x speedup | abs_err=1.02e+03, rel_err=1.33e-01
  Workload a7c2bcfd...: PASSED | 12.066 ms | 1.04x speedup | abs_err=3.20e+01, rel_err=5.92e-03
  Workload 2e69caee...: PASSED | 11.365 ms | 1.01x speedup | abs_err=5.12e+02, rel_err=6.71e-03
  Workload 8cba5890...: PASSED | 11.994 ms | 1.04x speedup | abs_err=6.40e+01, rel_err=3.26e-02
  Workload 5e8dc11c...: PASSED | 43.102 ms | 1.05x speedup | abs_err=1.02e+03, rel_err=2.00e+00
  Workload 58a34f27...: PASSED | 33.843 ms | 1.06x speedup | abs_err=2.05e+03, rel_err=8.05e-01
  Workload 5eadab1e...: PASSED | 12.920 ms | 1.06x speedup | abs_err=8.00e+00, rel_err=5.43e-03
  Workload eedc63b2...: PASSED | 12.684 ms | 1.07x speedup | abs_err=1.00e+00, rel_err=6.90e-03
  Workload e626d3e6...: PASSED | 14.281 ms | 1.07x speedup | abs_err=1.28e+02, rel_err=7.35e-03
  Workload 74d7ff04...: PASSED | 13.678 ms | 1.09x speedup | abs_err=5.12e+02, rel_err=7.14e-03
  Workload 4822167c...: PASSED | 14.299 ms | 1.06x speedup | abs_err=5.12e+02, rel_err=1.43e-01
  Workload 81955b1e...: PASSED | 13.402 ms | 1.08x speedup | abs_err=2.56e+02, rel_err=1.48e-02
  Workload 76010cb4...: PASSED | 13.173 ms | 1.08x speedup | abs_err=4.00e+00, rel_err=7.75e-03
  Workload fc378037...: PASSED | 13.460 ms | 1.09x speedup | abs_err=6.40e+01, rel_err=6.90e-03
  Workload f7d6ac7c...: PASSED | 12.458 ms | 1.07x speedup | abs_err=4.00e+00, rel_err=7.46e-03
  ```
  