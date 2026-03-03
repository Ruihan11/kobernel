"""
Profile MoE kernel on Modal B200 with PyTorch profiler + TensorBoard.

Usage:
    # Profile the reference python kernel
    modal run scripts/run_modal_profiling.py --function run-python-kernel --print-rows 20

    # Profile the triton kernel
    modal run scripts/run_modal_profiling.py --function run-triton-kernel --print-rows 20

    # Deploy TensorBoard for persistent viewing
    modal deploy scripts/run_modal_profiling.py
"""

from pathlib import Path
from typing import Optional

import modal

traces = modal.Volume.from_name("moe-profiling-traces", create_if_missing=True)
TRACE_DIR = Path("/traces")

app = modal.App("moe-kernel-profiling")

# Image: install deps + add local project source
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .add_local_dir("solution", remote_path="/root/kobernel/solution")
    .add_local_dir("tests", remote_path="/root/kobernel/tests")
)

with image.imports():
    import torch

config = {"gpu": "B200:1", "image": image, "timeout": 3600}


# ============================================================================
# Target functions to profile
# ============================================================================

@app.function(**config)
def run_python_kernel(T: int = 16, seed: int = 42):
    import sys
    sys.path.insert(0, "/root/kobernel")
    sys.path.insert(0, "/root/kobernel/tests")

    from test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 import (
        make_inputs, load_fn,
    )

    kernel_fn = load_fn("/root/kobernel/solution/python/kernel.py::kernel")
    inputs = make_inputs(T=T, seed=seed, device="cuda")
    output = kernel_fn(*inputs)
    torch.cuda.synchronize()
    return output


@app.function(**config)
def run_triton_kernel(T: int = 16, seed: int = 42):
    import sys
    sys.path.insert(0, "/root/kobernel")
    sys.path.insert(0, "/root/kobernel/tests")

    from test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 import (
        make_inputs, load_fn,
    )

    kernel_fn = load_fn("/root/kobernel/solution/triton/kernel.py::kernel")
    inputs = make_inputs(T=T, seed=seed, device="cuda")
    output = kernel_fn(*inputs)
    torch.cuda.synchronize()
    return output


# ============================================================================
# Profiler wrapper
# ============================================================================

@app.function(volumes={TRACE_DIR: traces}, **config)
def profile(
    function,
    label: Optional[str] = None,
    steps: int = 5,
    schedule=None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_rows: int = 0,
    **kwargs,
):
    from uuid import uuid4

    if isinstance(function, str):
        # Modal CLI converts underscores to hyphens, normalize back
        function_key = function.replace("-", "_")
        try:
            function = app.registered_functions[function_key]
        except KeyError:
            raise ValueError(
                f"Function '{function_key}' not found. "
                f"Available: {list(app.registered_functions.keys())}"
            )
    function_name = function.tag

    output_dir = (
        TRACE_DIR / (function_name + (f"_{label}" if label else "")) / str(uuid4())
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if schedule is None:
        if steps < 3:
            raise ValueError("Steps must be at least 3 when using default schedule")
        schedule = {"wait": 1, "warmup": 1, "active": steps - 2, "repeat": 0}

    schedule = torch.profiler.schedule(**schedule)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(steps):
            function.local(**kwargs)
            prof.step()

    if print_rows:
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=print_rows
            )
        )

    trace_path = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda pth: pth.stat().st_mtime,
        reverse=True,
    )[0]

    print(f"trace saved to {trace_path.relative_to(TRACE_DIR)}")

    return trace_path.read_text(), trace_path.relative_to(TRACE_DIR)


# ============================================================================
# TensorBoard
# ============================================================================

tb_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "tensorboard==2.18.0", "torch_tb_profiler==0.4.3"
)


class VolumeMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if (route := environ.get("PATH_INFO")) in ["/", "/modal-volume-reload"]:
            try:
                traces.reload()
            except Exception as e:
                print("Exception while re-loading traces: ", e)
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"
        return self.app(environ, start_response)


@app.function(
    volumes={TRACE_DIR: traces},
    image=tb_image,
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(TRACE_DIR))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )

    return wsgi_app._create_wsgi_app()


# ============================================================================
# CLI
# ============================================================================

@app.local_entrypoint()
def main(
    function: str = "run_python_kernel",
    label: Optional[str] = None,
    steps: int = 5,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_rows: int = 20,
    tokens: int = 16,
    seed: int = 42,
):
    results, remote_path = profile.remote(
        function,
        label=label,
        steps=steps,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
        T=tokens,
        seed=seed,
    )

    output_path = Path("/tmp") / remote_path.name
    output_path.write_text(results)
    print(f"trace saved locally at {output_path}")