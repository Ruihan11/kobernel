"""
FlashInfer-Bench Modal Profiling — PyTorch Profiler & TensorBoard on B200.

Profiles the first workload of your solution on NVIDIA B200 via Modal,
producing a Chrome/Perfetto trace.

Includes a deployable TensorBoard server for browsing saved traces.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    # Profile (default):
    modal run scripts/run_modal_profiling.py

    # Profile with options:
    modal run scripts/run_modal_profiling.py --profile-warmup 2 --profile-active 3 --print-rows 20

    # Deploy TensorBoard to browse all saved profiles:
    modal deploy scripts/run_modal_profiling.py
    # Then open the URL printed in terminal.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
profile_volume = modal.Volume.from_name("flashinfer-profiles", create_if_missing=True)

TRACE_SET_PATH = "/data"
PROFILE_DIR = "/profiles"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "flashinfer-bench", "torch", "triton", "numpy",
        "safetensors",
        "tensorboard", "torch_tb_profiler",
    )
)

tb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("tensorboard==2.18.0", "torch_tb_profiler==0.4.3")
)


# ===========================================================================
# Profiled benchmark — directly invokes the kernel under torch.profiler
#
# Benchmark.run_all() executes the kernel in a subprocess, so the parent
# process's torch.profiler sees nothing. Instead we:
#   1. Extract the kernel source from the Solution object
#   2. Materialize the workload tensors from the TraceSet
#   3. Call the kernel function directly under torch.profiler
# ===========================================================================
@app.function(
    image=image, gpu="B200:1", timeout=3600,
    volumes={
        TRACE_SET_PATH: trace_volume,
        PROFILE_DIR: profile_volume,
    },
)
def run_profiled_benchmark(
    solution,
    warmup: int = 2,
    active: int = 3,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_rows: int = 10,
) -> tuple[str, str]:
    """
    Profile the kernel on a B200 using torch.profiler.

    Directly imports the kernel function and materializes workload tensors
    so that torch.profiler captures all CPU + CUDA activity.

    Returns (trace_json, remote_path) for local Perfetto / remote TensorBoard.
    """
    import importlib
    import json
    import tempfile
    import torch
    from uuid import uuid4
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    workload = workloads[0]  # profile the first workload

    # ------------------------------------------------------------------
    # 1. Extract kernel function from Solution sources
    # ------------------------------------------------------------------
    # solution.spec.entry_point is e.g. "kernel02.py::kernel"
    entry_point = solution.spec.entry_point
    if "::" in entry_point:
        module_file, func_name = entry_point.split("::", 1)
    else:
        module_file = entry_point
        func_name = "kernel"

    # Find the source file in solution.sources
    source_code = None
    for src in solution.sources:
        if src.path.endswith(module_file):
            source_code = src.content
            break

    if source_code is None:
        raise ValueError(
            f"Could not find '{module_file}' in solution sources. "
            f"Available: {[s.path for s in solution.sources]}"
        )

    # Write to a temp file and import it
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / module_file
        src_path.write_text(source_code)
        sys.path.insert(0, tmpdir)
        try:
            mod = importlib.import_module(module_file.replace(".py", ""))
            kernel_fn = getattr(mod, func_name)
        finally:
            sys.path.pop(0)

    print(f"Loaded kernel: {module_file}::{func_name}")

    # ------------------------------------------------------------------
    # 2. Materialize workload tensors on GPU
    #
    # Workload inputs are a dict of:
    #   SafetensorsInput: {path, tensor_key} → load from .safetensors file
    #   RandomInput: {type: 'random'} → generate using defn shape/dtype + axes
    #   ScalarInput: {value} → pass as Python scalar
    #
    # The Trace object wraps: trace.workload.inputs (dict of above)
    # Definition has: defn.inputs (dict of TensorSpec with shape/dtype)
    # Axes resolve symbolic dims: defn.axes (consts) + workload.axes (vars)
    # ------------------------------------------------------------------
    trace_obj = workload  # ts.workloads returns Trace objects
    wl = trace_obj.workload
    wl_inputs = wl.inputs  # dict[str, SafetensorsInput|RandomInput|ScalarInput]
    defn_inputs = definition.inputs  # dict[str, TensorSpec]

    # Build axis values: merge const axes from definition + var axes from workload
    axis_values = {}
    for ax_name, ax in definition.axes.items():
        ax_dump = ax.model_dump() if hasattr(ax, 'model_dump') else ax
        if isinstance(ax_dump, dict) and 'value' in ax_dump:
            axis_values[ax_name] = ax_dump['value']
    for ax_name, ax_val in wl.axes.items():
        axis_values[ax_name] = ax_val  # var axes (e.g. seq_len=7)

    print(f"Axis values: {axis_values}")

    # Dtype mapping
    DTYPE_MAP = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
        'int32': torch.int32,
        'int64': torch.int64,
    }

    dataset_root = Path(TRACE_SET_PATH)
    kernel_kwargs = {}

    for input_name, wl_input in wl_inputs.items():
        inp_dump = wl_input.model_dump()
        inp_type = inp_dump['type']

        if inp_type == 'scalar':
            # ScalarInput → pass value directly
            kernel_kwargs[input_name] = inp_dump['value']
            print(f"  {input_name}: scalar = {inp_dump['value']}")

        elif inp_type == 'safetensors':
            # SafetensorsInput → load from .safetensors file
            from safetensors.torch import load_file
            sf_path = dataset_root / inp_dump['path']
            tensors = load_file(str(sf_path))
            tensor_key = inp_dump.get('tensor_key', input_name)
            t = tensors[tensor_key].cuda()
            kernel_kwargs[input_name] = t
            print(f"  {input_name}: safetensors {t.shape} {t.dtype}")

        elif inp_type == 'random':
            # RandomInput → generate from definition's TensorSpec + resolved axes
            spec = defn_inputs[input_name]
            spec_dump = spec.model_dump() if hasattr(spec, 'model_dump') else spec

            # Resolve shape: replace axis names with concrete values
            shape_spec = spec_dump.get('shape')
            if shape_spec is None:
                # Scalar spec with no shape — shouldn't be random, skip
                raise ValueError(f"RandomInput '{input_name}' has no shape in definition")

            shape = []
            for dim in shape_spec:
                if isinstance(dim, int):
                    shape.append(dim)
                elif isinstance(dim, str):
                    if dim not in axis_values:
                        raise ValueError(f"Unknown axis '{dim}' for input '{input_name}'")
                    shape.append(axis_values[dim])
                else:
                    shape.append(int(dim))

            # Resolve dtype
            dtype_str = spec_dump.get('dtype', 'float32')
            # Handle enum or string
            if hasattr(dtype_str, 'value'):
                dtype_str = dtype_str.value
            torch_dtype = DTYPE_MAP.get(dtype_str, torch.float32)

            # Generate random tensor
            if torch_dtype == torch.float8_e4m3fn:
                # Can't directly randn into fp8; generate in fp32 then cast
                t = torch.randn(shape, device='cuda', dtype=torch.float32).to(torch_dtype)
            else:
                t = torch.randn(shape, device='cuda', dtype=torch_dtype)

            kernel_kwargs[input_name] = t
            print(f"  {input_name}: random {list(shape)} {torch_dtype}")

        else:
            raise ValueError(f"Unknown input type '{inp_type}' for '{input_name}'")

    print(f"Materialized {len(kernel_kwargs)} inputs for workload {wl.uuid[:8]}...")

    # Build the callable
    run_kernel = lambda: kernel_fn(**kernel_kwargs)

    # ------------------------------------------------------------------
    # 3. Warmup outside profiler (torch.compile / CUDA lazy init)
    # ------------------------------------------------------------------
    print("Warming up (outside profiler)...")
    for _ in range(3):
        run_kernel()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 4. Profile
    # ------------------------------------------------------------------
    run_id = str(uuid4())
    output_dir = Path(PROFILE_DIR) / f"{solution.name}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = warmup + active
    schedule = torch.profiler.schedule(
        wait=0, warmup=warmup, active=active, repeat=0,
    )

    print(f"Profiling: {warmup} warmup + {active} active steps...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
    ) as prof:
        for _ in range(total_steps):
            run_kernel()
            torch.cuda.synchronize()
            prof.step()

    # ------------------------------------------------------------------
    # 5. Print summary tables
    # ------------------------------------------------------------------
    if print_rows:
        print("\n" + "=" * 80)
        print("CUDA kernel time summary (sorted by cuda_time_total)")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=print_rows,
            )
        )
        print()
        print("=" * 80)
        print("CPU time summary (sorted by cpu_time_total)")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=print_rows,
            )
        )

    # ------------------------------------------------------------------
    # 6. Persist to Volume and return trace JSON
    # ------------------------------------------------------------------
    profile_volume.commit()

    trace_files = sorted(
        output_dir.glob("**/*.pt.trace.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not trace_files:
        raise RuntimeError(f"No trace files found in {output_dir}")

    trace_path = trace_files[0]
    remote_rel = trace_path.relative_to(PROFILE_DIR)
    print(f"\ntrace saved to volume 'flashinfer-profiles' at: {remote_rel}")

    return trace_path.read_text(), str(remote_rel)


# ===========================================================================
# TensorBoard server — deploy with `modal deploy scripts/run_modal.py`
# ===========================================================================

class VolumeReloadMiddleware:
    """WSGI middleware that reloads the profile Volume on full page loads."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if (route := environ.get("PATH_INFO")) in ["/", "/modal-volume-reload"]:
            try:
                profile_volume.reload()
            except Exception as e:
                print(f"Exception reloading profile volume: {e}")
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"
        return self.app(environ, start_response)


@app.function(
    volumes={PROFILE_DIR: profile_volume},
    image=tb_image,
    max_containers=1,
    scaledown_window=5 * 60,  # 5 min idle before scale-down
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard():
    """
    Serve TensorBoard with the torch_tb_profiler plugin.

    Deploy:
        modal deploy scripts/run_modal_profiling.py

    Then open the printed URL. All profiles saved to the
    'flashinfer-profiles' volume will appear automatically.
    Reload the page to pick up new traces after a profiling run.
    """
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=PROFILE_DIR)

    try:
        # TensorBoard 2.18+ API
        (data_provider, deprecated_multiplexer) = board._make_data_provider()
        wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
            board.flags,
            board.plugin_loaders,
            data_provider,
            board.assets_zip_provider,
            deprecated_multiplexer,
            experimental_middlewares=[VolumeReloadMiddleware],
        )
        return wsgi_app._create_wsgi_app()
    except Exception as e:
        print(f"TensorBoard WSGI setup failed ({e}), trying fallback...")
        # Fallback: use the simpler launch-and-wrap approach
        board.configure(logdir=PROFILE_DIR, host="0.0.0.0", port=6006)
        board.launch()
        # Return a simple redirect WSGI app
        from werkzeug.serving import WSGIRequestHandler
        return board._make_app()


# ===========================================================================
# CLI entrypoint
# ===========================================================================
@app.local_entrypoint()
def main(
    profile_warmup: int = 2,
    profile_active: int = 3,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    print_rows: int = 10,
):
    from scripts.pack_solution import pack_solution
    from flashinfer_bench import Solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    # ---- Profile ----
    print("\n" + "=" * 80)
    print("PROFILING: running first workload under torch.profiler on B200...")
    print("=" * 80)

    trace_json, remote_path = run_profiled_benchmark.remote(
        solution,
        warmup=profile_warmup,
        active=profile_active,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        print_rows=print_rows,
    )

    # Save trace locally for Perfetto
    local_path = Path("/tmp") / Path(remote_path).name
    local_path.write_text(trace_json)
    print(f"\nTrace saved locally: {local_path}")
    print(f"View in Perfetto:    https://ui.perfetto.dev  (drag & drop the file)")
    print(f"Remote volume path:  flashinfer-profiles/{remote_path}")
    print(f"\nOr deploy TensorBoard to browse all traces:")
    print(f"  modal deploy scripts/run_modal_profiling.py")