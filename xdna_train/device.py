from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


_ROOT = Path(__file__).resolve().parent
_INITIALIZED = False
_C = None


def _device_index(device=None, optional: bool = False) -> int:
    del optional
    if device is None:
        return 0
    if isinstance(device, int):
        index = device
    else:
        d = torch.device(device)
        if d.type != "xdna":
            raise ValueError(f"expected xdna device, got {d}")
        index = 0 if d.index is None else d.index
    if index != 0:
        raise ValueError("only xdna:0 is currently supported")
    return index


class _XdnaDeviceModule:
    _utils = SimpleNamespace(_get_device_index=_device_index)

    @staticmethod
    def is_available() -> bool:
        return bool(_C.is_available())

    @staticmethod
    def device_count() -> int:
        return int(_C.device_count())

    @staticmethod
    def current_device() -> int:
        return int(_C.current_device())

    @staticmethod
    def set_device(device) -> None:
        _C.set_device(_device_index(device))

    @staticmethod
    def synchronize(device=None) -> None:
        if device is not None:
            _device_index(device)
        _C.synchronize()

    @staticmethod
    def empty_cache() -> None:
        _C.empty_cache()

    @staticmethod
    def memory_stats() -> dict[str, int]:
        return dict(_C.allocator_stats())

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.bfloat16, torch.float32]

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        torch.random.default_generator.manual_seed(seed)


def _configure_default_allocator_budget() -> None:
    if "XDNA_BO_CACHE_MB" in os.environ:
        return
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total_bytes = page_size * phys_pages
    except (AttributeError, OSError, ValueError):
        total_bytes = 0

    # Full-frame training has a multi-GiB activation working set. A tiny cache
    # causes repeated XRT BO allocation/free on every batch. Keep at most 25%
    # of system RAM cached, capped at 8 GiB and floored at 512 MiB.
    if total_bytes > 0:
        budget_mb = total_bytes // 4 // (1024 * 1024)
        budget_mb = max(512, min(8192, int(budget_mb)))
    else:
        budget_mb = 512
    os.environ["XDNA_BO_CACHE_MB"] = str(budget_mb)


def _manifest(artifact_dir: Path) -> dict[str, str]:
    path = artifact_dir / "manifest.txt"
    if not path.is_file():
        return {}
    return dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )


def _iron_python() -> str:
    """Python of an mlir-aie (IRON) environment, used to compile conv3x3w
    instruction streams on demand.  XDNA_IRON_PYTHON overrides; an empty
    value disables on-demand builds."""
    raw = os.environ.get("XDNA_IRON_PYTHON")
    if raw is not None:
        return raw
    for candidate in (
        _ROOT.parent.parent / "mlir-aie-env" / "bin" / "python",
        Path.home() / "mlir-aie-env" / "bin" / "python",
        Path.home() / "ironenv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


def _configure_iron_family(name: str, configure, min_macs: int) -> None:
    """Configure an on-demand IRON family: _designs/<name>.py + <name>_kernel.cc,
    streams cached under the source hash."""
    import shlex

    designs = _ROOT / "_designs"
    sources = [designs / f"{name}.py", designs / f"{name}_kernel.cc"]
    if not all(p.is_file() for p in sources):
        return
    h = hashlib.sha256()
    for p in sources:
        h.update(p.read_bytes())
    from ._build import _cache_root
    env = name.upper()
    cache = Path(os.environ.get(f"XDNA_{env}_DIR") or _cache_root() / name / h.hexdigest()[:16])
    python = _iron_python()
    build_cmd = ""
    if python:
        build_cmd = f"nice -n 10 {shlex.quote(python)} {shlex.quote(str(sources[0]))}"
    if not build_cmd and not (cache / f"{name}.xclbin").is_file():
        return
    configure(
        str(cache),
        build_cmd,
        int(os.environ.get(f"XDNA_{env}_MIN_MACS", str(min_macs))),
        os.environ.get("XDNA_CONV3X3W_SYNC_BUILD", "0") == "1",
    )


def _load_native():
    global _C
    if _C is not None:
        return _C
    from ._build import build_native
    _C = build_native()
    return _C

def register_xdna_device() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    _configure_default_allocator_budget()

    _load_native()

    current = torch._C._get_privateuse1_backend_name()
    if current == "privateuseone":
        torch.utils.rename_privateuse1_backend("xdna")
    elif current != "xdna":
        raise RuntimeError(
            f"PrivateUse1 is already claimed by backend {current!r}; "
            "XDNA cannot coexist with another PrivateUse1 backend in one process."
        )

    # Autograd runs a non-CPU device's backward on its own worker thread, which
    # gets its own OpenMP pool; the forward pool's spinning threads then compete
    # with it (MLP train step 8.0 -> 3.8 ms, autograd hand-off ~90 us per
    # backward).  Run backward on the calling thread, as for CPU tensors.  The
    # setting is thread-local: it applies to the thread that registers the
    # device.  XDNA_AUTOGRAD_THREAD=1 keeps PyTorch's device thread.
    if os.environ.get("XDNA_AUTOGRAD_THREAD", "0") != "1":
        torch.autograd.set_multithreading_enabled(False)

    gemm = _ROOT / "_artifacts" / "train_gemm_64x32x96"
    if gemm.is_dir():
        _C.configure_gemm_family(
            str(gemm / "base.xclbin"),
            str(gemm / "fwd.bin"),
            str(gemm / "dx.bin"),
            str(gemm / "dw.bin"),
        )

    conv = Path(
        os.environ.get("XDNA_PATCH_CONV_ARTIFACT_DIR")
        or str(_ROOT / "_artifacts" / "conv3x3_patch40_128")
    )
    # Diagnostic cost-policy gate: leave resident patch Conv disabled when its
    # complete fwd+dX+CPU dW costs more than the mapped CPU fallback.
    if conv.is_dir() and os.environ.get("XDNA_ENABLE_PATCH_CONV", "1") == "1":
        shapes = _manifest(conv)
        _C.configure_conv_family(
            str(conv / "final.xclbin"),
            str(conv / "fwd18.bin"),
            str(conv / "dx18.bin"),
            str(conv / "fwd36.bin"),
            str(conv / "dx36.bin"),
            int(shapes.get("fwd18_shape", "12960x").split("x")[0]),
            int(shapes.get("fwd36_shape", "51840x").split("x")[0]),
        )

    # General 3x3 conv family (BFP16 weights); XDNA_CONV3X3W=0 disables it.
    # Below ~20 MMAC host packing and dispatch cost more than the CPU conv; on
    # top of that floor, plan() runs a padded-work cost model (RGB input convs
    # stay on the CPU).  XDNA_CONV3X3W_FORCE=1 bypasses the model; see
    # XDNA_CONV3X3W_{NPU_OVERHEAD_MS,NPU_GMACS,HOST_GBS,CPU_GMACS}.
    if os.environ.get("XDNA_CONV3X3W", "1") == "1":
        _configure_iron_family("conv3x3w", _C.configure_conv3x3w, 20_000_000)
        # Its weight gradient (BFP16 operands, fp32 sums), deferred to the
        # optimizer.  Smaller dWs stay on the CPU, where they overlap the NPU dX
        # chain for free; XDNA_CONV3X3W_DW=0 keeps every dW on the CPU.
        if os.environ.get("XDNA_CONV3X3W_DW", "1") == "1":
            _configure_iron_family("conv3x3w_dw", _C.configure_conv3x3w_dw, 1_000_000_000)

    full = Path(
        os.environ.get(
            "XDNA_FULLFRAME_ARTIFACT_DIR",
            str(_ROOT / "_artifacts" / "fullframe_strix_b4_m96"),
        )
    )
    if (
        os.environ.get("XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_CONV", "1") == "1"
        and (full / "final.xclbin").is_file()
        and (full / "res146.bin").is_file()
    ):
        manifest = full / "manifest.txt"
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                if line.startswith("tile="):
                    artifact_tm = line.split("=", 1)[1].split("x", 1)[0]
                    configured_tm = os.environ.get("XDNA_FULLFRAME_TM")
                    if configured_tm is not None and configured_tm != artifact_tm:
                        raise RuntimeError(
                            "XDNA_FULLFRAME_TM does not match artifact tile: "
                            f"{configured_tm} != {artifact_tm}"
                        )
                    os.environ["XDNA_FULLFRAME_TM"] = artifact_tm
                    break

        def _optional_stream(name: str) -> str:
            path = full / name
            return str(path) if path.is_file() else ""

        _C.configure_fullframe_conv_family(
            str(full / "final.xclbin"),
            str(full / "res146.bin"),
            _optional_stream("fwd292.bin"),
            _optional_stream("fwd584.bin"),
            _optional_stream("dx292.bin"),
            _optional_stream("dx584s73.bin"),
        )

    fast_full_raw = os.environ.get(
        "XDNA_FAST_FULLFRAME_ARTIFACT_DIR",
        os.environ.get("XDNA_FAST_DX584_ARTIFACT_DIR"),
    )
    fast_full = Path(
        fast_full_raw
        if fast_full_raw
        else str(_ROOT / "_artifacts" / "fullframe_q4_resident_k1152_m96")
    )
    enable_fast = os.environ.get(
        "XDNA_ENABLE_FAST_FULLFRAME",
        os.environ.get("XDNA_ENABLE_FAST_DX584", "1"),
    )
    if (
        enable_fast == "1"
        and (fast_full / "final.xclbin").is_file()
        and (fast_full / "dx584s73.bin").is_file()
    ):
        fast_manifest = fast_full / "manifest.txt"
        if fast_manifest.is_file():
            for line in fast_manifest.read_text().splitlines():
                if line.startswith("tile="):
                    artifact_tm = line.split("=", 1)[1].split("x", 1)[0]
                    configured_tm = os.environ.get("XDNA_FULLFRAME_TM")
                    if configured_tm is not None and configured_tm != artifact_tm:
                        raise RuntimeError(
                            "XDNA_FULLFRAME_TM does not match fast full-frame artifact tile: "
                            f"{configured_tm} != {artifact_tm}"
                        )
                    os.environ["XDNA_FULLFRAME_TM"] = artifact_tm
                    break
        fast_res146 = fast_full / "res146.bin"
        fast_dx292 = fast_full / "dx292.bin"
        if fast_res146.is_file():
            _C.configure_fast_fullframe_family(
                str(fast_full / "final.xclbin"),
                str(fast_res146),
                str(fast_dx292) if fast_dx292.is_file() else "",
                str(fast_full / "dx584s73.bin"),
            )
        else:
            _C.configure_fast_dx584(
                str(fast_full / "final.xclbin"),
                str(fast_full / "dx584s73.bin"),
            )

    fast_conv11 = Path(
        os.environ.get(
            "XDNA_FAST_CONV11_ARTIFACT_DIR",
            str(_ROOT / "_artifacts" / "conv11_dx_q4_resident_k576_m96"),
        )
    )
    if (
        os.environ.get("XDNA_ENABLE_FAST_CONV11", "1") == "1"
        and (fast_conv11 / "final.xclbin").is_file()
        and (fast_conv11 / "dx584c64.bin").is_file()
    ):
        manifest = fast_conv11 / "manifest.txt"
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                if line.startswith("tile="):
                    artifact_tm = line.split("=", 1)[1].split("x", 1)[0]
                    configured_tm = os.environ.get("XDNA_FULLFRAME_TM")
                    if configured_tm is not None and configured_tm != artifact_tm:
                        raise RuntimeError(
                            "XDNA_FULLFRAME_TM does not match fast conv11 artifact tile: "
                            f"{configured_tm} != {artifact_tm}"
                        )
                    os.environ["XDNA_FULLFRAME_TM"] = artifact_tm
                    break
        _C.configure_fast_conv11_dx(
            str(fast_conv11 / "final.xclbin"),
            str(fast_conv11 / "dx584c64.bin"),
            128,
        )

    if not hasattr(torch, "xdna"):
        torch._register_device_module("xdna", _XdnaDeviceModule)

    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=False,
    )
    _install_scoped_param_swap()
    if os.environ.get("XDNA_CPU_ISLANDS", "1") == "1":
        from . import cpu_islands

        cpu_islands.install()
    _start_npu_mode_selection()
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    register_optimizer_step_post_hook(_bump_fused_param_versions)
    if os.environ.get("XDNA_FUSED_OPTIMIZER", "1") == "1":
        register_optimizer_step_pre_hook(_prefer_fused_optimizer)
    _INITIALIZED = True


class _SwapModuleParams:
    """Use torch.utils.swap_tensors while a module moves to XDNA.

    PyTorch keeps Parameter identity in Module.to() only for backends it
    considers shallow-copy compatible (CPU, CUDA, MPS, HIP, XPU, ...), not for
    PrivateUse1: model.to("xdna") would create new Parameter objects, and an
    optimizer built before the move would keep updating the old CPU tensors.
    Swapping keeps the same objects.  It is enabled only for conversions to
    XDNA, so other modules keep PyTorch's default behaviour (swapping fails on
    parameters that have weak references, e.g. after LSTM.load_state_dict).
    """

    def __enter__(self):
        self.prev = torch.__future__.get_swap_module_params_on_conversion()
        torch.__future__.set_swap_module_params_on_conversion(True)

    def __exit__(self, *exc):
        torch.__future__.set_swap_module_params_on_conversion(self.prev)


def _install_scoped_param_swap() -> None:
    import functools

    module = torch.nn.Module
    if getattr(module.to, "_xdna_scoped_swap", False):
        return

    def is_xdna(device) -> bool:
        return device is not None and torch.device(device).type == "xdna"

    def scoped(fn, targets_xdna):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not targets_xdna(*args, **kwargs):
                return fn(self, *args, **kwargs)
            with _SwapModuleParams():
                return fn(self, *args, **kwargs)

        wrapper._xdna_scoped_swap = True
        return wrapper

    def to_targets(*args, **kwargs):
        try:
            device = torch._C._nn._parse_to(*args, **kwargs)[0]
        except Exception:
            return False  # let Module.to raise its own error
        return is_xdna(device)

    module.to = scoped(module.to, to_targets)
    module.to_empty = scoped(module.to_empty, lambda *, device=None, **k: is_xdna(device))
    module.xdna = scoped(module.xdna, lambda *a, **k: True)


_FUSED_CHECKED: set[int] = set()


def _prefer_fused_optimizer(optimizer, args, kwargs) -> None:
    """Default XDNA optimizers to PyTorch's fused CPU kernels.

    PyTorch picks the `foreach` implementation by default for a PrivateUse1
    device, i.e. ~10 list ops per step that each run as a CPU fallback; the
    fused kernel (one pass per parameter, numerically the same update) is
    supported on XDNA storage through the CPU fallback (model zoo MLP
    4.4 -> 3.6 ms per step).  Only param groups that left both `fused` and
    `foreach` unset, contain only XDNA parameters and have no optimizer state
    yet are changed, so it happens before the first step or not at all.
    XDNA_FUSED_OPTIMIZER=0 disables it.
    """
    if id(optimizer) in _FUSED_CHECKED:
        return
    _FUSED_CHECKED.add(id(optimizer))
    for group in optimizer.param_groups:
        if (
            "fused" not in group
            or group.get("fused") is not None
            or group.get("foreach") is not None
            or group.get("differentiable")
        ):
            continue
        params = group["params"]
        if params and all(
            p.device.type == "xdna" and p.is_floating_point() and not optimizer.state.get(p)
            for p in params
        ):
            group["fused"] = True


def _bump_fused_param_versions(optimizer, args, kwargs) -> None:
    """Mark XDNA parameters updated by a fused optimizer step as modified.

    The fused kernels (`_fused_adam_` and friends) write the parameters through
    the CPU fallback without bumping their autograd version counters.  The NPU
    convolution weight cache keys on that counter, so without this every NPU
    conv kept using the weights it first saw while the optimizer moved on.
    """
    for group in optimizer.param_groups:
        if not group.get("fused"):
            continue
        params = [p for p in group["params"] if p.device.type == "xdna"]
        if params:
            torch.autograd.graph.increment_version(params)


# --------------------------------------------------------------------------
# Step autotuner: which NPU paths pay off for this model.
# --------------------------------------------------------------------------

NPU_MODES = {2: "all NPU paths", 1: "NPU without weight gradients", 0: "CPU only"}


class _StepAutotuner:
    """Times real training steps under each NPU mode and keeps the fastest.

    The per-op cost models decide shape by shape, but they cannot see costs
    that are paid once per step: NPU weight gradients run from a second
    xclbin, so a step that uses both pays two ~2.6 ms context switches, and a
    model with only a few NPU-friendly convolutions (UNet in the model zoo:
    67 ms with NPU convs, 50 ms without, 56 ms on CPU) loses overall.  Like
    cudnn.benchmark, this measures instead: steps are the intervals between
    calls of the first optimizer's step().  After `warmup` steps (and once
    pending kernel builds are done) the modes run in blocks of one transition
    step plus `steps` timed steps, in ABBA order (2 1 0 0 1 2, `rounds` times)
    so that drift (heat, background load) does not favour one mode; an NPU mode
    must beat CPU-only by `tolerance`.
    """

    def __init__(self, warmup: int, steps: int, rounds: int, tolerance: float) -> None:
        self.warmup, self.steps, self.tolerance = warmup, steps, tolerance
        self.optimizer_id = None
        self.last = None
        self.seen = 0
        self.order = [2, 1, 0, 0, 1, 2] * rounds
        self.phase = 0          # index into order while measuring
        self.skip = 1           # transition steps left in this block
        self.timed = 0          # timed steps in this block
        self.samples: dict[int, list[float]] = {m: [] for m in (2, 1, 0)}
        self.result: dict | None = None
        self.active = True

    def __call__(self, optimizer, args, kwargs) -> None:
        import time

        # The hook stays registered once done: removing it from inside a step
        # mutates the post-hook dict PyTorch is iterating over.
        if not self.active:
            return
        if self.optimizer_id is None:
            # Time the first optimizer that trains XDNA parameters; a CPU
            # model's optimizer in the same process says nothing about NPU modes.
            if not any(p.device.type == "xdna" for g in optimizer.param_groups for p in g["params"]):
                return
            self.optimizer_id = id(optimizer)
        if id(optimizer) != self.optimizer_id:
            return
        now = time.perf_counter()
        last, self.last = self.last, now
        self.seen += 1
        if last is None:
            return
        if self.seen <= self.warmup or (
            _C.iron_builds_pending() > 0 and self.seen <= 20 * self.warmup
        ):
            return
        if self.skip > 0:
            self.skip -= 1
            return
        self.samples[self.order[self.phase]].append((now - last) * 1e3)
        self.timed += 1
        if self.timed < self.steps:
            return
        self.phase += 1
        self.timed = 0
        if self.phase < len(self.order):
            _C.set_npu_mode(self.order[self.phase])
            self.skip = 1
            return
        self._decide()

    def _decide(self) -> None:
        import statistics

        med = {m: statistics.median(v) for m, v in self.samples.items()}
        best = min((1, 2), key=lambda m: med[m])
        if med[best] > med[0] * (1.0 - self.tolerance):
            best = 0
        _C.set_npu_mode(best)
        self.result = {"mode": best, "step_ms": med, "samples": self.samples}
        self.active = False
        if os.environ.get("XDNA_AUTOTUNE_VERBOSE", "0") == "1":
            times = ", ".join(f"{NPU_MODES[m]} {med[m]:.1f}" for m in (2, 1, 0))
            print(f"xdna: using {NPU_MODES[best]} (median step ms: {times})",
                  file=sys.stderr, flush=True)


_AUTOTUNER: _StepAutotuner | None = None


def _start_npu_mode_selection() -> None:
    """XDNA_NPU_MODE=0/1/2 fixes the mode; XDNA_AUTOTUNE=0 keeps mode 2
    (all NPU paths, per-op cost models only) without measuring."""
    global _AUTOTUNER
    forced = os.environ.get("XDNA_NPU_MODE")
    if forced is not None:
        _C.set_npu_mode(int(forced))
        return
    if os.environ.get("XDNA_AUTOTUNE", "1") != "1":
        return
    _AUTOTUNER = _StepAutotuner(
        warmup=int(os.environ.get("XDNA_AUTOTUNE_WARMUP", "3")),
        steps=int(os.environ.get("XDNA_AUTOTUNE_STEPS", "2")),
        rounds=int(os.environ.get("XDNA_AUTOTUNE_ROUNDS", "1")),
        tolerance=float(os.environ.get("XDNA_AUTOTUNE_TOLERANCE", "0.03")),
    )
    from torch.optim.optimizer import register_optimizer_step_post_hook

    register_optimizer_step_post_hook(_AUTOTUNER)


def autotune_result() -> dict | None:
    """The step autotuner's choice once it has finished, else None:
    {"mode": 0|1|2, "step_ms": {mode: median step ms}}."""
    return None if _AUTOTUNER is None else _AUTOTUNER.result


def set_npu_mode(mode: int) -> None:
    """2: all NPU paths, 1: no NPU weight gradients, 0: CPU only.  Stops the
    step autotuner."""
    register_xdna_device()
    if _AUTOTUNER is not None:
        _AUTOTUNER.active = False
    _C.set_npu_mode(int(mode))


def npu_mode() -> int:
    register_xdna_device()
    return _C.npu_mode()


def try_register_xdna_device() -> bool:
    """Register the device if possible (called on import).  On failure, warn
    with the cause: otherwise the first sign is a confusing
    "No module named 'torch.xdna'" much later."""
    try:
        register_xdna_device()
    except (ImportError, RuntimeError, OSError) as e:
        import warnings

        warnings.warn(f"xdna_train: XDNA device not registered: {type(e).__name__}: {e}",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


def is_available() -> bool:
    try:
        register_xdna_device()
    except (ImportError, RuntimeError, OSError):
        return False
    return _XdnaDeviceModule.is_available()


def synchronize(device=None) -> None:
    register_xdna_device()
    _XdnaDeviceModule.synchronize(device)


def empty_cache() -> None:
    register_xdna_device()
    _C.empty_cache()


def allocator_stats() -> dict[str, int]:
    register_xdna_device()
    return dict(_C.allocator_stats())


def reset_mapped_cpu_stats() -> None:
    """Reset profiling counters for XDNA handlers that execute on mapped CPU storage."""
    _C.reset_mapped_cpu_stats()


def mapped_cpu_stats() -> dict[str, dict[str, float | int]]:
    """Return profiling counters for mapped-CPU XDNA handlers."""
    return {
        str(name): {
            "calls": int(value["calls"]),
            "total_ms": float(value["total_ms"]),
            "avg_ms": float(value["avg_ms"]),
        }
        for name, value in dict(_C.mapped_cpu_stats()).items()
    }


def reset_fallback_stats() -> None:
    register_xdna_device()
    _C.reset_fallback_stats()


def fallback_stats() -> dict[str, dict[str, float | int]]:
    register_xdna_device()
    return {
        str(name): dict(value)
        for name, value in dict(_C.fallback_stats()).items()
    }


def native_library_path() -> Path:
    register_xdna_device()
    return Path(_C.__file__).resolve()


def cpu_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    register_xdna_device()
    return _C.cpu_mm(lhs, rhs)


def bo_handle(tensor: torch.Tensor) -> int:
    register_xdna_device()
    return int(_C.bo_handle(tensor))


def is_xrt_backed(tensor: torch.Tensor) -> bool:
    register_xdna_device()
    return bool(_C.is_xrt_backed(tensor))


def prepare_device_read(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.prepare_device_read(tensor)


def prepare_host_read(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.prepare_host_read(tensor)


def mark_device_dirty(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.mark_device_dirty(tensor)


def mark_host_dirty(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.mark_host_dirty(tensor)


def coherency_state(tensor: torch.Tensor) -> int:
    register_xdna_device()
    return int(_C.coherency_state(tensor))


def sync_to_device(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.sync_to_device(tensor)


def sync_from_device(tensor: torch.Tensor) -> None:
    register_xdna_device()
    _C.sync_from_device(tensor)
