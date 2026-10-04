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

    # Preserve Parameter object identity when Module.to("xdna") is called.
    # This matters for ordinary notebooks that construct an optimizer before
    # moving the model: Adam/SGD retain references to the same Parameter
    # objects and those objects are swapped to XDNA storage in place.
    future = getattr(torch, "__future__", None)
    set_swap = getattr(future, "set_swap_module_params_on_conversion", None)
    if set_swap is not None:
        set_swap(True)

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

    gemm = _ROOT / "_artifacts" / "train_gemm_64x32x96"
    if gemm.is_dir():
        _C.configure_gemm_family(
            str(gemm / "base.xclbin"),
            str(gemm / "fwd.bin"),
            str(gemm / "dx.bin"),
            str(gemm / "dw.bin"),
        )

    conv = _ROOT / "_artifacts" / "conv3x3_patch40_128"
    if conv.is_dir():
        _C.configure_conv_family(
            str(conv / "final.xclbin"),
            str(conv / "fwd18.bin"),
            str(conv / "dx18.bin"),
            str(conv / "fwd36.bin"),
            str(conv / "dx36.bin"),
        )

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

    full_dw = Path(
        os.environ.get(
            "XDNA_FULLFRAME_DW_ARTIFACT_DIR",
            str(_ROOT / "_artifacts" / "fullframe_dw_unbundled"),
        )
    )
    if (
        os.environ.get("XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_DW") == "1"
        and (full_dw / "final.xclbin").is_file()
        and (full_dw / "insts.txt").is_file()
    ):
        _C.configure_fullframe_dw584(
            str(full_dw / "final.xclbin"),
            str(full_dw / "insts.txt"),
        )

    if not hasattr(torch, "xdna"):
        torch._register_device_module("xdna", _XdnaDeviceModule)

    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=False,
    )
    _INITIALIZED = True


def try_register_xdna_device() -> bool:
    try:
        register_xdna_device()
    except (ImportError, RuntimeError, OSError):
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
