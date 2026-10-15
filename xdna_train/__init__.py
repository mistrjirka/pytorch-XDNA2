"""Experimental first-class PyTorch device backend for AMD XDNA2."""

from .device import (
    allocator_stats,
    async_dw_stats,
    autotune_result,
    bo_handle,
    coherency_state,
    empty_cache,
    fallback_stats,
    mapped_cpu_stats,
    is_available,
    is_xrt_backed,
    native_library_path,
    npu_mode,
    register_xdna_device,
    reset_fallback_stats,
    reset_mapped_cpu_stats,
    set_npu_mode,
    synchronize,
    try_register_xdna_device,
    wait_for_builds,
)

__version__ = "0.1.0"

try_register_xdna_device()

__all__ = [
    "__version__",
    "register_xdna_device",
    "try_register_xdna_device",
    "is_available",
    "synchronize",
    "empty_cache",
    "allocator_stats",
    "async_dw_stats",
    "autotune_result",
    "set_npu_mode",
    "npu_mode",
    "wait_for_builds",
    "reset_fallback_stats",
    "fallback_stats",
    "mapped_cpu_stats",
    "reset_mapped_cpu_stats",
    "native_library_path",
    "is_xrt_backed",
    "bo_handle",
    "coherency_state",
]
