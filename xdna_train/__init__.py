"""Experimental first-class PyTorch device backend for AMD XDNA2."""

from .device import (
    allocator_stats,
    bo_handle,
    coherency_state,
    empty_cache,
    fallback_stats,
    mapped_cpu_stats,
    is_available,
    is_xrt_backed,
    native_library_path,
    register_xdna_device,
    reset_fallback_stats,
    reset_mapped_cpu_stats,
    synchronize,
    try_register_xdna_device,
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
    "reset_fallback_stats",
    "fallback_stats",
    "mapped_cpu_stats",
    "reset_mapped_cpu_stats",
    "native_library_path",
    "is_xrt_backed",
    "bo_handle",
    "coherency_state",
]
