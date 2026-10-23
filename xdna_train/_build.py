from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from xdna_tools.toolchain import cache_root as _cache_root
from xdna_tools.diagnostics import find_xrt_paths

_ROOT = Path(__file__).resolve().parent
_NATIVE = _ROOT / "_native"
_SHIM = _ROOT / "_vendor" / "xrt_shim"


def _env_id() -> dict[str, object]:
    return {
        "python": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "torch_file": str(Path(torch.__file__).resolve()),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def build_native():
    sources = [
        _NATIVE / "torch_xdna_device.cpp",
        _NATIVE / "train_xrt_shim.cpp",
        _NATIVE / "conv_pack.cpp",
        _NATIVE / "region_kernels.cpp",
    ]
    hash_inputs = sources + [_NATIVE / "conv_pack.h", _SHIM / "xrt_shim.cpp", _SHIM / "xrt_shim.h",
                            Path(__file__)]
    xrt_include, xrt_lib = find_xrt_paths()
    if xrt_include is None or xrt_lib is None:
        raise RuntimeError("XRT development headers/libraries are missing. Run torch-xdna2-doctor; see docs/system-dependencies.md.")
    env = _env_id()
    env.update(xrt_include=str(xrt_include), xrt_lib=str(xrt_lib), cxx=os.environ.get("CXX", "c++"))
    h = hashlib.sha256(json.dumps(env, sort_keys=True).encode())
    for path in hash_inputs:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    fingerprint = h.hexdigest()[:16]
    module_name = f"_torch_xdna2_{fingerprint}"
    build_dir = _cache_root() / "native" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [str(_SHIM), str(xrt_include)]
    # ninja rebuilds when a source is newer than the library (e.g. after a
    # package update touched the files), so check that instead of only whether
    # the library exists: print only when a compile really is about to happen.
    library = build_dir / f"{module_name}.so"
    compiling = (not library.is_file()
                 or any(p.stat().st_mtime > library.stat().st_mtime for p in hash_inputs))
    started = time.monotonic()
    if compiling:
        print("torch-xdna2: compiling native extension (first run after install/update; "
              "takes about a minute)...", file=sys.stderr, flush=True)
    module = load(
        name=module_name,
        sources=[str(p) for p in sources],
        extra_include_paths=include_paths,
        extra_cflags=["-O2", "-std=c++20", "-fopenmp"],
        extra_ldflags=[
            f"-L{xrt_lib}",
            "-lxrt_coreutil",
            f"-Wl,-rpath,{xrt_lib}",
            "-fopenmp",
        ],
        build_directory=str(build_dir),
        verbose=os.environ.get("TORCH_XDNA2_BUILD_VERBOSE") == "1",
    )
    if compiling:
        print(f"torch-xdna2: native extension compiled in {time.monotonic() - started:.0f} s.",
              file=sys.stderr, flush=True)
    return module


def main() -> None:
    module = build_native()
    print(module.__file__)


if __name__ == "__main__":
    main()
