from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

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


def _cache_root() -> Path:
    raw = os.environ.get("TORCH_XDNA2_CACHE")
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "torch-xdna2"


def build_native():
    sources = [
        _NATIVE / "torch_xdna_device.cpp",
        _NATIVE / "train_xrt_shim.cpp",
        _NATIVE / "conv_pack.cpp",
    ]
    hash_inputs = sources + [_SHIM / "xrt_shim.cpp", _SHIM / "xrt_shim.h"]
    env = _env_id()
    h = hashlib.sha256(json.dumps(env, sort_keys=True).encode())
    for path in hash_inputs:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    fingerprint = h.hexdigest()[:16]
    module_name = f"_torch_xdna2_{fingerprint}"
    build_dir = _cache_root() / "native" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [str(_SHIM)]
    xrt_include = os.environ.get("XRT_INCLUDE_DIR")
    if xrt_include:
        include_paths.append(xrt_include)

    xrt_lib = os.environ.get("XRT_LIB_DIR", "/usr/lib")
    return load(
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


def main() -> None:
    module = build_native()
    print(module.__file__)


if __name__ == "__main__":
    main()
