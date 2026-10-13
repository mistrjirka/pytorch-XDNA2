"""Dependency diagnostics for the XDNA2 backend.

Standard library only: nothing here imports torch or xdna_train, opens the NPU,
dlopens a library, compiles or downloads anything. Run the doctor with
``python -m xdna_tools.diagnostics [--json]``; ``require_runtime_dependencies``
raises one aggregated RuntimeError for everything that blocks the backend.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import sys
import sysconfig
import warnings
from pathlib import Path

DOCS = "https://github.com/mistrjirka/pytorch-XDNA2/blob/main/docs/system-dependencies.md"
TORCH_SERIES = "2.14"
PYTHON_MIN, PYTHON_MAX = (3, 12), (3, 14)
MEMLOCK_ADVISORY_BYTES = 1 << 30
NPU_CHECK = "npu-device"  # the one warning that still blocks the runtime

# Module constants so tests can point the search at a temporary tree.
_SYSTEM_INCLUDE_DIRS = ("/usr/include",)
_SYSTEM_LIB_DIRS = ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu")
_OPT_XRT = "/opt/xilinx/xrt"
_DEVICE_DIR = "/dev/accel"
_WRAPPERS = {"ccache", "sccache", "distcc"}

_TOOLCHAIN_HINT = (
    "Arch: pacman -S base-devel python | Ubuntu/Debian: apt install build-essential python3-dev"
    " | Fedora: dnf install gcc-c++ python3-devel libgomp"
)
_XRT_HINT = (
    "Install XRT development files (xrt/xrt_device.h and the unversioned libxrt_coreutil.so) "
    "from the AMD XDNA driver stack (https://github.com/amd/xdna-driver); a generic 'xrt' "
    "package alone is not enough for the NPU. Otherwise export XRT_INCLUDE_DIR / XRT_LIB_DIR "
    "or XRT_ROOT. See " + DOCS
)


def _entry(name: str, status: str, detail: str, hint: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def _py_version() -> tuple[int, int]:
    return sys.version_info[:2]


def _memlock_soft() -> int | None:
    try:
        import resource

        return resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    except (ImportError, OSError, ValueError):
        return None


def _glob(dirs, pattern: str) -> list[Path]:
    found: list[Path] = []
    for directory in dirs:
        try:
            found.extend(sorted(Path(directory).glob(pattern)))
        except OSError:
            continue
    return found


def _roots() -> list[Path]:
    return [Path(os.environ[v]).expanduser() for v in ("XRT_ROOT", "XILINX_XRT") if os.environ.get(v)]


def _include_dirs() -> list[Path]:
    return [r / "include" for r in _roots()] + [Path(d) for d in _SYSTEM_INCLUDE_DIRS] + [Path(_OPT_XRT) / "include"]


def _lib_dirs() -> list[Path]:
    roots = [r / sub for r in _roots() for sub in ("lib", "lib64")]
    return roots + [Path(d) for d in _SYSTEM_LIB_DIRS] + [Path(_OPT_XRT) / "lib"]


def _has_header(directory: Path) -> bool:
    return (directory / "xrt" / "xrt_device.h").is_file()


def _has_lib(directory: Path) -> bool:
    return (directory / "libxrt_coreutil.so").is_file()  # follows symlinks: broken ones do not count


def _locate(var: str, candidates, found) -> Path | None:
    """An explicit variable is authoritative: a bad value yields None, never a fallback."""
    explicit = os.environ.get(var)
    if explicit:
        path = Path(explicit).expanduser()
        return path if found(path) else None
    return next((d for d in candidates if found(d)), None)


def find_xrt_paths() -> tuple[Path | None, Path | None]:
    """Return (include root containing xrt/xrt_device.h, directory with libxrt_coreutil.so)."""
    return (
        _locate("XRT_INCLUDE_DIR", _include_dirs(), _has_header),
        _locate("XRT_LIB_DIR", _lib_dirs(), _has_lib),
    )


def _check_platform() -> dict:
    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        return _entry("platform", "ok", f"{system} {machine}")
    return _entry("platform", "error", f"{system} {machine}", "XDNA2 NPUs are supported on Linux x86_64 only.")


def _check_python() -> dict:
    version = _py_version()
    text = f"Python {version[0]}.{version[1]}"
    if PYTHON_MIN <= version <= PYTHON_MAX:
        return _entry("python", "ok", text)
    return _entry("python", "error", f"{text} (need 3.12-3.14)", "Create a virtualenv with Python 3.12, 3.13 or 3.14.")


def _check_torch() -> dict:
    hint = f"pip install 'torch=={TORCH_SERIES}.*' (only the metadata is read; torch is never imported here)"
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return _entry("torch", "error", "torch is not installed in this Python environment", hint)
    if re.match(re.escape(TORCH_SERIES) + r"\.\d+(?:\+.*)?$", version):
        return _entry("torch", "ok", f"torch {version}")
    return _entry("torch", "error", f"torch {version} installed, {TORCH_SERIES}.x required", hint)


def _check_ninja() -> dict:
    path = shutil.which("ninja")
    if path:
        return _entry("ninja", "ok", path)
    if importlib.util.find_spec("ninja") is not None:  # find_spec does not import the package
        return _entry("ninja", "ok", "python package 'ninja'")
    return _entry(
        "ninja", "error", "ninja not on PATH and python package 'ninja' not installed",
        "pip install ninja | Arch: pacman -S ninja | Ubuntu/Fedora: install ninja-build",
    )


def _check_compiler() -> dict:
    cxx = os.environ.get("CXX", "").strip()
    if not cxx:
        found = shutil.which("c++") or shutil.which("g++")
        if found:
            return _entry("compiler", "ok", found)
        return _entry("compiler", "error", "no C++ compiler (CXX unset, c++ and g++ not on PATH)", _TOOLCHAIN_HINT)
    try:
        argv = shlex.split(cxx)
    except ValueError as exc:
        return _entry("compiler", "error", f"CXX={cxx!r} cannot be parsed: {exc}", "Fix or unset CXX.")
    tools = argv[:1]
    if argv and Path(argv[0]).name in _WRAPPERS and len(argv) > 1 and not argv[1].startswith("-"):
        tools.append(argv[1])  # ccache g++: both the wrapper and the compiler must exist
    missing = [t for t in tools if not shutil.which(t)]
    if missing:
        return _entry("compiler", "error", f"CXX={cxx!r}: {', '.join(missing)} not found",
                      "Fix or unset CXX (it is not silently replaced). " + _TOOLCHAIN_HINT)
    return _entry("compiler", "ok", f"CXX={cxx}")


def _check_python_headers() -> dict:
    dirs = sorted({d for k in ("include", "platinclude") if (d := sysconfig.get_path(k))})
    hits = [Path(d) / "Python.h" for d in dirs if (Path(d) / "Python.h").is_file()]
    if hits:
        return _entry("python-headers", "ok", str(hits[0]))
    return _entry("python-headers", "error", f"Python.h not found in {', '.join(dirs) or 'sysconfig include path'}",
                  _TOOLCHAIN_HINT)


def _xrt_entry(name: str, path: Path | None, var: str, what: str, searched, extra: str = "") -> dict:
    if path is not None:
        return _entry(name, "ok", f"{what} in {path}")
    if os.environ.get(var):
        detail = f"{var}={os.environ[var]} does not contain {what}"
        return _entry(name, "error", detail, f"Fix or unset {var}; it is not overridden silently. " + _XRT_HINT)
    detail = f"{what} not found in {', '.join(map(str, searched))}"
    return _entry(name, "error", detail + extra, _XRT_HINT)


def _check_xrt_headers() -> dict:
    include, _ = find_xrt_paths()
    return _xrt_entry("xrt-headers", include, "XRT_INCLUDE_DIR", "xrt/xrt_device.h", _include_dirs())


def _check_xrt_library() -> dict:
    _, lib = find_xrt_paths()
    dirs = _lib_dirs()
    versioned = _glob(dirs, "libxrt_coreutil.so.*")
    extra = f"; only the runtime soname {versioned[0]} exists, the -dev/devel files are missing" if versioned else ""
    return _xrt_entry("xrt-library", lib, "XRT_LIB_DIR", "libxrt_coreutil.so", dirs, extra)


def _check_device() -> dict:
    nodes = sorted(Path(_DEVICE_DIR).glob("accel*"))
    advice = (
        "Enable the NPU/IPU in BIOS/UEFI, use a kernel with the amdxdna driver (Linux 6.14+) or the "
        "xdna-driver DKMS module, and check `ls -l /dev/accel` and `dmesg | grep -i amdxdna`."
    )
    if not nodes:
        return _entry(NPU_CHECK, "warning", f"no {_DEVICE_DIR}/accel* device node", advice)
    usable = [n for n in nodes if os.access(n, os.R_OK | os.W_OK)]
    if usable:
        return _entry(NPU_CHECK, "ok", ", ".join(map(str, usable)))
    return _entry(
        NPU_CHECK, "error", f"{nodes[0]} exists but is not readable and writable by this user",
        "Add your user to the group that owns the node (see `ls -l /dev/accel`) and log in again, "
        "or install a udev rule granting that group access. This tool never changes permissions.",
    )


def _check_memlock() -> dict:
    soft = _memlock_soft()
    if soft is None or soft < 0 or soft >= MEMLOCK_ADVISORY_BYTES:
        return _entry("memlock", "ok", "unlimited or not limited" if soft in (None, -1) else f"{soft} bytes")
    return _entry(
        "memlock", "warning", f"RLIMIT_MEMLOCK soft limit is {soft // 1024} KiB (advisory)",
        "Large NPU buffers may fail to pin. Check `ulimit -l`; ask your administrator for a "
        "workload-sized per-user memlock limit. See " + DOCS,
    )


def _check_plugin() -> dict:
    dirs = _lib_dirs() + ([Path(os.environ["XRT_LIB_DIR"])] if os.environ.get("XRT_LIB_DIR") else [])
    hits = _glob(dirs, "libxrt_driver_xdna.so*")
    if hits:
        return _entry("xrt-plugin", "ok", str(hits[0]))
    return _entry(
        "xrt-plugin", "warning", "libxrt_driver_xdna.so* not found (advisory; the native build can still work)",
        "The XRT amdxdna plugin from the xdna-driver stack is needed at run time to open the NPU.",
    )


def _check_xclbinutil() -> dict:
    path = shutil.which("xclbinutil")
    path = path or next((str(p) for p in _glob([r / "bin" for r in _roots()] + [Path(_OPT_XRT) / "bin"], "xclbinutil")), None)
    if path:
        return _entry("xclbinutil", "ok", path)
    return _entry(
        "xclbinutil", "warning", "xclbinutil not found (advisory; the native build can still work)",
        "Part of the XRT tools (e.g. /opt/xilinx/xrt/bin); needed only to package xclbin files.",
    )


def _check_iron() -> dict:
    from . import toolchain
    raw = os.environ.get("XDNA_IRON_PYTHON")
    if raw is not None:
        if not raw:
            return _entry("convolution-compiler", "ok", "Compilation explicitly disabled by XDNA_IRON_PYTHON=''.")
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return _entry("convolution-compiler", "ok", f"External interpreter: {path} (not executed by doctor)")
        return _entry("convolution-compiler", "warning", f"XDNA_IRON_PYTHON is not executable: {path}",
                      "Correct or unset XDNA_IRON_PYTHON; the explicit override disables automatic setup.")
    cached = toolchain.managed_python()
    if cached:
        return _entry("convolution-compiler", "ok", f"Managed compiler: {cached}")
    for path in (Path(__file__).resolve().parents[2] / "mlir-aie-env/bin/python",
                 Path.home() / "mlir-aie-env/bin/python", Path.home() / "ironenv/bin/python"):
        if path.is_file() and os.access(path, os.X_OK):
            return _entry("convolution-compiler", "ok", f"Existing compiler: {path} (not executed by doctor)")
    if os.environ.get("XDNA_AUTO_SETUP", "1") == "0" or toolchain._offline():
        return _entry("convolution-compiler", "warning", "No compiler found; automatic setup is disabled/offline.",
                      "Cached programs can run. New general-convolution shapes use CPU fallback.")
    if not shutil.which("uv") and importlib.util.find_spec("uv") is None:
        return _entry("convolution-compiler", "warning", "No compiler or uv installer found.",
                      "Install this package with dependencies, or install uv, then run torch-xdna2-setup.")
    return _entry("convolution-compiler", "ok", "Not installed yet; first backend use prepares a private compiler (~410 MB).")


_CHECKS = (
    _check_platform, _check_python, _check_torch, _check_ninja, _check_compiler, _check_python_headers,
    _check_xrt_headers, _check_xrt_library, _check_device, _check_memlock, _check_plugin, _check_xclbinutil, _check_iron,
)


def probe_dependencies() -> list[dict]:
    """Run every check; each result has name, status (ok/warning/error), detail and hint."""
    results = []
    for check in _CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # a broken probe must not hide the other findings
            name = check.__name__.removeprefix("_check_").replace("_", "-")
            results.append(_entry(name, "error", f"check failed: {exc!r}"))
    return results


def _blocking(results: list[dict]) -> list[dict]:
    return [r for r in results if r["status"] == "error" or (r["name"] == NPU_CHECK and r["status"] != "ok")]


def require_runtime_dependencies() -> list[dict]:
    """Raise RuntimeError listing every blocking prerequisite; return the results otherwise."""
    results = probe_dependencies()
    blocking = _blocking(results)
    if blocking:
        lines = [f"- {r['name']}: {r['detail']}" + (f"\n  fix: {r['hint']}" if r["hint"] else "") for r in blocking]
        raise RuntimeError(
            "XDNA2 backend prerequisites are missing:\n" + "\n".join(lines)
            + f"\nRun `python -m xdna_tools.diagnostics` for the full report; see {DOCS}."
        )
    advisory = [r for r in results if r["status"] == "warning"]
    if advisory:
        warnings.warn("XDNA2 dependency advisory: " + "; ".join(
            r["detail"] + " " + r["hint"] for r in advisory
        ), RuntimeWarning, stacklevel=2)
    return results


def _format(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"[{r['status'].upper():<7}] {r['name']}: {r['detail']}")
        if r["hint"] and r["status"] != "ok":
            lines.append(f"          fix: {r['hint']}")
    blocking = _blocking(results)
    lines.append(f"{len(blocking)} blocking problem(s); see {DOCS}" if blocking else "All required prerequisites found.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the doctor report (or JSON); return 0 if the runtime could start, else 1."""
    parser = argparse.ArgumentParser(prog="python -m xdna_tools.diagnostics", description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the raw check results as JSON")
    args = parser.parse_args(argv)
    results = probe_dependencies()
    print(json.dumps(results, indent=2) if args.json else _format(results))
    return 1 if _blocking(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
