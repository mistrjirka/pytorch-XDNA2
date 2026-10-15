"""Provision a private, pinned IRON compiler without importing PyTorch.

Only ensure_toolchain()/main() install anything. Cache inspection and imports do
not perform I/O beyond reading local metadata. The application venv is never a
pip target. The compiler requires CPython 3.12 even when the application differs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Iterator
import warnings

MLIR_VERSION = "1.3.4"
PEANO_VERSION = "21.0.0.2026062301+cb664e8c"
PYTHON_VERSION = "3.12"
REQUIREMENTS = Path(__file__).with_name("toolchain-requirements.txt")
SCHEMA = "1"
SETUP_TIMEOUT = 900
FAILURE_COOLDOWN = 600
_attempted = False


class SetupError(RuntimeError):
    """A failed compiler setup; see the setup log for underlying tool output."""


def cache_root() -> Path:
    raw = os.environ.get("TORCH_XDNA2_CACHE")
    return (Path(raw).expanduser() if raw else
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "torch-xdna2")


def toolchain_id() -> str:
    return hashlib.sha256(SCHEMA.encode() + REQUIREMENTS.read_bytes()).hexdigest()[:16]


def toolchain_dir() -> Path:
    return cache_root() / "toolchains" / toolchain_id()


def _site(venv: Path) -> Path:
    return venv / "lib" / f"python{PYTHON_VERSION}" / "site-packages"


def _read_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def managed_python() -> Path | None:
    """Return only a completed environment. Never install or run a subprocess."""
    base = toolchain_dir()
    marker = _read_json(base / "ready.json")
    venv = base / "env"
    python = venv / "bin" / "python"
    expected = {"id": toolchain_id(), "mlir": MLIR_VERSION, "peano": PEANO_VERSION}
    if any(marker.get(k) != v for k, v in expected.items()):
        return None
    if (not python.is_file() or not os.access(python, os.X_OK)
            or not (_site(venv) / "mlir_aie/python/aie/__init__.py").is_file()
            or not (_site(venv) / "llvm-aie/bin/clang").is_file()):
        return None
    return python


def _supported_host() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SetupError("Automatic IRON setup requires Linux x86_64. Use XDNA_IRON_PYTHON for a custom compiler.")
    libc, version = platform.libc_ver()
    match = re.match(r"(\d+)\.(\d+)", version)
    if libc != "glibc" or not match or tuple(map(int, match.groups())) < (2, 35):
        raise SetupError("The pinned IRON wheel requires glibc >= 2.35 (Ubuntu 22.04 or newer).")


def _offline() -> bool:
    return os.environ.get("UV_OFFLINE", "").lower() in {"1", "true", "yes"}


def _uv_command() -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "--no-config"]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv", "--no-config"]
    raise SetupError("uv is missing. Install this package's dependencies, or run python -m pip install uv.")


@contextmanager
def _setup_lock(base: Path, timeout: float = SETUP_TIMEOUT) -> Iterator[None]:
    # flock is released by the OS on crashes; unlike a pidfile it cannot stay stale.
    import fcntl
    base.mkdir(parents=True, exist_ok=True)
    with (base / "setup.lock").open("a") as lock:
        deadline = time.monotonic() + timeout
        waiting = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waiting:
                    waiting = True
                    print("torch-xdna2: another process is setting up the NPU kernel compiler; "
                          "waiting for it to finish...", file=sys.stderr, flush=True)
                if time.monotonic() >= deadline:
                    raise SetupError(f"Another compiler setup is still running; see {base / 'setup.log'}")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _install_environment(base: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Never inherit a target interpreter or site-packages from the training venv.
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", UV_NO_PROGRESS="1",
               UV_PYTHON_INSTALL_DIR=str(base.parent / "python"))
    return env


def compiler_environment(venv: Path) -> dict[str, str]:
    """Paths apply only to compiler children, never to the training process."""
    from .diagnostics import find_xrt_paths
    env = _install_environment(venv.parent)
    site = _site(venv)
    mlir, peano = site / "mlir_aie", site / "llvm-aie"
    inc, lib = find_xrt_paths()
    roots = [Path(env[k]) for k in ("XRT_ROOT", "XILINX_XRT") if env.get(k)]
    roots += [Path("/opt/xilinx/xrt"), Path("/usr")]
    bins = [venv / "bin", mlir / "bin"] + [p / "bin" for p in roots]
    # Peano must load its matching LLVM, not a system libLLVM of another ABI.
    libs = [peano / "lib", mlir / "lib"] + ([lib] if lib is not None else [])
    py_paths = [mlir / "python"]
    # Add XRT's Python bindings only when they match the compiler Python ABI.
    candidates = [p / "python" for p in roots] + [
        Path("/usr/lib/python3/dist-packages"), Path("/usr/lib/python3.12/site-packages")]
    for directory in candidates:
        if directory.is_dir() and any(
            p.name == "pyxrt.so" or ".cpython-312-" in p.name
            for p in directory.glob("pyxrt*.so")
        ):
            py_paths.append(directory)
    env["PATH"] = os.pathsep.join(str(p) for p in bins if p.is_dir()) + os.pathsep + env.get("PATH", "")
    lib_parts = [str(p) for p in libs if p.is_dir()]
    lib_parts.extend(p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_parts)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in py_paths)
    env["MLIR_AIE_INSTALL_DIR"] = str(mlir)
    env["PEANO_INSTALL_DIR"] = str(peano)
    env["NPU2"] = "1"
    return env


def _run(command: list[str], env: dict[str, str], log, timeout: float = SETUP_TIMEOUT) -> None:
    print("$ " + " ".join(command), file=log, flush=True)
    try:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError(f"Compiler setup command failed ({type(exc).__name__}); see {log.name}") from exc


def _probe(venv: Path, log) -> None:
    code = (
        "from importlib.metadata import version; "
        f"assert version('mlir-aie') == {MLIR_VERSION!r}; "
        f"assert version('llvm-aie') == {PEANO_VERSION!r}; "
        "from aie.iron import WorkerRuntimeBarrier, ExternalFunction, CompileTime; "
        "from aie.dialects.aiex import v8bfp16ebs8; "
        "import aie.iron as iron; assert callable(iron.jit); "
        "print('IRON imports OK')"
    )
    env = compiler_environment(venv)
    _run([str(venv / "bin/python"), "-c", code], env, log, 60)
    _run([str(_site(venv) / "llvm-aie/bin/clang"), "--version"], env, log, 30)


def ensure_toolchain(*, retry: bool = False) -> Path:
    """Install pinned wheels in an owned venv; serialize competing processes."""
    cached = managed_python()
    if cached is not None:
        return cached
    _supported_host()
    if _offline():
        raise SetupError("IRON is not cached and UV_OFFLINE is enabled. Run torch-xdna2-setup while online first.")
    uv = _uv_command()
    base = toolchain_dir()
    with _setup_lock(base):
        cached = managed_python()
        if cached is not None:
            return cached
        failure = _read_json(base / "failed.json")
        failed_at = failure.get("time", 0)
        if not isinstance(failed_at, (int, float)):
            failed_at = 0
        if not retry and time.time() - failed_at < FAILURE_COOLDOWN:
            raise SetupError(f"Recent IRON setup failed. See {base / 'setup.log'}; retry with torch-xdna2-setup --retry.")
        venv = base / "env"
        owner = base / "owner.json"
        if venv.exists():
            if _read_json(owner).get("id") != toolchain_id() or venv.is_symlink():
                raise SetupError(f"Refusing to replace an unrecognized compiler environment: {venv}")
            shutil.rmtree(venv)
        (base / "ready.json").unlink(missing_ok=True)
        _atomic_json(owner, {"id": toolchain_id()})
        started = time.monotonic()
        print("torch-xdna2: setting up the NPU kernel compiler (first run only: ~410 MB download "
              "and install, can take several minutes)...\n"
              f"  It lives in a private cache; the training environment is unchanged. Log: {base / 'setup.log'}\n"
              "  Set XDNA_AUTO_SETUP=0 before import to disable automatic downloads.",
              file=sys.stderr, flush=True)
        try:
            with (base / "setup.log").open("a") as log:
                print(f"\nSetup started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", file=log)
                env = _install_environment(base)
                _run(uv + ["venv", "--python", PYTHON_VERSION, "--no-project", str(venv)], env, log)
                _run(uv + ["pip", "install", "--python", str(venv / "bin/python"),
                           "--only-binary", ":all:", "--require-hashes", "-r", str(REQUIREMENTS)], env, log)
                # Older wheels place aie below mlir_aie/python without a usable .pth.
                (_site(venv) / "torch_xdna2_iron.pth").write_text("mlir_aie/python\n")
                _probe(venv, log)
            _atomic_json(base / "ready.json", {"id": toolchain_id(), "mlir": MLIR_VERSION, "peano": PEANO_VERSION})
            (base / "failed.json").unlink(missing_ok=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            _atomic_json(base / "failed.json", {"time": time.time(), "error": str(exc)})
            raise SetupError(f"IRON setup failed: {exc}") from exc
        print(f"torch-xdna2: NPU kernel compiler is ready (set up in {time.monotonic() - started:.0f} s).",
              file=sys.stderr, flush=True)
        return venv / "bin/python"


def automatic_python() -> Path | None:
    """Best-effort setup once per process; caller can continue using CPU paths."""
    global _attempted
    cached = managed_python()
    if cached is not None:
        return cached
    if _attempted or os.environ.get("XDNA_AUTO_SETUP", "1") == "0":
        return None
    _attempted = True
    try:
        return ensure_toolchain()
    except (OSError, RuntimeError) as exc:
        warnings.warn(f"torch-xdna2: {exc} General 3x3 convolution will use CPU fallback. "
                      "See docs/system-dependencies.md and torch-xdna2-doctor.", RuntimeWarning, stacklevel=2)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the private XDNA2 convolution compiler; never uses sudo.")
    parser.add_argument("--check", action="store_true", help="Only check the managed compiler cache; no downloads")
    parser.add_argument("--retry", action="store_true", help="Retry a failed setup without waiting for the cooldown")
    args = parser.parse_args(argv)
    if args.check:
        path = managed_python()
        print(str(path) if path else "Managed IRON compiler not installed. Run torch-xdna2-setup.")
        return 0 if path else 1
    try:
        print(ensure_toolchain(retry=args.retry))
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"torch-xdna2-setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
