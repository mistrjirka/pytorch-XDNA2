"""Compiler child entry point; executed by path to work in editable installs too."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xdna_tools.toolchain import compiler_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("design")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python = Path(args.python).absolute()  # Do not resolve the venv symlink.
    env = compiler_environment(python.parent.parent)
    os.execve(str(python), [str(python), args.design, *args.args], env)


if __name__ == "__main__":
    main()
