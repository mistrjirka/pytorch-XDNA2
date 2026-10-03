from __future__ import annotations

import torch
import xdna_train


def main() -> None:
    xdna_train.register_xdna_device()
    print(f"torch={torch.__version__}")
    print(f"xdna_available={xdna_train.is_available()}")
    print(f"native={xdna_train.native_library_path()}")
    print(f"allocator={xdna_train.allocator_stats()}")


if __name__ == "__main__":
    main()
