#!/usr/bin/env python3
"""conv3x3w cost model: which shapes go to the NPU (fwd and dX) and which stay on the CPU.

Run with the NIS venv python and PYTHONPATH=<repo>.  The decision is read from
the `npu_conv3x3w.{fwd,dx}.run` mapped-CPU phases.  dX of a conv is the conv of
dY with swapped channels, so e.g. 64->3 (dX = 3-channel output) must stay on CPU.
"""
import os
import sys

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"
os.environ["XDNA_CONV3X3W_MIN_MACS"] = "0"
os.environ["XDNA_CONV3X3W_DW"] = "0"
os.environ.pop("XDNA_CONV3X3W_FORCE", None)

import torch  # noqa: E402
import xdna_train  # noqa: E402
from xdna_train import device as xd  # noqa: E402

xdna_train.register_xdna_device()

# B, Cin, H, Cout, expect NPU (fwd; dX follows unless Cin is 3)
CASES = [
    (40, 3, 36, 64, False, True),      # RGB input conv: both directions have a 3-channel side
    (40, 64, 36, 64, True, True),
    (40, 128, 9, 128, True, True),
    (40, 195, 36, 64, True, True),
    (40, 256, 18, 128, True, True),
    (40, 256, 36, 128, True, True),
]


def main():
    ok = True
    dev = torch.device("xdna")
    for B, ci, h, co, want_fwd, _ in CASES:
        x = torch.randn(B, ci, h, h).bfloat16().to(dev).requires_grad_()
        w = (torch.randn(co, ci, 3, 3) * 0.05).bfloat16().to(dev)
        dy = torch.randn(B, co, h, h).bfloat16().to(dev)
        xd.reset_mapped_cpu_stats()
        y = torch.nn.functional.conv2d(x, w, padding=1)
        y.backward(dy)
        st = xd.mapped_cpu_stats()
        fwd = "npu_conv3x3w.fwd.run" in st
        dx = "npu_conv3x3w.dx.run" in st
        # dX is the 3x3 conv of dY (co channels) producing ci channels
        want_dx = want_fwd if ci != 3 else False
        good = fwd == want_fwd and dx == want_dx
        ok &= good
        print(f"B{B} {ci}->{co} @{h}: npu fwd={fwd} (want {want_fwd}) dx={dx} (want {want_dx}) "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
