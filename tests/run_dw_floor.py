#!/usr/bin/env python3
"""The default conv3x3w dW floor keeps small weight gradients on the CPU.

With the NPU forward forced on, a dW below the floor (B31 128->128 16x16, 1.2 GMAC)
must not dispatch the NPU dW program (its program switch made such steps 20-45 %
slower), and one above it (B40 192->128 36x36, 5.1 GMAC) must.  DW_MIN_MACS is
deliberately left at its default.  Counts only, no timing.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import os
import sys

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")        # forward on the NPU
os.environ.setdefault("XDNA_CONV3X3W_MIN_MACS", "0")
os.environ.pop("XDNA_CONV3X3W_DW_MIN_MACS", None)

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []


def runs(kind):
    return xdna_train.mapped_cpu_stats().get(f"npu_conv3x3w.{kind}.run", {}).get("calls", 0)


def dw_dispatches(B, cin, cout, hw):
    x = torch.randn(B, cin, hw, hw).bfloat16().to("xdna")
    w = (torch.randn(cout, cin, 3, 3) * 0.05).bfloat16().to("xdna").requires_grad_()
    fwd0, dw0 = runs("fwd"), runs("dw")
    F.conv2d(x, w, padding=1).float().square().mean().backward()
    w.grad.cpu()                                          # a deferred dW runs on first read
    return runs("fwd") - fwd0, runs("dw") - dw0


for name, shape, want_dw in (("below floor", (31, 128, 128, 16), False),
                             ("above floor", (40, 192, 128, 36), True)):
    fwd, dw = dw_dispatches(*shape)
    ok = fwd > 0 and (dw > 0) == want_dw
    print(f"{name} {shape}: NPU fwd runs {fwd}, NPU dW runs {dw}: {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        failures.append(name)
print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(0 if not failures else 1)
