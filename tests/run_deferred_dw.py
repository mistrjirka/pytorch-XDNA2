#!/usr/bin/env python3
"""An NPU conv weight gradient is deferred (run at the optimizer), so it must
not read X / dY after backward returned: mutating either afterwards must not
change the returned gradient.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import os
import sys

os.environ["XDNA_CONV3X3W_SYNC_BUILD"] = "1"  # build NPU streams up front
os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"    # NPU call counters
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")  # small shapes must still hit the NPU
os.environ.setdefault("XDNA_CONV3X3W_MIN_MACS", "0")
os.environ.setdefault("XDNA_CONV3X3W_DW_MIN_MACS", "0")

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []

BIG = (16, 192, 128, 36, 36)  # B, Cin, Cout, H, W: the audit's shape
WIDE = (2, 64, 128, 20, 64)   # W > 36: dY is repacked with zero halos (tiled)


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


def calls(key):
    return xdna_train.mapped_cpu_stats().get(key, {}).get("calls", 0)


def operands(shape, ones):
    b, cin, cout, h, w = shape
    g = torch.Generator().manual_seed(0)
    mk = torch.ones if ones else (lambda *s: torch.randn(*s, generator=g))
    return [mk(*s).bfloat16() for s in ((b, cin, h, w), (cout, cin, 3, 3), (b, cout, h, w))]


def dw(shape, mutate=None, ones=False, with_dx=False, dev="xdna"):
    """Weight gradient of one conv; `mutate` ("dy" / "x") zeroes that operand
    after backward returned, before the gradient is read."""
    x, w, go = (t.to(dev) for t in operands(shape, ones))
    w.requires_grad_()
    x.requires_grad_(with_dx)
    gw = torch.autograd.grad(F.conv2d(x, w, padding=1), [w, x][:1 + with_dx], go)[0]
    if mutate:
        {"dy": go, "x": x}[mutate].detach().zero_()
    return gw.cpu().float()


def close(got, ref):
    return ((got - ref).norm() / ref.norm()).item() < 0.03


def case(name, shape, key, **kw):
    """Mutating an operand must not change the gradient; `key` counted calls
    prove the path under test ran (a CPU fallback would be worthless)."""
    ref = dw(shape, **kw)
    before = calls(key)
    for mutate in ("dy", "x"):
        got = dw(shape, mutate, **kw)
        check(f"{name}: mutate {mutate}", torch.equal(got, ref) and ref.abs().sum() > 0)
    check(f"{name}: ran {key}", calls(key) > before)
    check(f"{name}: matches CPU", close(ref, dw(shape, dev="cpu", **kw)))


# The audit's reproduction (ones; autograd.grad(w) only, so no dX run packed dY).
before = calls("npu_conv3x3w.dw.run")
ref = dw(BIG, ones=True)
check("audit repro: NPU dW ran, equals CPU",
      calls("npu_conv3x3w.dw.run") > before and torch.equal(ref, dw(BIG, ones=True, dev="cpu")))
check("audit repro: dY zeroed after autograd.grad",
      torch.equal(dw(BIG, "dy", ones=True), ref) and ref.abs().sum() > 0)

case("npu dW", BIG, "npu_conv3x3w.dw.run")
case("npu dW + dX", BIG, "npu_conv3x3w.dw.run", with_dx=True)
case("npu dW, tiled width", WIDE, "npu_conv3x3w.dw.run", with_dx=True)

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
