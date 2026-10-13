#!/usr/bin/env python3
"""Normalization layers whose backward does not need every gradient.

native_batch_norm_backward runs with an output mask: no weight/bias gradient
for BatchNorm2d(affine=False) and InstanceNorm, no input gradient when the
input does not require grad.  Backward must work and match the CPU for every
mask, in BF16 and FP32.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import os
import sys

os.environ.setdefault("XDNA_AUTO_SETUP", "0")

import torch
import torch.nn as nn
import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        failures.append(name)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / (b.norm() + 1e-9)).item()


def grads(make, shape, dtype, dev, input_grad=True):
    torch.manual_seed(0)
    model = make().to(dtype)
    x = torch.randn(*shape, dtype=dtype)
    torch.manual_seed(1)
    w = torch.randn(*shape, dtype=dtype)           # fixed upstream gradient
    model, x, w = model.to(dev), x.to(dev), w.to(dev)
    if input_grad:
        x.requires_grad_()
    (model(x) * w).float().sum().backward()
    out = {"x": x.grad if input_grad else None}
    for name, p in model.named_parameters():
        out[name] = p.grad
    return out


cases = [
    ("BatchNorm2d(affine=False)", lambda: nn.BatchNorm2d(4, affine=False), (8, 4, 6, 6), True),
    ("BatchNorm1d(affine=False)", lambda: nn.BatchNorm1d(4, affine=False), (8, 4), True),
    ("InstanceNorm2d()", lambda: nn.InstanceNorm2d(4), (8, 4, 6, 6), True),
    ("InstanceNorm1d()", lambda: nn.InstanceNorm1d(4), (8, 4, 6), True),
    ("InstanceNorm2d(affine=True)", lambda: nn.InstanceNorm2d(4, affine=True), (8, 4, 6, 6), True),
    ("BatchNorm2d(track_running_stats=False, affine=False)",
     lambda: nn.BatchNorm2d(4, affine=False, track_running_stats=False), (8, 4, 6, 6), True),
    ("BatchNorm2d(), input without grad (weight/bias only)", lambda: nn.BatchNorm2d(4), (8, 4, 6, 6), False),
    ("BatchNorm2d() control", lambda: nn.BatchNorm2d(4), (8, 4, 6, 6), True),
]

for dtype in (torch.bfloat16, torch.float32):
    tol = 0.03 if dtype == torch.bfloat16 else 1e-4
    for name, make, shape, input_grad in cases:
        label = f"{name} {str(dtype).split('.')[-1]}"
        try:
            ref = grads(make, shape, dtype, "cpu", input_grad)
            got = grads(make, shape, dtype, "xdna", input_grad)
        except Exception as e:
            check(label, False, f"raised {type(e).__name__}: {str(e)[:90]}")
            continue
        errs = {k: rel(got[k], ref[k]) for k in ref if ref[k] is not None}
        none_ok = all((got[k] is None) == (ref[k] is None) for k in ref)
        check(label, none_ok and all(e < tol for e in errs.values()),
              "max rel " + (f"{max(errs.values()):.4f}" if errs else "n/a"))

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(0 if not failures else 1)
