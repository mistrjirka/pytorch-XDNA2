#!/usr/bin/env python3
"""BF16 avg_pool3d on XDNA: PyTorch has no BF16 CPU kernel, XDNA computes it in FP32.

Reference: the same op on FP32 CPU tensors holding the BF16-rounded inputs, rounded
to BF16.  Covers padding, ceil_mode, count_include_pad, divisor_override, odd and
unit dimensions, the module (nn.AvgPool3d) and gradient accumulation.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import sys

import torch
import torch.nn.functional as F
from torch import nn

import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond, detail=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {detail}")
    if not cond:
        failures.append(name)


def run(name, shape, **kw):
    torch.manual_seed(1)
    x = torch.randn(*shape).bfloat16()
    xf = x.float().requires_grad_()
    ref = F.avg_pool3d(xf, **kw)
    g = torch.randn_like(ref).bfloat16()
    ref.backward(g.float())
    xd = x.to("xdna").requires_grad_()
    out = F.avg_pool3d(xd, **kw)
    out.backward(g.to("xdna"))
    ok_dtype = out.dtype == torch.bfloat16 and xd.grad.dtype == torch.bfloat16
    fwd = torch.allclose(out.detach().cpu().float(), ref.detach().bfloat16().float(), atol=0, rtol=0)
    bwd = torch.allclose(xd.grad.cpu().float(), xf.grad.bfloat16().float(), atol=1e-2, rtol=1e-2)
    check(name, ok_dtype and fwd and bwd, f"dtype={ok_dtype} fwd={fwd} bwd={bwd}")


run("basic 2x2x2", (2, 3, 6, 6, 6), kernel_size=2)
run("odd sizes, stride 2", (2, 3, 5, 7, 9), kernel_size=3, stride=2)
run("padding, count_include_pad=True", (1, 2, 6, 6, 6), kernel_size=3, stride=2, padding=1)
run("padding, count_include_pad=False", (1, 2, 6, 6, 6), kernel_size=3, stride=2, padding=1, count_include_pad=False)
run("ceil_mode", (1, 2, 7, 7, 7), kernel_size=3, stride=2, ceil_mode=True)
run("divisor_override", (1, 2, 6, 6, 6), kernel_size=2, divisor_override=3)
run("anisotropic kernel", (2, 2, 4, 6, 8), kernel_size=(1, 2, 4), stride=(1, 2, 2))
run("kernel covers whole volume", (1, 1, 2, 2, 2), kernel_size=2)
run("unit batch and channel", (1, 1, 4, 4, 4), kernel_size=2)

# module + gradient accumulation
torch.manual_seed(2)
x = torch.randn(2, 3, 4, 4, 4).bfloat16()
pool = nn.AvgPool3d(2)
xd = x.to("xdna").requires_grad_()
for _ in range(2):
    pool(xd).float().sum().backward()
check("accumulated gradient (module)", torch.allclose(xd.grad.cpu().float(), torch.full_like(x, 2 / 8).float()))

# FP32 still works and takes the same path
xf = torch.randn(1, 2, 4, 4, 4)
check("fp32 matches CPU", torch.allclose(F.avg_pool3d(xf.to("xdna"), 2).cpu(), F.avg_pool3d(xf, 2), atol=1e-6))

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
