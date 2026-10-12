#!/usr/bin/env python3
"""Coherency across storage lifetimes (a few seconds on the XDNA2 machine):

1. A CPU-fallback temporary (served from the BO cache inside an
   AdoptHostAllocations scope) that is freed without becoming an output was
   written by the CPU.  Its BO must come back HostDirty, or the next NPU GEMM
   that recycles the address gets the stale cache lines written over its result.
2. resize_() of a gradient whose host write is still pending (deferred NPU dW)
   must carry the finished gradient into the new storage, not the stale bytes.

Each check asserts that the hazard was actually set up (same address reused,
NPU kernel ran), so it cannot pass vacuously.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import os
import sys

os.environ["XDNA_CONV3X3W_SYNC_BUILD"] = "1"  # build NPU streams up front
os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"    # counts CPU-fallback mm and NPU dW runs

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []


def check(name, cond, detail=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {detail}")
    if not cond:
        failures.append(name)


def calls(key):
    return xdna_train.mapped_cpu_stats().get(key, {}).get("calls", 0)


# --- 1. freed CPU-fallback temporary, then an NPU GEMM on the same address ---
lib = torch.library.Library("coherency_lifetime", "DEF")
lib.define("scratch(Tensor x) -> Tensor")
pointers = []


def scratch(x):
    p = torch.full((256, 3072), 42., dtype=torch.bfloat16)  # CPU-written, then freed
    pointers.append(p.data_ptr())
    del p
    return x + 1


lib.impl("scratch", scratch, "CPU")

torch.manual_seed(42)
a = (torch.randn(256, 768) * .03).bfloat16()
b = (torch.randn(768, 3072) * .03).bfloat16()
ref = (a.float() @ b.float()).bfloat16().float()
ax, bx = a.to("xdna"), b.to("xdna")
x = torch.ones(1, device="xdna")
reused = wrong = 0
worst = 0.0
xdna_train.reset_mapped_cpu_stats()
for _ in range(3):
    xdna_train.empty_cache()
    torch.ops.coherency_lifetime.scratch(x)
    out = ax @ bx
    if out.data_ptr() == pointers[-1]:
        reused += 1
        got = out.cpu().float()
        err = (got - ref).abs().max().item()  # bf16 rounding of the NPU accumulation: ~2e-4
        worst = max(worst, err)
        wrong += err >= 0.01 or int((got == 42).sum()) != 0
check("freed CPU temporary: address reused by the GEMM output", reused > 0,
      f"{reused}/3 attempts" if reused else "reuse not reproduced")
check("freed CPU temporary: stale cache lines do not reach the GEMM result",
      reused > 0 and wrong == 0, f"{wrong}/{reused} reused attempts wrong, max err {worst:.6f}")
fallbacks = calls("aten::mm.cpu_mapped")
check("freed CPU temporary: GEMM ran on the NPU", fallbacks == 0,
      f"cpu fallback mm calls: {fallbacks}")


# --- 2. resize_ of a gradient whose deferred dW has not run yet ---
def gradient():
    x = torch.ones(16, 192, 36, 36, device="xdna", dtype=torch.bfloat16)
    w = torch.ones(128, 192, 3, 3, device="xdna", dtype=torch.bfloat16, requires_grad=True)
    go = torch.ones(16, 128, 36, 36, device="xdna", dtype=torch.bfloat16)
    return torch.autograd.grad(F.conv2d(x, w, padding=1), w, go)[0]


g = gradient()
ref_g = g.cpu().flatten().clone()
del g
runs = calls("npu_conv3x3w.dw.run")
g = gradient()
n = g.numel()
g.resize_(n + 1)  # before the first read of g
got = g.cpu().flatten()[:n]
check("resize_ of a pending dW: dW ran on the NPU", calls("npu_conv3x3w.dw.run") > runs)
check("resize_ of a pending dW: reference is not trivially zero", ref_g.float().sum().item() != 0.0)
check("resize_ of a pending dW: prefix survives", torch.equal(got, ref_g),
      f"sum {got.float().sum().item():.0f} vs {ref_g.float().sum().item():.0f}")

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
