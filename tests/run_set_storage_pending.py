#!/usr/bin/env python3
"""set_(Storage) must preserve a pending gradient when it grows that storage.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>. The reference and the
resize_ control retain their XDNA allocations, so the failing path cannot
accidentally recycle their already-correct gradient buffers.
"""
import os
import sys

os.environ["XDNA_CONV3X3W_SYNC_BUILD"] = "1"
os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []
DW_KEY = "npu_conv3x3w.dw.run"


def check(name, condition, detail=""):
    print(f"{name}: {'OK' if condition else 'FAIL'} {detail}", flush=True)
    if not condition:
        failures.append(name)


def dw_calls():
    return xdna_train.mapped_cpu_stats().get(DW_KEY, {}).get("calls", 0)


def gradient():
    x = torch.ones(16, 192, 36, 36, device="xdna", dtype=torch.bfloat16)
    w = torch.ones(128, 192, 3, 3, device="xdna",
                   dtype=torch.bfloat16, requires_grad=True)
    go = torch.ones(16, 128, 36, 36, device="xdna", dtype=torch.bfloat16)
    return torch.autograd.grad(F.conv2d(x, w, padding=1), w, go)[0]


def pending_gradient(label):
    before = dw_calls()
    g = gradient()
    check(f"{label}: dW remains pending before storage growth",
          dw_calls() == before, f"calls {before} -> {dw_calls()}")
    return g, before


def prefix_matches(label, got, ref):
    check(label, torch.equal(got, ref),
          f"sum {got.float().sum().item():.0f} vs {ref.float().sum().item():.0f}")


# CPU control: set_(Storage) may enlarge shared storage and preserves its prefix.
cpu_base = torch.arange(1, 9, dtype=torch.bfloat16)
cpu_expected = cpu_base.clone()
cpu_out = torch.empty(0, dtype=cpu_base.dtype)
cpu_out.set_(cpu_base.untyped_storage(), 0, (cpu_base.numel() + 1,), (1,))
prefix_matches("CPU set_(Storage) growth preserves the prefix",
               cpu_out[:cpu_base.numel()], cpu_expected)
check("CPU set_(Storage) keeps shared storage",
      cpu_out.untyped_storage()._cdata == cpu_base.untyped_storage()._cdata)

# Read one gradient normally to establish the exact NPU reference.
xdna_train.reset_mapped_cpu_stats()
before = dw_calls()
reference_owner = gradient()
reference = reference_owner.cpu().flatten().clone()
check("reference: dW ran on the NPU", dw_calls() > before,
      f"calls {before} -> {dw_calls()}")
check("reference: finite and nonzero",
      bool(torch.isfinite(reference).all()) and reference.float().abs().sum().item() > 0)

# Passing control: the regular resize_ path already fences the pending writer.
direct, before = pending_gradient("resize_ control")
n = direct.numel()
direct.resize_(n + 1)
direct_prefix = direct.cpu().flatten()[:n]
check("resize_ control: pending dW ran on the NPU", dw_calls() > before,
      f"calls {before} -> {dw_calls()}")
prefix_matches("resize_ control preserves the prefix", direct_prefix, reference)

# Regression: set_storage_cpu_ can grow the same storage through its own
# raw-copy path. It must wait before copying, just like resize_ above.
pending, before = pending_gradient("set_(Storage)")
n = pending.numel()
out = torch.empty(0, device="xdna", dtype=pending.dtype)
out.set_(pending.untyped_storage(), 0, (n + 1,), (1,))
got = out.cpu().flatten()[:n]
check("set_(Storage): pending dW ran on the NPU", dw_calls() > before,
      f"calls {before} -> {dw_calls()}")
check("set_(Storage): storage grew and remains shared",
      out.untyped_storage()._cdata == pending.untyped_storage()._cdata
      and pending.untyped_storage().nbytes() >= (n + 1) * pending.element_size())
prefix_matches("set_(Storage) growth preserves the pending gradient", got, reference)

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
