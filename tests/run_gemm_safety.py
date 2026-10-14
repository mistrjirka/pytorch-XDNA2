#!/usr/bin/env python3
"""NPU GEMM safety (a few seconds on the XDNA2 machine):

1. A recycled output buffer the CPU wrote earlier (dirty cache lines) must not
   overwrite the NPU result (host memory is not coherent with NPU writes).
2. A contiguous view with a storage offset (a[1:]) must be multiplied from its
   own start, not from the start of the allocation.

The 256x768 @ 768x3072 GEMM is one of the shapes the NPU GEMM family handles;
the script checks that no mm fell back to the CPU, so it cannot pass vacuously.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import os
import sys

os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")  # counts CPU-fallback mm

import torch
import xdna_train
from xdna_train import device

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []


def check(name, cond, detail=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {detail}")
    if not cond:
        failures.append(name)


check("GEMM family configured", device._C.gemm_family_configured())
xdna_train.reset_mapped_cpu_stats()

torch.manual_seed(42)
a = (torch.randn(256, 768) * .03).bfloat16()
b = (torch.randn(768, 3072) * .03).bfloat16()
ref = (a.float() @ b.float()).bfloat16().float()
ax, bx = a.to("xdna"), b.to("xdna")

# 1. Recycled output buffer.  bf16 rounding of the NPU accumulation: ~2e-4.
for mode in ("fresh", "dirty", "dirty again", "flushed"):
    xdna_train.empty_cache()
    if mode != "fresh":
        p = torch.full((256, 3072), 42., device="xdna", dtype=torch.bfloat16)
        if mode == "flushed":
            device._C.sync_to_device(p)
        del p
    out = ax @ bx
    got = out.cpu().float()
    err = (got - ref).abs().max().item()
    check(f"recycled output ({mode})", err < 0.01 and int((got == 42).sum()) == 0,
          f"max err {err:.6f}, elements still 42: {int((got == 42).sum())}")
    del out

# 2. Storage offsets of either operand.
a0 = (torch.randn(257, 768) * .03).bfloat16()
eye = torch.zeros(768, 3072, dtype=torch.bfloat16)
eye[:, :768] = torch.eye(768, dtype=torch.bfloat16)
expected = torch.zeros(256, 3072, dtype=torch.bfloat16)
expected[:, :768] = a0[1:]
av = a0.to("xdna")[1:]
got = (av @ eye.to("xdna")).cpu()
check("lhs view with storage offset",
      av.is_contiguous() and av.storage_offset() == 768
      and (got.float() - expected.float()).abs().max().item() == 0.0,
      f"offset {av.storage_offset()}")

# rhs = rows 1.. of a taller matrix whose row 0 is garbage: lhs picks column
# block 0 of rhs, so a wrong base would multiply by the garbage row instead.
big = torch.zeros(769, 3072, dtype=torch.bfloat16)
big[0] = 7.0
big[1:, :768] = torch.eye(768, dtype=torch.bfloat16)
bv = big.to("xdna")[1:]
expected = torch.zeros(256, 3072, dtype=torch.bfloat16)
expected[:, :768] = a
got = (ax @ bv).cpu()
check("rhs view with storage offset",
      bv.is_contiguous() and bv.storage_offset() == 3072
      and (got.float() - expected.float()).abs().max().item() == 0.0,
      f"offset {bv.storage_offset()}")

# The NPU must actually have run (no CPU fallback mm).
fallbacks = xdna_train.mapped_cpu_stats().get("aten::mm.cpu_mapped", {}).get("calls", 0)
check("every mm ran on the NPU", fallbacks == 0, f"cpu fallback mm calls: {fallbacks}")

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
