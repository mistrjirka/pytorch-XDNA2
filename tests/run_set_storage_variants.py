#!/usr/bin/env python3
"""Storage growth must preserve pending values and validate the device."""
import os
import sys
os.environ["XDNA_CONV3X3W_SYNC_BUILD"] = "1"
os.environ["XDNA_PROFILE_MAPPED_CPU"] = "1"
import torch
import torch.nn.functional as F
import xdna_train
xdna_train.set_npu_mode(2)
failures = []
def check(name, good, detail=""):
    print(f"{name}: {'OK' if good else 'FAIL'} {detail}", flush=True)
    if not good:
        failures.append(name)
def runs():
    return xdna_train.mapped_cpu_stats().get("npu_conv3x3w.dw.run", {}).get("calls", 0)
def gradient():
    x = torch.ones(16,192,36,36,device="xdna",dtype=torch.bfloat16)
    w = torch.ones(128,192,3,3,device="xdna",dtype=torch.bfloat16,requires_grad=True)
    go = torch.ones(16,128,36,36,device="xdna",dtype=torch.bfloat16)
    return torch.autograd.grad(F.conv2d(x,w,padding=1),w,go)[0]
reference_owner = gradient()
reference = reference_owner.cpu().flatten().clone()
check("reference finite/nonzero", bool(torch.isfinite(reference).all()) and reference.float().abs().sum().item() > 0)
held = []
for explicit in (True,False):
    before = runs()
    g = gradient()
    check(f"pending before set_, explicit={explicit}", runs() == before)
    n = g.numel()
    out = torch.empty(0,device="xdna",dtype=g.dtype)
    if explicit:
        out.set_(g.untyped_storage(),0,(n+1,),(1,))
    else:
        out.set_(g.untyped_storage(),0,(n+1,))
    got = out.cpu().flatten()[:n]
    check(f"NPU dW completed, explicit={explicit}", runs() > before)
    check(f"prefix preserved, explicit={explicit}", torch.equal(got,reference),
          f"sum={got.float().sum().item():.0f} expected={reference.float().sum().item():.0f}")
    held.extend((g,out))
# A concat that shares its storage with a grown tensor keeps its values.
a = torch.ones(1,32,2,2,device="xdna",dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
b = torch.full_like(a,2)
expected = torch.cat((a.cpu(),b.cpu()),1)
c = torch.cat((a,b),1)
out = torch.empty(0,device="xdna",dtype=c.dtype)
out.set_(c.untyped_storage(),0,(c.numel()+1,),(1,))
got = c.cpu()
check("concat survives shared storage growth", torch.equal(got,expected),
      f"sum={got.float().sum().item():.0f} expected={expected.float().sum().item():.0f}")
held.extend((a,b,c,out))
# A rejected cross-device binding must not alter its source.
source = torch.arange(4.,dtype=torch.float32)
destination = torch.empty(0,device="xdna",dtype=source.dtype)
old_bytes = source.untyped_storage().nbytes()
old_pointer = source.data_ptr()
rejected = False
try:
    destination.set_(source.untyped_storage(),0,(8,),(1,))
except RuntimeError as error:
    rejected = True
    print(f"device mismatch error: {error}", flush=True)
check("CPU storage cannot bind to XDNA tensor", rejected)
check("CPU source storage remains CPU", source.untyped_storage().device.type == "cpu")
check("rejected set_ preserves source allocation",
      source.untyped_storage().nbytes() == old_bytes and source.data_ptr() == old_pointer,
      f"bytes={source.untyped_storage().nbytes()} expected={old_bytes}")
print("ALL OK" if not failures else f"FAILED: {failures}", flush=True)
sys.exit(1 if failures else 0)
