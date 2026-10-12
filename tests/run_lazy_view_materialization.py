#!/usr/bin/env python3
"""Filing regression: lazy cat must materialize through arbitrary storage aliases.

Expected: eager and lazy modes match CPU for views and shared-storage growth.
Current at 6211134: eager passes, lazy raises for views/alias growth.
Run: PYTHONPATH=<repo> python tests/run_lazy_view_materialization.py
"""
import os
import sys
import torch
import xdna_train

xdna_train.register_xdna_device()

def cat(device):
    a = torch.arange(128, dtype=torch.float32).reshape(1, 32, 2, 2).to(
        device=device, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    b = (torch.arange(128, dtype=torch.float32) + 1000).reshape(1, 32, 2, 2).to(
        device=device, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    return torch.cat((a, b), 1)

cases = {
    "permute": lambda t: t.permute(0, 2, 3, 1),
    "transpose": lambda t: t.transpose(1, 3),
    "storage alias": lambda t: t.as_strided((t.numel(),), (1,)),
    "narrow": lambda t: t.narrow(1, 0, 16),
}
failures = []
for lazy in ("0", "1"):
    os.environ["XDNA_LAZY_DECODER_VALUES"] = lazy
    for name, transform in cases.items():
        reference = transform(cat("cpu")).clone()
        candidate = cat("xdna")
        try:
            actual = transform(candidate).cpu()
            ok = torch.equal(actual, reference)
            detail = f"equal={ok}"
        except Exception as e:
            ok = False
            detail = f"{type(e).__name__}: {e}"
        print(f"lazy={lazy} {name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
        if not ok:
            failures.append((lazy, name))
    # Growing through a different view of the storage must not lose the cat.
    reference = cat("cpu")
    candidate = cat("xdna")
    try:
        ref_view = reference.as_strided((reference.numel(),), (1,))
        ref_view.resize_(reference.numel() + 1)
        view = candidate.as_strided((candidate.numel(),), (1,))
        view.resize_(candidate.numel() + 1)
        ok = torch.equal(candidate.cpu(), reference)
        detail = f"equal={ok}"
    except Exception as e:
        ok = False
        detail = f"{type(e).__name__}: {e}"
    print(f"lazy={lazy} alias resize: {'OK' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        failures.append((lazy, "alias resize"))
print(f"failures={failures}", flush=True)
sys.exit(bool(failures))
