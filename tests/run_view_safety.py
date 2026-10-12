#!/usr/bin/env python3
"""View safety: XDNA must reject the same out-of-bounds / overlapping views as CPU,
and view ops (unfold) must share storage with their base.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import sys

import torch
import xdna_train

xdna_train.register_xdna_device()
failures = []


def same_error(name, fn):
    """fn(device) must raise on CPU and on XDNA."""
    outcome = {}
    for dev in ("cpu", "xdna"):
        try:
            fn(dev)
            outcome[dev] = None
        except (RuntimeError, ValueError) as e:
            outcome[dev] = type(e).__name__
    ok = outcome["cpu"] is not None and outcome["xdna"] is not None
    print(f"{name}: cpu {outcome['cpu'] or 'accepts'}, xdna {outcome['xdna'] or 'accepts'} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


same_error("as_strided past end of storage",
           lambda d: torch.zeros(8, device=d).as_strided((16,), (1,)))
same_error("as_strided negative storage offset",
           lambda d: torch.zeros(8, device=d).as_strided((4,), (1,), -1))
same_error("as_strided offset past end",
           lambda d: torch.zeros(8, device=d).as_strided((2,), (1,), 7))
same_error("copy_ between overlapping slices",
           lambda d: (lambda t: t[1:5].copy_(t[0:4]))(torch.arange(8.0, device=d)))
same_error("copy_ into self-overlapping (expanded) destination",
           lambda d: torch.zeros(1, device=d).expand(4).copy_(torch.arange(4.0, device=d)))

# In-bounds views keep working.
t = torch.arange(8.0, device="xdna")
check("as_strided in bounds", torch.equal(t.as_strided((2, 2), (4, 1), 2).cpu(),
                                          torch.arange(8.0).as_strided((2, 2), (4, 1), 2)))
check("as_strided empty view at end", t.as_strided((0,), (1,), 8).numel() == 0)
check("copy_ between disjoint slices",
      torch.equal((lambda u: (u[4:8].copy_(u[0:4]), u)[1])(torch.arange(8.0, device="xdna")).cpu(),
                  torch.tensor([0., 1, 2, 3, 0, 1, 2, 3])))
check("copy_ onto itself", torch.equal((lambda u: (u.copy_(u), u)[1])(torch.arange(4.0, device="xdna")).cpu(),
                                       torch.arange(4.0)))

# unfold is a view: writes through it reach the base.
for dev in ("cpu", "xdna"):
    base = torch.arange(10.0, device=dev)
    u = base.unfold(0, 4, 2)
    u[1, 0] = 100.0
    check(f"unfold shares storage ({dev})", base[2].item() == 100.0 and u._base is base)
x = torch.arange(24.0).reshape(2, 3, 4)
check("unfold values (2d, dim 2)", torch.equal(x.to("xdna").unfold(2, 2, 1).cpu(), x.unfold(2, 2, 1)))

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
