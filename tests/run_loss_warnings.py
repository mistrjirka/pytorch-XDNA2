#!/usr/bin/env python3
"""Loss functions must give the CPU's values, gradients and warnings.

mse_loss and smooth_l1_loss with mean/sum reduction used to reach the CPU
kernel through its out= overload with a 0-d out tensor, which PyTorch resizes
with a UserWarning ("An output with one or more elements was resized ...").
CPU users never see it.  Every loss below must produce the same number of
warnings on XDNA as on the CPU (none), and the same values and gradients.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import os
import sys
import warnings

os.environ.setdefault("XDNA_AUTO_SETUP", "0")

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        failures.append(name)


def t(dev, dt, *s, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*s, generator=g).to(dt).to(dev)
    return x.requires_grad_() if requires_grad else x


def sign(dev, dt, *s, seed=5):
    return torch.sign(t(dev, dt, *s, seed=seed) - .5)


def labels(dev, n, c):
    g = torch.Generator().manual_seed(3)
    return torch.randint(0, c, (n,), generator=g).to(dev)


LOSSES = {
    "mse_loss": lambda d, dt, r, x: F.mse_loss(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "l1_loss": lambda d, dt, r, x: F.l1_loss(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "smooth_l1_loss": lambda d, dt, r, x: F.smooth_l1_loss(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "smooth_l1_loss(beta=.3)": lambda d, dt, r, x: F.smooth_l1_loss(x, t(d, dt, 8, 16, seed=1), reduction=r, beta=0.3),
    "huber_loss": lambda d, dt, r, x: F.huber_loss(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "binary_cross_entropy": lambda d, dt, r, x: F.binary_cross_entropy(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "bce_with_logits": lambda d, dt, r, x: F.binary_cross_entropy_with_logits(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "cross_entropy": lambda d, dt, r, x: F.cross_entropy(x[:, :10], labels(d, 8, 10), reduction=r),
    "nll_loss": lambda d, dt, r, x: F.nll_loss(F.log_softmax(x[:, :10], 1), labels(d, 8, 10), reduction=r),
    "soft_margin": lambda d, dt, r, x: F.soft_margin_loss(x, sign(d, dt, 8, 16), reduction=r),
    "poisson_nll": lambda d, dt, r, x: F.poisson_nll_loss(x, t(d, dt, 8, 16, seed=1), reduction=r),
    "multi_margin": lambda d, dt, r, x: F.multi_margin_loss(x[:, :10], labels(d, 8, 10), reduction=r),
}

# mse_loss and smooth_l1_loss are the ones that used to warn; the rest guard
# against the same problem elsewhere.
for dtype in (torch.bfloat16, torch.float32):
    tol = 0.02 if dtype == torch.bfloat16 else 1e-5
    for name, fn in LOSSES.items():
        for red in ("mean", "sum", "none"):
            label = f"{name} {red} {str(dtype).split('.')[-1]}"
            res = {}
            for dev in ("cpu", "xdna"):
                x = t(dev, dtype, 8, 16, seed=2, requires_grad=True)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        loss = fn(dev, dtype, red, x)
                        (loss if loss.dim() == 0 else (loss * torch.arange(loss.numel(), dtype=loss.dtype, device=dev)
                                                       .view_as(loss)).sum()).backward()
                        res[dev] = (loss.detach().float().cpu(), x.grad.float().cpu(), len(w))
                    except Exception as e:
                        res[dev] = (None, None, f"ERR {type(e).__name__}: {str(e)[:60]}")
            if res["cpu"][0] is None or res["xdna"][0] is None:
                check(label, res["cpu"][2] == res["xdna"][2], f"cpu {res['cpu'][2]} xdna {res['xdna'][2]}")
                continue
            (lc, gc, wc), (lx, gx, wx) = res["cpu"], res["xdna"]
            def rel(a, b):
                return ((a - b).norm() / (b.norm() + 1e-9)).item()
            ok = (wx == wc and lx.shape == lc.shape and rel(lx, lc) < tol and rel(gx, gc) < tol)
            check(label, ok, f"warnings cpu {wc} xdna {wx}; loss rel {rel(lx, lc):.4f} grad rel {rel(gx, gc):.4f}")

print("ALL OK" if not failures else f"FAILED ({len(failures)}): {failures[:6]}")
sys.exit(0 if not failures else 1)
