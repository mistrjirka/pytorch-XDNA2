#!/usr/bin/env python3
"""Fast cat / nearest-2x upsample (+backward) vs CPU ATen.

Run with the NIS venv python and PYTHONPATH=<repo>.  cat and upsample forward
are pure copies: exact.  upsample backward sums each 2x2 block in fp32 in
ATen's order (start at 0, raster order) and rounds once: bit-exact too.
"""
import sys

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
dev = torch.device("xdna")
ok = True


def check(name, good):
    global ok
    ok &= bool(good)
    print(f"{name}: {'OK' if good else 'FAIL'}", flush=True)


def same(a, b):
    a, b = a.cpu(), b.cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    u = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    # -0.0 vs +0.0 differ in bits; ATen's own results are compared bitwise too
    return torch.equal(a.contiguous().view(u), b.contiguous().view(u))


def rnd(*shape, dt):
    return (torch.randn(*shape) * 3).to(dt)


for dt in (torch.bfloat16, torch.float32):
    tag = str(dt).split(".")[1]
    # cat, NCHW contiguous, several dims and sizes (incl. odd and size-1)
    for shapes, dim in [
        ([(40, 128, 36, 36), (40, 64, 36, 36), (40, 3, 36, 36)], 1),
        ([(2, 5, 7, 9), (2, 1, 7, 9)], 1),
        ([(3, 4, 5, 6), (3, 4, 5, 2)], 3),
        ([(3, 4, 5, 6), (1, 4, 5, 6)], 0),
        ([(3, 4, 5, 6), (3, 4, 2, 6), (3, 4, 1, 6)], -2),
        ([(1, 1, 1, 1), (1, 3, 1, 1)], 1),
        ([(2, 0, 3, 3), (2, 4, 3, 3)], 1),
        ([(6, 7), (6, 2)], 1),
    ]:
        ts = [rnd(*s, dt=dt) for s in shapes]
        check(f"cat {tag} {shapes} dim{dim}",
              same(torch.cat([t.to(dev) for t in ts], dim), torch.cat(ts, dim)))
    # non-contiguous / channels_last inputs go the fallback path
    a, b = rnd(2, 6, 5, 5, dt=dt), rnd(2, 3, 5, 5, dt=dt)
    check(f"cat {tag} channels_last",
          same(torch.cat([a.to(dev).contiguous(memory_format=torch.channels_last),
                          b.to(dev).contiguous(memory_format=torch.channels_last)], 1),
               torch.cat([a.contiguous(memory_format=torch.channels_last),
                          b.contiguous(memory_format=torch.channels_last)], 1)))
    check(f"cat {tag} non-contiguous (transposed)",
          same(torch.cat([a.to(dev).transpose(2, 3), b.to(dev).transpose(2, 3)], 1),
               torch.cat([a.transpose(2, 3), b.transpose(2, 3)], 1)))
    check(f"cat {tag} sliced",
          same(torch.cat([a.to(dev)[:, ::2], b.to(dev)], 1), torch.cat([a[:, ::2], b], 1)))

    # upsample forward / backward
    for n, c, h, w in [(40, 256, 18, 18), (2, 5, 7, 9), (1, 1, 1, 1), (3, 7, 4, 1), (2, 64, 3, 5)]:
        for cl in (False, True):
            x = rnd(n, c, h, w, dt=dt)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            ref = F.interpolate(x, scale_factor=2, mode="nearest")
            got = F.interpolate(x.to(dev), scale_factor=2, mode="nearest")
            check(f"upsample {tag} {(n, c, h, w)} cl={cl}", same(got, ref))
            ref = torch.ops.aten.upsample_nearest2d(x, [2 * h, 2 * w], None, None)
            got = torch.ops.aten.upsample_nearest2d(x.to(dev), [2 * h, 2 * w], 2.0, 2.0)
            check(f"upsample {tag} {(n, c, h, w)} cl={cl} explicit scales", same(got, ref))
            g = rnd(n, c, 2 * h, 2 * w, dt=dt)
            # -0.0 inputs: ATen starts the sum at +0
            g = g.masked_fill(torch.rand(g.shape) < 0.3, -0.0)
            if cl:
                g = g.contiguous(memory_format=torch.channels_last)
            ref = torch.ops.aten.upsample_nearest2d_backward(g, [2 * h, 2 * w], [n, c, h, w], None, None)
            got = torch.ops.aten.upsample_nearest2d_backward(g.to(dev), [2 * h, 2 * w], [n, c, h, w], None, None)
            check(f"upsample_backward {tag} {(n, c, h, w)} cl={cl}", same(got, ref))
    # fallbacks: non-2x, non-contiguous
    x = rnd(2, 3, 5, 6, dt=dt)
    check(f"upsample {tag} 3x fallback",
          same(F.interpolate(x.to(dev), scale_factor=3, mode="nearest"),
               F.interpolate(x, scale_factor=3, mode="nearest")))
    check(f"upsample {tag} size 11x12 fallback",
          same(F.interpolate(x.to(dev), size=(11, 12), mode="nearest"),
               F.interpolate(x, size=(11, 12), mode="nearest")))
    xt = x.transpose(2, 3)
    check(f"upsample {tag} non-contiguous",
          same(F.interpolate(xt.to(dev), scale_factor=2, mode="nearest"),
               F.interpolate(xt, scale_factor=2, mode="nearest")))
    g = rnd(2, 3, 10, 12, dt=dt).transpose(2, 3)
    ref = torch.ops.aten.upsample_nearest2d_backward(g, [12, 10], [2, 3, 6, 5], None, None)
    got = torch.ops.aten.upsample_nearest2d_backward(g.to(dev), [12, 10], [2, 3, 6, 5], None, None)
    check(f"upsample_backward {tag} non-contiguous", same(got, ref))
    g3 = rnd(2, 3, 15, 18, dt=dt)
    check(f"upsample_backward {tag} 3x fallback",
          same(torch.ops.aten.upsample_nearest2d_backward(g3.to(dev), [15, 18], [2, 3, 5, 6], None, None),
               torch.ops.aten.upsample_nearest2d_backward(g3, [15, 18], [2, 3, 5, 6], None, None)))


# autograd: cat + upsample model vs CPU
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = torch.nn.Conv2d(8 + 3 + 8, 4, 3, padding=1)

    def forward(self, x):
        a = torch.relu(self.c1(x))
        low = F.avg_pool2d(a, 2)
        up = F.interpolate(low, scale_factor=2, mode="nearest")
        return self.c2(torch.cat([a, x, up], 1))


for dt in (torch.float32, torch.bfloat16):
    torch.manual_seed(0)
    m = Net().to(dt)
    x = torch.randn(4, 3, 12, 12).to(dt)
    m2 = Net().to(dt)
    m2.load_state_dict(m.state_dict())
    m2 = m2.to(dev)
    xr = x.clone().requires_grad_()
    xg = x.to(dev).requires_grad_()
    yr = m(xr)
    yg = m2(xg)
    w = torch.randn_like(yr)
    yr.backward(w)
    yg.backward(w.to(dev))
    tol = 3e-2 if dt == torch.bfloat16 else 1e-4
    def rel(a, b):
        return ((a.float().cpu() - b.float()).norm() / b.float().norm()).item()
    errs = [rel(yg.detach(), yr.detach()), rel(xg.grad, xr.grad)] + [
        rel(p2.grad, p1.grad) for p1, p2 in zip(m.parameters(), m2.parameters())]
    check(f"autograd cat+upsample {dt} max rel err {max(errs):.2e}", max(errs) < tol)

print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
