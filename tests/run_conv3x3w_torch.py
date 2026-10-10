#!/usr/bin/env python3
"""conv3x3w through PyTorch: forward and dX on XDNA vs an FP32 CPU reference.

Run with the NIS venv python and PYTHONPATH=<repo>.  Sets
XDNA_CONV3X3W_SYNC_BUILD=1 so missing streams are compiled before the first
call instead of falling back to the CPU; checks that the NPU path ran.
"""
import os
import sys

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")
os.environ.setdefault("XDNA_CONV3X3W_MIN_MACS", "0")
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")  # tiny shapes must still hit the NPU
os.environ.setdefault("XDNA_CONV3X3W_DW_MIN_MACS", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import xdna_train  # noqa: E402
from xdna_train import device as xd  # noqa: E402

SHAPES = [  # B, Cin, H, W, Cout, bias
    (40, 256, 36, 36, 128, False),
    (40, 64, 36, 36, 64, True),
    (4, 195, 36, 36, 70, False),
    (40, 128, 9, 9, 128, False),
    (40, 256, 18, 18, 256, False),
    (3, 3, 5, 7, 10, True),
]


xdna_train.register_xdna_device()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def counts():
    st = xd.mapped_cpu_stats()
    return (st.get("npu_conv3x3w.weights", {}).get("calls", 0),
            st.get("npu_conv3x3w.weights_hit", {}).get("calls", 0))


def fwd_dx(conv, x, dy):
    """(y, dx, dw) of one conv step on xdna, bit patterns as CPU tensors."""
    xx = x.clone().requires_grad_()
    conv.weight.grad = None
    y = conv(xx)
    y.backward(dy)
    return y.detach().cpu(), xx.grad.cpu(), conv.weight.grad.cpu()


def weight_cache_test(dev):
    """The encoded-weight cache must follow every in-place weight update path."""
    good = True
    torch.manual_seed(1)
    B, C, H, W = 4, 64, 12, 12
    x = torch.randn(B, C, H, W).bfloat16().to(dev)
    dy = torch.randn(B, C, H, W).bfloat16().to(dev)

    def uncached(conv):
        os.environ["XDNA_CONV3X3W_WEIGHT_CACHE"] = "0"
        try:
            return fwd_dx(conv, x, dy)
        finally:
            os.environ.pop("XDNA_CONV3X3W_WEIGHT_CACHE")

    def same(a, b):
        return all(torch.equal(u.view(torch.int16), v.view(torch.int16)) for u, v in zip(a, b))

    def update_paths():
        def add_(w, o): w.add_(0.05)
        def mul_(w, o): w.mul_(1.5)
        def copy_(w, o): w.copy_(o)
        def addcmul(w, o): w.addcmul_(o, o, value=0.1)
        def fill_(w, o): w.fill_(0.03)
        def data_add(w, o): w.data.add_(0.05)
        return [("add_", add_), ("mul_", mul_), ("copy_", copy_), ("addcmul_", addcmul),
                ("fill_", fill_), ("data.add_", data_add)]

    # optimizers: every step the cached output must equal the uncached one
    for name, mk in [
        ("SGD", lambda p: torch.optim.SGD(p, lr=0.5)),
        ("SGD+mom+wd", lambda p: torch.optim.SGD(p, lr=0.5, momentum=0.9, weight_decay=0.1)),
        ("Adam", lambda p: torch.optim.Adam(p, lr=0.05)),
        ("AdamW", lambda p: torch.optim.AdamW(p, lr=0.05)),
    ]:
        conv = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).bfloat16().to(dev)
        opt = mk(conv.parameters())
        prev, steps_ok = None, True
        for step in range(4):
            xd.reset_mapped_cpu_stats()
            r = fwd_dx(conv, x, dy)
            enc, hit = counts()
            r0 = uncached(conv)
            steps_ok &= same(r, r0)
            steps_ok &= prev is None or not torch.equal(prev, r[0])
            steps_ok &= enc == 2 and hit == 0  # fwd + dx, both stale after the update
            prev = r[0]
            opt.zero_grad()
            conv(x).backward(dy)
            opt.step()
        print(f"weight cache {name}: {'OK' if steps_ok else 'FAIL'}", flush=True)
        good &= steps_ok

    # explicit in-place update paths
    conv = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).bfloat16().to(dev)
    other = torch.randn_like(conv.weight)
    for name, fn in update_paths():
        fwd_dx(conv, x, dy)
        v0 = conv.weight._version
        with torch.no_grad():
            fn(conv.weight, other)
        r = fwd_dx(conv, x, dy)
        r0 = uncached(conv)
        res = same(r, r0)
        if name == "data.add_":
            # .data has its own version counter: not tracked, reported only
            print(f"weight cache w.data.add_: version bumped={conv.weight._version != v0} "
                  f"stale={'no' if res else 'YES (known limitation)'}", flush=True)
            fwd_dx(conv, x, dy)
            with torch.no_grad():
                conv.weight.add_(0.0)  # counter bump recovers
            continue
        print(f"weight cache {name}: version {v0}->{conv.weight._version} {'OK' if res else 'FAIL'}", flush=True)
        good &= res and conv.weight._version != v0

    # frozen weight: encoded once (1 block) per direction, then hits
    conv = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).bfloat16().to(dev)
    conv.weight.requires_grad_(False)
    xx = x.clone().requires_grad_()
    xd.reset_mapped_cpu_stats()
    for _ in range(5):
        conv(xx).backward(dy)
    enc, hit = counts()
    ok = enc == 2 and hit == 8
    print(f"frozen weight: encodes={enc} hits={hit} {'OK' if ok else 'FAIL'}", flush=True)
    good &= ok

    # two layers sharing one stream do not evict each other
    c1 = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).bfloat16().to(dev)
    c2 = torch.nn.Conv2d(C, C, 3, padding=1, bias=False).bfloat16().to(dev)
    c1(x), c2(x)
    xd.reset_mapped_cpu_stats()
    y1, y2 = c1(x), c2(x)
    enc, hit = counts()
    ref1 = F.conv2d(x.cpu().float(), c1.weight.detach().cpu().float(), padding=1)
    ok = enc == 0 and hit == 2 and rel(y1.cpu(), ref1) < 0.02
    print(f"shared stream layers: encodes={enc} hits={hit} {'OK' if ok else 'FAIL'}", flush=True)
    good &= ok
    return good


def main():
    dev = torch.device("xdna")
    torch.manual_seed(0)
    ok = True
    for B, Cin, H, W, Cout, use_bias in SHAPES:
        x = torch.randn(B, Cin, H, W).bfloat16()
        w = (torch.randn(Cout, Cin, 3, 3) * (2.0 / (9 * Cin)) ** 0.5).bfloat16()
        b = torch.randn(Cout).bfloat16() if use_bias else None
        dy = torch.randn(B, Cout, H, W).bfloat16()

        xr = x.float().requires_grad_()
        wr = w.float().requires_grad_()
        yr = F.conv2d(xr, wr, None if b is None else b.float(), padding=1)
        yr.backward(dy.float())

        xd_ = x.to(dev).requires_grad_()
        wd = w.to(dev).requires_grad_()
        bd = None if b is None else b.to(dev)
        xd.reset_mapped_cpu_stats()
        y = F.conv2d(xd_, wd, bd, padding=1)
        y.backward(dy.to(dev))
        stats = xd.mapped_cpu_stats()
        ran_fwd = "npu_conv3x3w.fwd.run" in stats
        ran_dx = "npu_conv3x3w.dx.run" in stats
        e_y, e_dx, e_dw = rel(y.cpu(), yr), rel(xd_.grad.cpu(), xr.grad), rel(wd.grad.cpu(), wr.grad)
        # dW is deferred: it runs when the gradient is first read (above).
        ran_dw = "npu_conv3x3w.dw.run" in xd.mapped_cpu_stats()
        good = ran_fwd and ran_dx and ran_dw and e_y < 0.02 and e_dx < 0.02 and e_dw < 0.02
        ok &= good
        print(f"B{B} Cin{Cin} {H}x{W} Cout{Cout} bias={use_bias}: fwd {100*e_y:.2f}% "
              f"dx {100*e_dx:.2f}% dw {100*e_dw:.2f}% npu fwd={ran_fwd} dx={ran_dx} dw={ran_dw} {'OK' if good else 'FAIL'}",
              flush=True)
    ok &= weight_cache_test(dev)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
