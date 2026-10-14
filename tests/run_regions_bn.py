#!/usr/bin/env python3
"""Fused training BatchNorm2d + LeakyReLU/ReLU (regions.fuse_norm_activation).

Per case: output, input / weight / bias gradients, running statistics and
num_batches_tracked of the fused pair are compared with the eager bf16 pair on
XDNA and with an fp32 CPU reference.  Gate: the fused error vs fp32 must be at
most 1.5x the eager error plus a small floor.  Also: fallbacks, the pairing
guard, several steps of running statistics.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import copy
import sys

import torch
import torch.nn as nn

import xdna_train
from xdna_train import regions

xdna_train.register_xdna_device()
regions._NORM_ACT_MIN_BYTES = 0  # the cases below are small; exercise the fused op on all of them
ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'FAIL'} {extra}")


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def make(C, slope, seed):
    torch.manual_seed(seed)
    bn = nn.BatchNorm2d(C)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.normal_(0, 0.3)
        bn.running_mean.normal_(0, 0.1)
        bn.running_var.uniform_(0.5, 1.5)
    act = nn.LeakyReLU(slope) if slope > 0 else nn.ReLU()
    return bn, act


CASES = [(8, 64, 18, 18, 0.2), (4, 128, 9, 9, 0.2), (2, 3, 5, 7, 0.2), (1, 16, 6, 6, 0.2),
         (3, 1, 4, 4, 0.2), (6, 32, 11, 13, 0.0), (40, 64, 36, 36, 0.2)]
for N, C, H, W, slope in CASES:
    bn32, act32 = make(C, slope, 1)
    steps = 3
    xs = [torch.randn(N, C, H, W) * 1.5 + 0.3 for _ in range(steps)]
    gs = [torch.randn(N, C, H, W) for _ in range(steps)]
    # fp32 reference
    ref = copy.deepcopy(bn32)
    ref_ys, ref_dx = [], []
    for x, g in zip(xs, gs):
        xr = x.bfloat16().float().requires_grad_()
        y = act32(ref(xr))
        y.backward(g.bfloat16().float())
        ref_ys.append(y.detach())
        ref_dx.append(xr.grad)
    res = {}
    for name in ("eager", "fused"):
        bn = copy.deepcopy(bn32).to("xdna", torch.bfloat16)
        act = copy.deepcopy(act32)
        if name == "fused":
            assert regions.fuse_norm_activation(bn, act)
        ys, dxs = [], []
        for x, g in zip(xs, gs):
            xd = x.bfloat16().to("xdna").requires_grad_()
            y = act(bn(xd))
            y.backward(g.bfloat16().to("xdna"))
            ys.append(y.detach())
            dxs.append(xd.grad)
        res[name] = (ys, dxs, bn)
    def err(name):
        ys, dxs, bn = res[name]
        e_y = max(rel(a, b) for a, b in zip(ys, ref_ys))
        e_dx = max(rel(a, b) for a, b in zip(dxs, ref_dx))
        e_w = rel(bn.weight.grad, ref.weight.grad)
        e_b = rel(bn.bias.grad, ref.bias.grad)
        e_rm = rel(bn.running_mean, ref.running_mean)
        e_rv = rel(bn.running_var, ref.running_var)
        return e_y, e_dx, e_w, e_b, e_rm, e_rv
    ee, ef = err("eager"), err("fused")
    nb_ok = int(res["fused"][2].num_batches_tracked) == steps == int(res["eager"][2].num_batches_tracked)
    good = all(f <= 1.5 * e + 6e-3 for e, f in zip(ee, ef)) and nb_ok
    lab = "y dx dW db rm rv"
    report(f"BN+act N{N} C{C} {H}x{W} slope {slope}", good,
           f"| err vs fp32 ({lab}) eager " + " ".join(f"{v:.4f}" for v in ee) +
           " | fused " + " ".join(f"{v:.4f}" for v in ef))

# eager BatchNorm on a strided view (the backend used to read it with the wrong indexing)
torch.manual_seed(7)
base = torch.randn(4, 6, 6, 8).bfloat16()
bnv = nn.BatchNorm2d(8).to("xdna", torch.bfloat16)
bnc = nn.BatchNorm2d(8).to(torch.bfloat16)
xv = base.to("xdna").requires_grad_()
yv = bnv(xv.transpose(1, 3))
yc = bnc(base.transpose(1, 3))
gv = torch.randn(yc.shape).bfloat16()
yv.backward(gv.to("xdna"))
xc = base.clone().requires_grad_()
bnc(xc.transpose(1, 3)).backward(gv)
report("eager BatchNorm on a transposed view matches CPU (forward, repeated, backward)",
       rel(yv, yc) < 0.01 and rel(bnv(xv.transpose(1, 3)), yc) < 0.01 and rel(xv.grad, xc.grad) < 0.02,
       f"fwd {rel(yv, yc):.4f} dx {rel(xv.grad, xc.grad):.4f}")

# pairing guard and fallbacks
bn, act = make(8, 0.2, 2)
bn = bn.to("xdna", torch.bfloat16)
bn_e, act32 = copy.deepcopy(bn), copy.deepcopy(act)  # eager twin: own modules, never patched
regions.fuse_norm_activation(bn, act)
x = torch.randn(4, 8, 6, 6).bfloat16().to("xdna")
y_f, y_e = act(bn(x)), act32(bn_e(x))
report("activation applied exactly once (no double leaky)", rel(y_f, y_e) < 0.02, f"rel {rel(y_f, y_e):.4f}")
bn.eval(); bn_e.eval()
# running statistics differ slightly between the pairs (rounding), so eval outputs agree to ~1e-2
rel_eval = rel(act(bn(x)), act32(bn_e(x)))
report("eval mode falls back to the eager pair", rel_eval < 0.01, f"rel {rel_eval:.4f}")
bn.train()
xt = torch.randn(4, 6, 6, 8).bfloat16().to("xdna").transpose(1, 3)  # non-contiguous NCHW view
bn.train(); bn_e.train()
rel_nc = rel(act(bn(xt)), act32(bn_e(xt)))
report("non-contiguous input falls back", rel_nc < 0.01, f"rel {rel_nc:.4f}")
other = nn.BatchNorm2d(4, affine=False).to("xdna", torch.bfloat16)
report("affine=False is not fused (and works)", regions.fuse_norm_activation(other, nn.ReLU())
       and other(torch.randn(2, 4, 3, 3).bfloat16().to("xdna")) is not None)
report("unsupported activation is not paired", not regions.fuse_norm_activation(nn.BatchNorm2d(4), nn.Sigmoid()))
mom = nn.BatchNorm2d(4, momentum=None).to("xdna", torch.bfloat16)
nn_act = nn.ReLU()
regions.fuse_norm_activation(mom, nn_act)
xm = torch.randn(2, 4, 3, 3).bfloat16().to("xdna")
report("momentum=None runs the eager path", rel(nn_act(mom(xm)), torch.relu(nn.functional.batch_norm(
    xm.cpu().float(), None, None, mom.weight.float().cpu(), mom.bias.float().cpu(), True, 0.1, mom.eps))) < 0.02)
xi = torch.randn(4, 8, 6, 6).bfloat16().to("xdna").requires_grad_()
yi = act(bn(xi))
with torch.no_grad():
    xi.mul_(2)
try:
    yi.sum().backward()
    det = False
except RuntimeError as e:
    det = "modified by an inplace operation" in str(e)
report("in-place change of a saved tensor is detected", det)

# the size threshold: small tensors keep the eager pair (the fused op is not used)
regions._NORM_ACT_MIN_BYTES = 4_000_000
bs, as_ = make(8, 0.2, 3)
bs = bs.to("xdna", torch.bfloat16)
regions.fuse_norm_activation(bs, as_)
ys = bs(torch.randn(2, 8, 4, 4).bfloat16().to("xdna"))
report("small tensors use the eager pair (output untagged)", getattr(ys, "_xdna_act_for", None) is None)
big = torch.randn(40, 64, 36, 36).bfloat16().to("xdna")
bb, ab = make(64, 0.2, 4)
bb = bb.to("xdna", torch.bfloat16)
regions.fuse_norm_activation(bb, ab)
report("large tensors use the fused op (output tagged)", getattr(bb(big), "_xdna_act_for", None) is not None)

print("ALL OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
