#!/usr/bin/env python3
"""Training regions (xdna_train/regions.py): frozen_feature_mse vs the eager loss.

For several chains / shapes the region's loss and input gradient are compared
with (a) the eager bf16 path on XDNA (clone taps, cat, sub, pow, mean) and (b) an
fp32 CPU reference.  Gate: the region must be at least as close to fp32 as the
eager bf16 path (times 1.5 plus a small floor), and close to the eager result.
Also checks the max-pool/ReLU kernels bit-exactly against torch, the fallbacks,
and that backward can run twice (retain_graph).

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import sys

import torch
import torch.nn as nn

import xdna_train
from xdna_train import regions

xdna_train.register_xdna_device()
C = xdna_train.device._C
ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'FAIL'} {extra}")


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def build(layers_spec, seed):
    torch.manual_seed(seed)
    layers = []
    for s in layers_spec:
        if s[0] == "conv":
            cin, cout, k = s[1], s[2], s[3]
            c = nn.Conv2d(cin, cout, k, padding=k // 2)
            nn.init.kaiming_normal_(c.weight)
            nn.init.normal_(c.bias, 0, 0.1)
            layers.append(c)
        elif s[0] == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.MaxPool2d(2, 2))
    for l in layers:
        for p in l.parameters():
            p.requires_grad = False
    return nn.Sequential(*layers)


def eager_loss(feats, taps, x, t):
    def run(v):
        out, y = [], v
        for i in range(max(taps) + 1):
            y = feats[i](y)
            if i in taps:
                out.append(y.clone().view(y.shape[0], -1))
        return torch.cat(out, dim=1)
    fx = run(x)
    with torch.no_grad():
        ft = run(t)
    return ((fx - ft) ** 2).mean()


VGG = [("conv", 3, 64, 3), ("relu",), ("conv", 64, 64, 3), ("relu",), ("pool",), ("conv", 64, 128, 3)]
CASES = [
    ("vgg prefix taps 0,3,5  B8 36x36", VGG, (0, 3, 5), (8, 3, 36, 36)),
    ("vgg prefix taps 0,3,5  B4 odd 35x35", VGG, (0, 3, 5), (4, 3, 35, 35)),
    ("taps 1,2,4 (relu, conv, pool outputs) B4 24x24", VGG, (1, 2, 4), (4, 3, 24, 24)),
    ("single tap 5, B2 16x16", VGG, (5,), (2, 3, 16, 16)),
    ("standalone pool + 1x1 conv, taps 2,4",
     [("conv", 3, 16, 3), ("pool",), ("conv", 16, 24, 1), ("relu",), ("conv", 24, 24, 3)], (1, 4), (3, 3, 20, 20)),
]

for name, spec, taps, shape in CASES:
    feats32 = build(spec, 1)
    feats = nn.Sequential(*[(__import__("copy").deepcopy(l)) for l in feats32]).to("xdna", torch.bfloat16)
    chain = regions.FrozenFeatureChain.from_sequential(feats, taps)
    torch.manual_seed(2)
    x32 = torch.randn(*shape)
    t32 = torch.rand(*shape) * 2 - 1
    x = x32.bfloat16().to("xdna").requires_grad_()
    t = t32.bfloat16().to("xdna")
    # fp32 reference from the same (bf16-rounded) inputs and weights
    ref = nn.Sequential(*[__import__("copy").deepcopy(l) for l in feats32])
    for l, lb in zip(ref, feats):
        for p, pb in zip(l.parameters(), lb.parameters()):
            p.data = pb.data.cpu().float()
    xr = x.detach().cpu().float().requires_grad_()
    lr = eager_loss(ref, taps, xr, t.cpu().float())
    lr.backward()
    # eager bf16 on XDNA
    xe = x.detach().clone().requires_grad_()
    le = eager_loss(feats, taps, xe, t)
    le.backward()
    # region
    xg = x.detach().clone().requires_grad_()
    lg = regions.frozen_feature_mse(chain, xg, t)
    lg.backward(retain_graph=True)
    g1 = xg.grad.clone()
    xg.grad = None
    lg.backward()
    # measured placement alternates CPU/NPU between calls, so a repeat differs by rounding
    same = rel(xg.grad.cpu(), g1.cpu()) < 0.01
    e_loss_eager = abs(le.float().item() - lr.item()) / lr.item()
    e_loss_region = abs(lg.float().item() - lr.item()) / lr.item()
    e_grad_eager = rel(xe.grad.cpu(), xr.grad)
    e_grad_region = rel(xg.grad.cpu(), xr.grad)
    d_loss = abs(lg.float().item() - le.float().item()) / lr.item()
    d_grad = rel(xg.grad.cpu(), xe.grad.cpu())
    good = (e_loss_region <= 1.5 * e_loss_eager + 5e-3 and e_grad_region <= 1.5 * e_grad_eager + 5e-3
            and same and d_grad < 0.05)
    report(name, good,
           f"loss err vs fp32: eager {e_loss_eager:.4f} region {e_loss_region:.4f} | "
           f"grad rel err vs fp32: eager {e_grad_eager:.4f} region {e_grad_region:.4f} | "
           f"region-vs-eager grad {d_grad:.4f}, repeat backward equal {same}")

# the measured placement settles after enough calls and stays accurate
feats32 = build(VGG, 1)
featsd = nn.Sequential(*[__import__("copy").deepcopy(l) for l in feats32]).to("xdna", torch.bfloat16)
chain = regions.FrozenFeatureChain.from_sequential(featsd, (0, 3, 5))
xs = torch.randn(8, 3, 36, 36).bfloat16().to("xdna")
ts = torch.rand(8, 3, 36, 36).bfloat16().to("xdna")
grads = []
for _ in range(16):
    xi = xs.clone().requires_grad_()
    regions.frozen_feature_mse(chain, xi, ts).backward()
    grads.append(xi.grad.cpu())
choices = {k: v["choice"] for k, v in chain.placer.state.items()}
# conv0 (3 input channels) may never reach the NPU: it stays in its trial phase (<= 48
# calls, both placements are CPU then) and then falls back to the cost model.
settled = {k: c for k, c in choices.items() if k[0] != 0}
report("placer settles (convs the NPU can run are decided)",
       len(settled) == 4 and all(c is not None for c in settled.values()),
       str({f"{k[1]}{k[0]}": {1: "npu", 2: "cpu", 0: "cost"}[c] for k, c in choices.items() if c is not None}))
report("results stay consistent across placements", all(rel(g, grads[0]) < 0.02 for g in grads[1:]))

# saved tensors go through save_for_backward: modifying one in place before backward raises
featsm = nn.Sequential(*[__import__("copy").deepcopy(l) for l in feats32]).to("xdna", torch.bfloat16)
chm = regions.FrozenFeatureChain.from_sequential(featsm, (0, 3, 5))
xm = torch.randn(2, 3, 16, 16).bfloat16().to("xdna").requires_grad_()
lm = regions.frozen_feature_mse(chm, xm, torch.rand(2, 3, 16, 16).bfloat16().to("xdna"))
with torch.no_grad():
    xm.mul_(2)  # x is saved for backward
try:
    lm.backward()
    detected = False
except RuntimeError as e:
    detected = "modified by an inplace operation" in str(e)
report("in-place change of a saved tensor is detected", detected)

# the placer decides per input shape, not per op alone
chs = regions.FrozenFeatureChain.from_sequential(featsm, (0, 3, 5))
for shp in ((2, 3, 16, 16), (3, 3, 20, 20)):
    xs_ = torch.randn(*shp).bfloat16().to("xdna").requires_grad_()
    regions.frozen_feature_mse(chs, xs_, torch.rand(*shp).bfloat16().to("xdna")).backward()
shapes = {k[2] for k in chs.placer.state}
report("placer keys include the input shape", any(sh[0] == 2 for sh in shapes) and any(sh[0] == 3 for sh in shapes))

# kernel-level checks against torch
torch.manual_seed(3)
a = (torch.randn(3, 5, 12, 10) * 2).bfloat16()
a[0, 0, 0, :4] = float("nan")
a[1, 1, 2:4, 2:4] = 0  # ties and zeros
r = torch.empty_like(a, device="xdna")
p = torch.empty(3, 5, 6, 5, dtype=torch.bfloat16, device="xdna")
C.region_relu_maxpool2(a.to("xdna"), r, p)
rr = torch.relu(a)
pp = torch.nn.functional.max_pool2d(rr, 2, 2)
def same_bits(x, y):
    """Identical bits, except that any two NaNs are equal (payloads differ by op)."""
    x, y = x.cpu(), y.cpu()
    nx, ny = torch.isnan(x), torch.isnan(y)
    return torch.equal(nx, ny) and torch.equal(x.view(torch.int16)[~nx], y.view(torch.int16)[~ny])


report("relu+maxpool2 kernel bit-exact (incl. NaN, ties)", same_bits(r, rr) and same_bits(p, pp))
g = torch.randn(3, 5, 6, 5).bfloat16()
out = torch.empty_like(a, device="xdna")
C.region_maxpool2_bwd(rr.to("xdna"), g.to("xdna"), out)
xr = rr.clone().requires_grad_()
torch.nn.functional.max_pool2d(xr, 2, 2).backward(g)
report("maxpool2 backward bit-exact (scatter to first max)",
       torch.equal(out.cpu().view(torch.int16), xr.grad.view(torch.int16)))
a2 = (torch.randn(4, 7, 9, 9)).bfloat16()
b2 = (torch.randn(4, 7, 9, 9)).bfloat16()
base = torch.randn(4, 7, 9, 9).bfloat16()
o = torch.empty_like(a2, device="xdna")
C.region_relu_bwd(a2.to("xdna").clone(), base.to("xdna"), a2.to("xdna"), b2.to("xdna"), o, 0.37)
want = torch.where(a2.float() <= 0, torch.zeros(()), base.float() + 0.37 * (a2.float() - b2.float())).bfloat16()
report("relu_bwd + tap kernel", torch.equal(o.cpu().view(torch.int16), want.view(torch.int16)))

# fallbacks and opt-out
class FakeLoss(nn.Module):
    def __init__(self, feats, taps, dtype):
        super().__init__()
        self.perceptual_vgg = nn.Module()
        self.perceptual_vgg.model = nn.Module()
        self.perceptual_vgg.model.features = feats
        self.perceptual_vgg.feature_layers = taps
        self.perceptual_vgg.use_norm = False
        self.called = 0

    def compute_perceptual_loss(self, input, target):
        self.called += 1
        return eager_loss(self.perceptual_vgg.model.features, self.perceptual_vgg.feature_layers, input, target)


feats = build(VGG, 1).to("xdna", torch.bfloat16)
fl = FakeLoss(feats, (0, 3, 5), torch.bfloat16)
report("fuse_perceptual_loss installs on the pattern", regions.fuse_perceptual_loss(fl))
x = torch.randn(2, 3, 16, 16).bfloat16().to("xdna").requires_grad_()
t = torch.rand(2, 3, 16, 16).bfloat16().to("xdna")
fl.compute_perceptual_loss(x, t)
report("region used for eligible inputs (eager not called)", fl.called == 0)
fl.compute_perceptual_loss(x.detach(), t)  # no grad needed -> still a region? no: grad mode on, x has no grad
report("falls back when x does not require grad", fl.called == 1)
fl2 = FakeLoss(build(VGG + [("pool",), ("conv", 128, 8, 3)], 1).to("xdna", torch.bfloat16), (0, 3, 5), None)
fl2.perceptual_vgg.model.features[0].weight.requires_grad_()
report("trainable weights are not eligible", not regions.fuse_perceptual_loss(fl2))
bad = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Sigmoid()).to("xdna", torch.bfloat16)
for p in bad.parameters():
    p.requires_grad = False
report("unsupported op is not eligible", regions.FrozenFeatureChain.from_sequential(bad, (1,)) is None)

print("ALL OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
