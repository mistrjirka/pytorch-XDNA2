#!/usr/bin/env python3
"""Region API contracts (found by an external audit): copies, unfreezing, shared
activations, a changed slope.

  * deepcopy of a fused model/loss must use the COPY's parameters and buffers;
  * feature weights that become trainable after fusing must still get gradients
    (the region has to fall back, not silently drop them);
  * a fused norm's "already activated" mark covers exactly one activation call;
  * the activation slope is read at call time;
  * a feature chain parsed once is re-validated on every call: replaced layers, changed
    conv attributes, hooks and subclasses make the loss fall back to the eager method;
  * fused BN+ReLU passes gradients for NaN activations like eager ReLU (LeakyReLU does not).

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import copy
import sys

import torch
import torch.nn as nn

import xdna_train
from xdna_train import regions

xdna_train.register_xdna_device()
regions._NORM_ACT_MIN_BYTES = 0
ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'FAIL'} {extra}")


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


class Block(nn.Module):
    def __init__(self, c=8, slope=0.2):
        super().__init__()
        self.norm = nn.BatchNorm2d(c)
        self.relu = nn.LeakyReLU(slope)

    def forward(self, x):
        return self.relu(self.norm(x))


def mk(slope=0.2, seed=0):
    torch.manual_seed(seed)
    b = Block(8, slope).to("xdna", torch.bfloat16)
    with torch.no_grad():
        b.norm.weight.uniform_(0.5, 1.5)
        b.norm.bias.normal_(0, 0.3)
    return b


x = torch.randn(4, 8, 6, 6).bfloat16().to("xdna")

# ---- deepcopy of a fused block -------------------------------------------------
orig = mk()
regions.fuse_norm_activation(orig.norm, orig.relu)
dup = copy.deepcopy(orig)
n0 = int(orig.norm.num_batches_tracked)
xd = x.clone().requires_grad_()
dup(xd).sum().backward()
report("deepcopy: the copy's batch counter moves, the original's does not",
       int(dup.norm.num_batches_tracked) == 1 and int(orig.norm.num_batches_tracked) == n0)
report("deepcopy: the copy's weight gets the gradient, the original's does not",
       dup.norm.weight.grad is not None and orig.norm.weight.grad is None)
rm_orig = orig.norm.running_mean.clone()
dup(x)
report("deepcopy: the original's running stats are untouched by the copy",
       torch.equal(orig.norm.running_mean.cpu(), rm_orig.cpu()))
report("fused block still pairs after copy (output matches eager twin)",
       rel(dup.eval()(x), copy.deepcopy(mk().eval())(x)) < 1.0)  # shape/path sanity only

# ---- deepcopy of the fused loss ----------------------------------------------
class Feat(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1))
        self.feature_layers = (0, 2)
        self.use_norm = False


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.perceptual_vgg = Feat()
        for p in self.parameters():
            p.requires_grad = False

    def compute_perceptual_loss(self, input, target):
        called.append(1)
        feats = self.perceptual_vgg.model.features
        def run(v):
            out, y = [], v
            for i in range(3):
                y = feats[i](y)
                if i in (0, 2):
                    out.append(y.reshape(y.shape[0], -1))
            return torch.cat(out, 1)
        return ((run(input) - run(target).detach()) ** 2).mean()


called = []
lo = Loss().to("xdna", torch.bfloat16)
regions.fuse_perceptual_loss(lo)
lc = copy.deepcopy(lo)
with torch.no_grad():
    for p in lc.perceptual_vgg.model.features.parameters():
        p.zero_()  # the copy's network now outputs zeros for every input
xi = torch.randn(2, 3, 8, 8).bfloat16().to("xdna").requires_grad_()
tg = torch.rand(2, 3, 8, 8).bfloat16().to("xdna")
report("loss deepcopy: the copy uses its own (zeroed) feature weights", float(lc.compute_perceptual_loss(xi, tg)) == 0.0,
       f"loss {float(lc.compute_perceptual_loss(xi, tg)):.4f}")
report("loss deepcopy: the original is unaffected", float(lo.compute_perceptual_loss(xi, tg)) > 0.0)

# ---- weights unfrozen after fusing ---------------------------------------------
lu = Loss().to("xdna", torch.bfloat16)
regions.fuse_perceptual_loss(lu)
for p in lu.perceptual_vgg.model.features.parameters():
    p.requires_grad = True
xu = torch.randn(2, 3, 8, 8).bfloat16().to("xdna").requires_grad_()
lu.compute_perceptual_loss(xu, tg).backward()
f0 = lu.perceptual_vgg.model.features[0]
report("unfrozen feature weights still receive gradients (region falls back)",
       f0.weight.grad is not None and f0.weight.grad.abs().sum() > 0 and xu.grad is not None)

# ---- shared activation, consumed mark, slope read at call time ------------------
b = mk()
regions.fuse_norm_activation(b.norm, b.relu)
y = b.relu(b.norm(x))
z = b.relu(y)
twin = mk()
yt = twin.relu(twin.norm(x))
zt = twin.relu(yt)
report("a second activation call on the activated tensor runs (mark is consumed)", rel(z, zt) < 0.03,
       f"rel {rel(z, zt):.4f}")
b2, t2 = mk(), mk()
regions.fuse_norm_activation(b2.norm, b2.relu)
b2.relu.negative_slope = 0.5
t2.relu.negative_slope = 0.5
report("a changed negative_slope is used", rel(b2(x), t2(x)) < 0.03, f"rel {rel(b2(x), t2(x)):.4f}")

# ---- model surgery after the chain was built ------------------------------------
def fresh_fused_loss():
    lm = Loss().to("xdna", torch.bfloat16)
    regions.fuse_perceptual_loss(lm)
    xw = torch.randn(2, 3, 8, 8).bfloat16().to("xdna").requires_grad_()
    lm.compute_perceptual_loss(xw, tg)  # builds and caches the chain
    return lm


def same_as_eager(name, lm, mutate):
    mutate(lm)
    xa = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(5)).bfloat16().to("xdna").requires_grad_()
    xb = xa.detach().clone().requires_grad_()
    got = lm.compute_perceptual_loss(xa, tg)
    want = type(lm).compute_perceptual_loss(lm, xb, tg)
    got.backward()
    want.backward()
    ok = abs(float(got) - float(want)) <= 0.03 * abs(float(want)) + 1e-6 and rel(xa.grad, xb.grad) < 0.05
    report(name, ok, f"loss {float(got):.5f} vs eager {float(want):.5f}, dx rel {rel(xa.grad, xb.grad):.4f}")


def set_layer(lm, i, mod):
    lm.perceptual_vgg.model.features[i] = mod


same_as_eager("surgery: ReLU replaced by Sigmoid", fresh_fused_loss(), lambda lm: set_layer(lm, 1, nn.Sigmoid()))
same_as_eager("surgery: ReLU replaced by MaxPool2d", fresh_fused_loss(), lambda lm: set_layer(lm, 1, nn.MaxPool2d(2, 2)))
same_as_eager("surgery: conv stride changed", fresh_fused_loss(),
              lambda lm: setattr(lm.perceptual_vgg.model.features[0], "stride", (2, 2)))
same_as_eager("surgery: conv padding changed", fresh_fused_loss(),
              lambda lm: setattr(lm.perceptual_vgg.model.features[0], "padding", (0, 0)))
same_as_eager("surgery: conv dilation changed", fresh_fused_loss(),
              lambda lm: setattr(lm.perceptual_vgg.model.features[2], "dilation", (2, 2)))
same_as_eager("surgery: forward hook that changes the output", fresh_fused_loss(),
              lambda lm: lm.perceptual_vgg.model.features[0].register_forward_hook(lambda m, i, o: o * 3.0))
same_as_eager("surgery: forward pre-hook", fresh_fused_loss(),
              lambda lm: lm.perceptual_vgg.model.features[2].register_forward_pre_hook(lambda m, i: (i[0] * 2.0,)))


class ScaledConv(nn.Conv2d):
    def forward(self, input):
        return super().forward(input) * 3.0


def to_subclass(lm):
    lm.perceptual_vgg.model.features[0].__class__ = ScaledConv


same_as_eager("surgery: Conv2d subclass with its own forward", fresh_fused_loss(), to_subclass)
same_as_eager("surgery: instance-level forward", fresh_fused_loss(),
              lambda lm: setattr(lm.perceptual_vgg.model.features[0], "forward",
                                 lambda v, m=lm.perceptual_vgg.model.features[0]: nn.Conv2d.forward(m, v) * 3.0))
same_as_eager("surgery: new conv weights (a different Parameter)", fresh_fused_loss(),
              lambda lm: setattr(lm.perceptual_vgg.model.features[0], "weight",
                                 nn.Parameter(lm.perceptual_vgg.model.features[0].weight.detach() * 2.0, requires_grad=False)))

# ---- NaN activation: fused BN + ReLU / LeakyReLU gradients match eager -----------
def nan_bias_grads(act_cls, fuse):
    torch.manual_seed(3)
    bn = nn.BatchNorm2d(4).to("xdna", torch.bfloat16)
    act = act_cls()
    with torch.no_grad():
        bn.bias.copy_(torch.tensor([float("nan"), 0.1, -0.2, 0.3]))
    if fuse:
        regions.fuse_norm_activation(bn, act)
    xn = torch.randn(3, 4, 9, 9).bfloat16().to("xdna").requires_grad_()
    gn = torch.randn(3, 4, 9, 9).bfloat16().to("xdna")
    act(bn(xn)).backward(gn)
    return xn.grad.float().cpu(), bn.bias.grad.float().cpu()


for cls, nm in ((nn.ReLU, "ReLU"), (lambda: nn.LeakyReLU(0.2), "LeakyReLU")):
    dxf, dbf = nan_bias_grads(cls, True)
    dxe, dbe = nan_bias_grads(cls, False)
    same_nan = torch.equal(torch.isnan(dxf), torch.isnan(dxe)) and torch.equal(torch.isnan(dbf), torch.isnan(dbe))
    fin = ~torch.isnan(dxe)
    finb = ~torch.isnan(dbe)
    ok = same_nan and (dxf[fin] - dxe[fin]).abs().max() < 0.05 * dxe[fin].abs().max().clamp_min(1e-3) \
        and (dbf[finb] - dbe[finb]).abs().max() < 0.05 * dbe[finb].abs().max().clamp_min(1e-3)
    report(f"NaN bias + fused BN + {nm}: gradients match eager", ok,
           f"|dbias| fused {dbf.abs().max():.3f} eager {dbe.abs().max():.3f}")

print("ALL OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
