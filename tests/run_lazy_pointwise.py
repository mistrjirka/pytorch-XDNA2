#!/usr/bin/env python3
"""Lazy pointwise values (XDNA_LAZY_POINTWISE): a lazy LeakyReLU must read like
the eager one through every access path, and the conv3x3w pack must apply it
without writing it.  ReLU, in-place ReLU on a BatchNorm output, clone and the
residual add keep or consume lazy values with the eager results, and so do
channel cat and nearest 2x upsample, which the conv pack reads from their
inputs.

Run with the hardware Python: PYTHONPATH=<repo> python tests/run_lazy_pointwise.py
"""
import os
import sys

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")
os.environ.setdefault("XDNA_NPU_MODE", "2")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")

import torch
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()
failures = []
SLOPE = 0.2


def check(name, ok, detail=""):
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        failures.append(name)


def stats():
    s = xdna_train.allocator_stats()
    return s["lazy_created"], s["lazy_consumed"], s["lazy_materialized"]


torch.manual_seed(0)
x_cpu = torch.randn(4, 64, 8, 8).bfloat16()
x_cpu[0, 0, 0, :3] = torch.tensor([float("nan"), -0.0, float("-inf")])
ref = F.leaky_relu(x_cpu, SLOPE)


def lazy():
    return F.leaky_relu(x_cpu.to("xdna"), SLOPE)


def eq(a, b):
    return torch.equal(torch.nan_to_num(a, 7.0), torch.nan_to_num(b, 7.0))


c0 = stats()
y = lazy()
check("leaky_relu is lazy", stats()[0] == c0[0] + 1 and stats()[2] == c0[2])
check(".cpu() equals eager", eq(y.cpu(), ref))

# Reads through other access paths and aliases.
cases = {
    "clone": lambda t: t.clone(),
    "permute": lambda t: t.permute(0, 2, 3, 1),
    "narrow": lambda t: t.narrow(1, 8, 16),
    "storage alias": lambda t: t.as_strided((t.numel(),), (1,)),
    "reshape": lambda t: t.reshape(4, -1),
    "add": lambda t: t + 1,
    "sum": lambda t: t.float().sum(dim=(0, 2, 3)),
    "str": lambda t: torch.tensor(len(str(t).replace(", device='xdna:0'", ""))),
}
for name, fn in cases.items():
    got = fn(lazy())
    want = fn(ref)
    check(f"read via {name}", eq(got.cpu() if got.device.type == "xdna" else got, want))

# A write into the lazy tensor keeps the rest of its value.
t = lazy()
t[0, 0].zero_()
want = ref.clone()
want[0, 0].zero_()
check("partial in-place write", eq(t.cpu(), want))

# Growing the storage through an alias.
t = lazy()
v = t.as_strided((t.numel(),), (1,))
v.resize_(t.numel() + 16)
check("alias resize", eq(t.cpu(), ref))

# set_ onto the lazy storage.
t = lazy()
u = torch.empty(0, device="xdna", dtype=torch.bfloat16)
u.set_(t.untyped_storage(), 0, t.shape, t.stride())
check("set_ to lazy storage", eq(u.cpu(), ref))

# Writing the source after the lazy value exists: the value must keep the
# old input (materialized before the write) or raise, never read new data.
src = x_cpu.to("xdna")
t = F.leaky_relu(src, SLOPE)
try:
    src.mul_(0)
    check("source written afterwards", eq(t.cpu(), ref))
except RuntimeError as e:
    check("source written afterwards (raised)", "lazy" in str(e), str(e)[:80])

# Dropped unread: no materialization.
c0 = stats()
t = lazy()
del t
check("dropped unread is never written", stats()[2] == c0[2])

# conv3x3w consumes the lazy value; a second reader (add) still sees it.
# (8x8 images are below the conv3x3w family's size, so use 16x16.)
torch.manual_seed(1)
xc_cpu = torch.randn(8, 64, 16, 16).bfloat16()
cref = F.leaky_relu(xc_cpu, SLOPE)
w_cpu = (torch.randn(128, 64, 3, 3) * 0.05).bfloat16()
conv_ref = F.conv2d(cref.float(), w_cpu.float(), padding=1)
c0 = stats()
t = F.leaky_relu(xc_cpu.to("xdna"), SLOPE)
out = F.conv2d(t, w_cpu.to("xdna"), padding=1)
c1 = stats()
check("conv consumed without materializing", c1[1] == c0[1] + 1 and c1[2] == c0[2],
      f"consumed {c1[1]-c0[1]} materialized {c1[2]-c0[2]}")
err = (out.cpu().float() - torch.nan_to_num(conv_ref, 0)).abs()
finite = torch.isfinite(conv_ref)
check("conv on lazy input matches eager",
      bool(((err <= 0.05 + 0.02 * conv_ref.abs()) | ~finite).all()),
      f"max err {err[finite].max().item():.4f}")
check("second reader after conv", eq((t * 2).cpu(), cref * 2))

# ReLU is clamp_min(x, 0): -inf -> 0 and NaN stays (LeakyReLU slope 0 differs).
c0 = stats()
r = F.relu(x_cpu.to("xdna"))
check("relu is lazy", stats()[0] == c0[0] + 1)
check("relu equals eager", eq(r.cpu(), F.relu(x_cpu)),
      f"{r.cpu()[0, 0, 0, :3].tolist()} vs {F.relu(x_cpu)[0, 0, 0, :3].tolist()}")

# clone of a lazy value stays lazy; add reads lazy values without writing them.
c0 = stats()
a = lazy()
b = a.clone()
c1 = stats()
check("clone of a lazy value is lazy", c1[0] == c0[0] + 2 and c1[2] == c0[2])
p_cpu = torch.randn(4, 64, 8, 8).bfloat16()
p = p_cpu.to("xdna")
cases = {
    "lazy + lazy": (lambda: lazy() + F.relu(p), lambda: ref + F.relu(p_cpu)),
    "lazy + plain": (lambda: lazy() + p, lambda: ref + p_cpu),
    "plain + lazy": (lambda: p + lazy(), lambda: p_cpu + ref),
    "lazy += plain": (lambda: lazy().add_(p), lambda: ref + p_cpu),
    "plain += lazy": (lambda: p.clone().add_(lazy()), lambda: p_cpu + ref),
    "x += relu(x)": (lambda: (lambda t: t.add_(F.relu(t)))(p.clone()),
                     lambda: p_cpu + F.relu(p_cpu)),
    "relu(x) + x": (lambda: F.relu(p) + p, lambda: F.relu(p_cpu) + p_cpu),
}
for name, (got_fn, want_fn) in cases.items():
    c0 = stats()
    got = got_fn()
    c1 = stats()
    check(f"add {name}", eq(got.cpu(), want_fn()) and c1[1] > c0[1],
          f"consumed {c1[1]-c0[1]} materialized {c1[2]-c0[2]}")

# relu_ on a lazy BatchNorm output joins the record; its backward
# (threshold_backward on the result) reads the BatchNorm input.
torch.manual_seed(3)
bx_cpu = torch.randn(4, 64, 8, 8).bfloat16()
bw_cpu = torch.rand(64).bfloat16() + 0.5
bb_cpu = torch.randn(64).bfloat16()
g_cpu = torch.randn(4, 64, 8, 8).bfloat16()


def bn_relu(dev):
    x = bx_cpu.to(dev).requires_grad_()
    y = F.batch_norm(x, None, None, bw_cpu.to(dev), bb_cpu.to(dev), training=True)
    if dev == "xdna":
        bn_relu.before = stats()
    y = F.relu(y, inplace=True)
    if dev == "xdna":
        bn_relu.after = stats()
    out = y.cpu()
    y.backward(g_cpu.to(dev))
    return out, x.grad.cpu()


(yl, gl), (yc, gc) = bn_relu("xdna"), bn_relu("cpu")
check("relu_ on lazy BN output stays lazy",
      bn_relu.after[0] == bn_relu.before[0] and bn_relu.after[2] == bn_relu.before[2])
check("BN -> relu_ value vs CPU", eq(yl, yc) or (yl.float() - yc.float()).abs().max() <= 0.0625,
      f"max diff {(yl.float() - yc.float()).abs().max().item()}")
check("BN -> relu_ grad vs CPU", ((gl.float() - gc.float()).norm() / gc.float().norm()) < 0.05)

# threshold_backward on a lazy ReLU value: an infinite gradient where the ReLU
# is off gives 0 (a select, not a multiply by 0).
r = F.relu(x_cpu.to("xdna"))
gi_cpu = torch.randn_like(x_cpu)
gi_cpu[0, 0, 0, :4] = float("inf")
c0 = stats()
tb = torch.ops.aten.threshold_backward(gi_cpu.to("xdna"), r, 0).cpu()
check("threshold_backward reads the lazy value", stats()[1] == c0[1] + 1 and stats()[2] == c0[2])
check("threshold_backward equals eager",
      eq(tb, torch.ops.aten.threshold_backward(gi_cpu, F.relu(x_cpu), 0)))

# cat along channels and nearest 2x upsample: lazy over lazy and plain inputs,
# channel counts not multiples of the 64-channel pack chunk.
torch.manual_seed(4)
pa_cpu = torch.randn(4, 3, 8, 8).bfloat16()
pb_cpu = torch.randn(4, 61, 8, 8).bfloat16()
pa, pb = pa_cpu.to("xdna"), pb_cpu.to("xdna")
cref = torch.cat([ref, pa_cpu, F.relu(pb_cpu)], 1)
c0 = stats()
cv = torch.cat([lazy(), pa, F.relu(pb)], 1)
check("cat is lazy", stats()[0] == c0[0] + 3 and stats()[2] == c0[2])
check("cat equals eager", eq(cv.cpu(), cref))
check("cat read via narrow", eq(torch.cat([lazy(), pa, F.relu(pb)], 1).narrow(1, 60, 10).cpu(),
                               cref.narrow(1, 60, 10)))
uref = F.interpolate(cref, scale_factor=2)
c0 = stats()
uv = F.interpolate(torch.cat([lazy(), pa, F.relu(pb)], 1), scale_factor=2)
check("upsample of cat is lazy", stats()[2] == c0[2])
check("upsample of cat equals eager", eq(uv.cpu(), uref))
check("upsample of plain equals eager", eq(F.interpolate(pa, scale_factor=2).cpu(),
                                           F.interpolate(pa_cpu, scale_factor=2)))
check("leaky of upsample equals eager",
      eq(F.leaky_relu(F.interpolate(pa, scale_factor=2), SLOPE).cpu(),
         F.leaky_relu(F.interpolate(pa_cpu, scale_factor=2), SLOPE)))
check("cat along dim 2 equals eager", eq(torch.cat([lazy(), lazy()], 2).cpu(), torch.cat([ref, ref], 2)))
# A plain input of a lazy cat written afterwards: the cat keeps the old input.
q = pa_cpu.to("xdna")
cq = torch.cat([q, pb], 1)
q.mul_(0)
check("cat input written afterwards", eq(cq.cpu(), torch.cat([pa_cpu, pb_cpu], 1)))

# The conv packs upsample(cat(...)) from the inputs without writing it.
torch.manual_seed(5)
ca_cpu = torch.randn(8, 64, 8, 8).bfloat16()
cb_cpu = torch.randn(8, 3, 8, 8).bfloat16()
wc_cpu = (torch.randn(128, 67, 3, 3) * 0.05).bfloat16()
c0 = stats()
inp = F.interpolate(torch.cat([F.leaky_relu(ca_cpu.to("xdna"), SLOPE), cb_cpu.to("xdna")], 1),
                    scale_factor=2)
out = F.conv2d(inp, wc_cpu.to("xdna"), padding=1)
c1 = stats()
check("conv consumes upsample(cat) without materializing", c1[2] == c0[2] and c1[1] > c0[1],
      f"consumed {c1[1]-c0[1]} materialized {c1[2]-c0[2]}")
cin_ref = F.interpolate(torch.cat([F.leaky_relu(ca_cpu, SLOPE), cb_cpu], 1), scale_factor=2)
conv_ref = F.conv2d(cin_ref.float(), wc_cpu.float(), padding=1)
err = (out.cpu().float() - conv_ref).abs()
check("conv on upsample(cat) matches", bool((err <= 0.05 + 0.02 * conv_ref.abs()).all()),
      f"max err {err.max().item():.4f}")
check("upsample(cat) still reads after conv", eq(inp.cpu(), cin_ref))

# BatchNorm right after an NPU conv takes the statistics the conv's unpack
# summed (no read pass); the result equals BatchNorm on a copy, which runs the
# statistics kernel.
torch.manual_seed(6)
sx_cpu = torch.randn(8, 64, 16, 16).bfloat16()
sw_cpu = (torch.randn(128, 64, 3, 3) * 0.05).bfloat16()
co = F.conv2d(sx_cpu.to("xdna"), sw_cpu.to("xdna"), padding=1)
bw, bb = (torch.rand(128) + 0.5).bfloat16().to("xdna"), torch.randn(128).bfloat16().to("xdna")
r0 = xdna_train.allocator_stats()["bn_stats_reused"]
fused = torch.ops.aten.native_batch_norm(co, bw, bb, None, None, True, 0.1, 1e-5)
r1 = xdna_train.allocator_stats()["bn_stats_reused"]
plain = torch.ops.aten.native_batch_norm(co.clone().add_(0), bw, bb, None, None, True, 0.1, 1e-5)
check("BN after conv reuses the unpack statistics", r1 == r0 + 1)
check("BN with unpack statistics equals the kernel's",
      all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(fused, plain)))
co.add_(1)
r0 = xdna_train.allocator_stats()["bn_stats_reused"]
torch.ops.aten.native_batch_norm(co, bw, bb, None, None, True, 0.1, 1e-5)
check("written conv output does not reuse stale statistics",
      xdna_train.allocator_stats()["bn_stats_reused"] == r0)

# Stage 4: a conv output stays in the NPU's rows; the next conv packs from them.
# Packing from rows must give the same packed bytes, so the same result, as
# packing from the written tensor (forced here by an in-place add of 0).
def rows_chain(x_cpu, w1_cpu, w2_cpu, between, written):
    y = F.conv2d(x_cpu.to("xdna"), w1_cpu.to("xdna"), padding=1)
    if written:
        y.add_(0)
    return F.conv2d(between(y), w2_cpu.to("xdna"), padding=1).cpu()


torch.manual_seed(7)
bn_w = (torch.rand(192) + 0.5).bfloat16().to("xdna")
bn_b = torch.randn(192).bfloat16().to("xdna")
plain3 = torch.randn(8, 3, 40, 40).bfloat16().to("xdna")
rows_cases = {
    "conv -> conv 16x16": ((8, 64, 16, 16), 128, 128, lambda y: y),
    "conv -> conv strips W=40": ((8, 64, 40, 40), 128, 128, lambda y: y),
    "conv(192) -> leaky -> cat -> conv": (
        (8, 64, 40, 40), 192, 195, lambda y: torch.cat([F.leaky_relu(y, SLOPE), plain3], 1)),
    "conv -> BN -> leaky -> upsample -> conv": (
        (8, 64, 20, 20), 192, 192,
        lambda y: F.interpolate(F.leaky_relu(F.batch_norm(y, None, None, bn_w, bn_b, True),
                                             SLOPE), scale_factor=2)),
}
for name, (xs, c1, c2, between) in rows_cases.items():
    xr = torch.randn(*xs).bfloat16()
    w1r = (torch.randn(c1, xs[1], 3, 3) * 0.05).bfloat16()
    w2r = (torch.randn(128, c2, 3, 3) * 0.05).bfloat16()
    c0 = stats()
    got = rows_chain(xr, w1r, w2r, between, False)
    c1s = stats()
    want = rows_chain(xr, w1r, w2r, between, True)
    # One write: reading the final result with .cpu().
    check(f"rows: {name} equals packing written bytes", torch.equal(got, want))
    check(f"rows: {name} writes only the result", c1s[2] - c0[2] == 1,
          f"materialized {c1s[2]-c0[2]}")
y = F.conv2d(xc_cpu.to("xdna"), w_cpu.to("xdna"), padding=1)
check("rows: conv output reads like eager",
      bool(((y.cpu().float() - F.conv2d(xc_cpu.float(), w_cpu.float(), padding=1)).abs()
            <= 0.05 + 0.02 * F.conv2d(xc_cpu.float(), w_cpu.float(), padding=1).abs()).all()))

# Stage 4 backward: activation and BatchNorm backward on tensors kept in rows
# (a conv output as input, another conv output as gradient) give rows.  Each
# is compared with the eager op on the same values: reading the operands with
# .cpu() writes them, after which the op runs eagerly.
def rows_pair(c, hw, seed):
    torch.manual_seed(seed)
    x = torch.randn(8, 64, hw, hw).bfloat16().to("xdna")
    w1 = (torch.randn(c, 64, 3, 3) * 0.05).bfloat16().to("xdna")
    w2 = (torch.randn(c, 64, 3, 3) * 0.05).bfloat16().to("xdna")
    return F.conv2d(x, w1, padding=1), F.conv2d(x, w2, padding=1)


def phases():
    return {k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()}


for c, hw in ((128, 16), (192, 40)):
    t, g = rows_pair(c, hw, 8)
    p0 = phases()
    r_leaky = torch.ops.aten.leaky_relu_backward(g, t, SLOPE, False).cpu()
    bw_ = (torch.rand(c) + 0.5).bfloat16().to("xdna")
    bb_ = torch.randn(c).bfloat16().to("xdna")
    _, sm, si = torch.ops.aten.native_batch_norm(t, bw_, bb_, None, None, True, 0.1, 1e-5)
    r_bn = [v.cpu() for v in torch.ops.aten.native_batch_norm_backward(
        g, t, bw_, None, None, sm, si, True, 1e-5, [True, True, True])]
    used = phases().get("aten::act_backward.rows", 0) - p0.get("aten::act_backward.rows", 0)
    used_bn = (phases().get("aten::native_batch_norm_backward.rows", 0)
               - p0.get("aten::native_batch_norm_backward.rows", 0))
    g.cpu(), t.cpu()  # written: the ops below run eagerly
    e_leaky = torch.ops.aten.leaky_relu_backward(g, t, SLOPE, False).cpu()
    e_bn = [v.cpu() for v in torch.ops.aten.native_batch_norm_backward(
        g, t, bw_, None, None, sm, si, True, 1e-5, [True, True, True])]
    tag = f"rows backward C={c} {hw}x{hw}"
    check(f"{tag}: rows paths used", used == 1 and used_bn == 1, f"act {used} bn {used_bn}")
    check(f"{tag}: leaky backward equals eager", eq(r_leaky, e_leaky))
    # Double sums against ATen's float sums: at most one bf16 step apart, rarely.
    for name, a, b in zip(("dx", "dgamma", "dbeta"), r_bn, e_bn):
        a, b = a.float(), b.float()
        step = b.abs().clamp_min(1e-30) * 2.0 ** -7
        off = ((a - b).abs() > step).float().mean().item()
        rel = ((a - b).norm() / b.norm()).item()
        check(f"{tag}: BN backward {name} vs eager", off < 1e-4 and rel < 1e-3,
              f"over one step {off:.2e} rel {rel:.2e}")

# The decoder's backward: upsample backward of a gradient in rows, then the
# cat backward's channel slices, each fed to an activation backward.
torch.manual_seed(9)
xs_ = torch.randn(8, 64, 40, 40).bfloat16().to("xdna")
g_big = F.conv2d(xs_, (torch.randn(192, 64, 3, 3) * 0.05).bfloat16().to("xdna"), padding=1)
t_small = F.conv2d(xs_[:, :, :20, :20].contiguous(),
                   (torch.randn(128, 64, 3, 3) * 0.05).bfloat16().to("xdna"), padding=1)
p0 = phases()
g_small = torch.ops.aten.upsample_nearest2d_backward(g_big, [40, 40], [8, 192, 20, 20], None, None)
r_slices = [torch.ops.aten.leaky_relu_backward(g_small.narrow(1, c0, n), t_small.narrow(1, 0, n),
                                               SLOPE, False).cpu()
            for c0, n in ((0, 128), (128, 64))]
r_up = g_small.cpu()
p1 = phases()
used_up = (p1.get("aten::upsample_nearest2d_backward.rows", 0)
           - p0.get("aten::upsample_nearest2d_backward.rows", 0))
used_act = p1.get("aten::act_backward.rows", 0) - p0.get("aten::act_backward.rows", 0)
g_big.cpu(), t_small.cpu()
e_up = torch.ops.aten.upsample_nearest2d_backward(g_big, [40, 40], [8, 192, 20, 20], None, None).cpu()
check("rows upsample backward used", used_up == 1 and used_act >= 1, f"up {used_up} act {used_act}")
check("rows upsample backward equals eager", eq(r_up, e_up))
check("rows leaky backward on cat slices equals eager",
      all(eq(r, torch.ops.aten.leaky_relu_backward(
                 e_up.narrow(1, c0, n), t_small.cpu().narrow(1, 0, n), SLOPE, False))
          for r, (c0, n) in zip(r_slices, ((0, 128), (128, 64)))))

# Training, without and with BatchNorm (conv -> [BN ->] leaky -> conv).  A child
# process with XDNA_LAZY_POINTWISE=0 gives the eager XDNA results, which lazy
# must match exactly, except the running variance (one bf16 step, see the TODO
# in lazy_batch_norm_training).  The CPU comparison guards gross errors only
# (the NPU uses bfp16 weights).
# "res" is a ResNet block: conv -> BN -> relu_ -> clone (skip) -> conv -> BN,
# + skip -> relu -> conv.  "dec" is lab04's decoder: conv -> BN -> leaky,
# cat with the block input, upsample 2x -> conv -> BN -> leaky -> conv.
def net(dev, mode):
    torch.manual_seed(2)
    x = torch.randn(8, 64, 16, 16).bfloat16().to(dev).requires_grad_()
    w1 = (torch.randn(128, 64, 3, 3) * 0.05).bfloat16().to(dev).requires_grad_()
    w2 = (torch.randn(128, 192 if mode == "dec" else 128, 3, 3) * 0.05).bfloat16().to(dev).requires_grad_()
    w3 = (torch.randn(128, 128, 3, 3) * 0.05).bfloat16().to(dev).requires_grad_()
    norms = [torch.nn.BatchNorm2d(128).to(dev, torch.bfloat16) for _ in range(2)]
    for norm in norms:
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    h = F.conv2d(x, w1, padding=1)
    if mode == "res":
        h = F.relu(norms[0](h), inplace=True)
        skip = h.clone()
        h = norms[1](F.conv2d(h, w2, padding=1)) + skip
        out = F.conv2d(F.relu(h), w3, padding=1)
    elif mode == "dec":
        h = F.leaky_relu(norms[0](h), SLOPE)
        h = F.interpolate(torch.cat([h, x], 1), scale_factor=2)
        h = F.leaky_relu(norms[1](F.conv2d(h, w2, padding=1)), SLOPE)
        out = F.conv2d(h, w3, padding=1)
    else:
        if mode == "bn":
            h = norms[0](h)
        out = F.conv2d(F.leaky_relu(h, SLOPE), w2, padding=1)
    if dev == "xdna":
        net.forward_stats = stats()
    out.float().square().mean().backward()
    res = [t.grad.cpu().float() for t in (x, w1, w2) + ((w3,) if mode in ("res", "dec") else ())]
    for norm in norms[: {"plain": 0, "bn": 1, "res": 2, "dec": 2}[mode]]:
        res += [norm.weight.grad.cpu().float(), norm.bias.grad.cpu().float(),
                norm.running_mean.cpu().float(), norm.running_var.cpu().float()]
    return res


MODES = ("plain", "bn", "res", "dec")
if os.environ.get("_LAZY_CHILD"):
    torch.save({m: net("xdna", m) for m in MODES}, os.environ["_LAZY_CHILD"])
    sys.exit(0)
import subprocess, tempfile
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "eager.pt")
    env = dict(os.environ, XDNA_LAZY_POINTWISE="0", _LAZY_CHILD=path)
    subprocess.run([sys.executable, __file__], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    eager = torch.load(path)
for tag in MODES:
    names = ["dx", "dw1", "dw2"] + (["dw3"] if tag in ("res", "dec") else [])
    for i in range({"plain": 0, "bn": 1, "res": 2, "dec": 2}[tag]):
        names += [f"{n}{i}" for n in ("dgamma", "dbeta", "running_mean", "running_var")]
    c0 = stats()
    lazy_r = net("xdna", tag)
    c1 = stats()
    cpu_r = net("cpu", tag)
    # Backward may materialize (a CPU weight gradient needs real input bytes:
    # the write eager did in forward), forward must not, except the res
    # block's add, which unpacks the two conv outputs it sums (as eager does).
    f = net.forward_stats
    allowed = 2 if tag == "res" else 0
    check(f"{tag}: forward consumes lazily, writes only what eager writes",
          f[2] - c0[2] <= allowed and f[1] - c0[1] >= 1,
          f"created {f[0]-c0[0]} consumed {f[1]-c0[1]} materialized {f[2]-c0[2]}; "
          f"after backward materialized {c1[2]-c0[2]}")
    for name, a, b, e in zip(names, cpu_r, lazy_r, eager[tag], strict=True):
        rel_e = ((b - e).norm() / e.norm().clamp_min(1e-12)).item()
        rel_c = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
        if name.startswith("running_var"):  # see the TODO in lazy_batch_norm_training
            check(f"{tag}: {name} vs eager XDNA", rel_e < 0.01, f"rel {rel_e:.5f}")
        else:
            check(f"{tag}: {name} equals eager XDNA", torch.equal(b, e), f"rel {rel_e:.5f}")
        # The deeper res block accumulates more bfp16 weight rounding.
        check(f"{tag}: {name} vs CPU", rel_c < (0.2 if tag in ("res", "dec") else 0.1), f"rel {rel_c:.4f}")

print(f"stats {xdna_train.allocator_stats()}")
print("ALL OK" if not failures else f"FAILED {failures}")
sys.exit(bool(failures))
