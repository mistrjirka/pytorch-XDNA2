#!/usr/bin/env python3
"""Lazy pointwise values (XDNA_LAZY_POINTWISE): a lazy LeakyReLU must read like
the eager one through every access path, and the conv3x3w pack must apply it
without writing it.

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

# Training, without and with BatchNorm (conv -> [BN ->] leaky -> conv).  A child
# process with XDNA_LAZY_POINTWISE=0 gives the eager XDNA results, which lazy
# must match exactly, except the running variance (one bf16 step, see the TODO
# in lazy_batch_norm_training).  The CPU comparison guards gross errors only
# (the NPU uses bfp16 weights).
def net(dev, bn):
    torch.manual_seed(2)
    x = torch.randn(8, 64, 16, 16).bfloat16().to(dev).requires_grad_()
    w1 = (torch.randn(128, 64, 3, 3) * 0.05).bfloat16().to(dev).requires_grad_()
    w2 = (torch.randn(128, 128, 3, 3) * 0.05).bfloat16().to(dev).requires_grad_()
    norm = torch.nn.BatchNorm2d(128).to(dev, torch.bfloat16)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.5, 0.5)
    h = F.conv2d(x, w1, padding=1)
    if bn:
        h = norm(h)
    out = F.conv2d(F.leaky_relu(h, SLOPE), w2, padding=1)
    if dev == "xdna":
        net.forward_stats = stats()
    out.float().square().mean().backward()
    res = [t.grad.cpu().float() for t in (x, w1, w2)]
    if bn:
        res += [norm.weight.grad.cpu().float(), norm.bias.grad.cpu().float(),
                norm.running_mean.cpu().float(), norm.running_var.cpu().float()]
    return res


if os.environ.get("_LAZY_CHILD"):
    torch.save([net("xdna", False), net("xdna", True)], os.environ["_LAZY_CHILD"])
    sys.exit(0)
import subprocess, tempfile
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "eager.pt")
    env = dict(os.environ, XDNA_LAZY_POINTWISE="0", _LAZY_CHILD=path)
    subprocess.run([sys.executable, __file__], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    eager = torch.load(path)
names = ("dx", "dw1", "dw2", "dgamma", "dbeta", "running_mean", "running_var")
for bn in (False, True):
    c0 = stats()
    lazy_r = net("xdna", bn)
    c1 = stats()
    cpu_r = net("cpu", bn)
    tag = "bn" if bn else "plain"
    # Backward may materialize (a CPU weight gradient needs real input bytes:
    # the write eager did in forward), forward must not.
    f = net.forward_stats
    check(f"{tag}: forward consumes lazily, materializes nothing",
          f[2] == c0[2] and f[1] - c0[1] >= 1,
          f"created {f[0]-c0[0]} consumed {f[1]-c0[1]} materialized {f[2]-c0[2]}; "
          f"after backward materialized {c1[2]-c0[2]}")
    for name, a, b, e in zip(names, cpu_r, lazy_r, eager[int(bn)]):
        rel_e = ((b - e).norm() / e.norm().clamp_min(1e-12)).item()
        rel_c = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
        if name == "running_var":  # see the TODO in lazy_batch_norm_training
            check(f"{tag}: {name} vs eager XDNA", rel_e < 0.01, f"rel {rel_e:.5f}")
        else:
            check(f"{tag}: {name} equals eager XDNA", torch.equal(b, e), f"rel {rel_e:.5f}")
        check(f"{tag}: {name} vs CPU", rel_c < 0.1, f"rel {rel_c:.4f}")

print(f"stats {xdna_train.allocator_stats()}")
print("ALL OK" if not failures else f"FAILED {failures}")
sys.exit(bool(failures))
