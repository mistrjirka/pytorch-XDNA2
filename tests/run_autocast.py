#!/usr/bin/env python3
"""torch.autocast("xdna") correctness (needs the XDNA2 machine; ~1 minute).

Reference is CPU autocast with the SAME bf16 policy on identical FP32 weights,
not an all-BF16 CPU model, so the check isolates the backend's autocast
registration, casts and gradient dtypes from rounding of the policy itself.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import copy
import os
import sys
import warnings

os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")  # counts NPU/CPU runs

import torch
import torch.nn.functional as F
from torch import nn

import xdna_train

xdna_train.register_xdna_device()
xdna_train.set_npu_mode(2)
failures = []
BF16 = torch.bfloat16


def check(name, cond, detail=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {detail}")
    if not cond:
        failures.append(name)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def close(a, b, tol=0.05):
    return rel(a, b) < tol


def ac(device, enabled=True):
    return torch.autocast(device, dtype=BF16, enabled=enabled)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # No bias before BatchNorm: its true gradient is ~0 (BN cancels it), so a
        # relative comparison of that gradient only measures rounding noise.
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.adaptive_avg_pool2d(x, 6).flatten(1)
        return self.fc2(F.gelu(self.fc1(x)))


def step(model, opt, x, y, dev):
    opt.zero_grad()
    with ac(dev):
        out = model(x)
        loss = F.cross_entropy(out, y)
    loss.backward()
    opt.step()
    return out, loss


torch.manual_seed(7)
base = Net()
x = torch.randn(8, 3, 6, 6)
y = torch.randint(0, 5, (8,))

# 1. dtypes of outputs, fp32 master weights and gradients ----------------------
m = copy.deepcopy(base).to("xdna")
xd, yd = x.to("xdna"), y.to("xdna")
with ac("xdna"):
    out = m(xd)
    loss = F.cross_entropy(out, yd)
check("matmul output is bf16", out.dtype == BF16, str(out.dtype))
check("loss is fp32 (policy)", loss.dtype == torch.float32, str(loss.dtype))
loss.backward()
check("params stay fp32", all(p.dtype == torch.float32 for p in m.parameters()))
check("grads are fp32", all(p.grad is not None and p.grad.dtype == torch.float32 for p in m.parameters()))

# 2. forward + grads equal CPU autocast on the same weights --------------------
c = copy.deepcopy(base)
with ac("cpu"):
    out_c = c(x)
    loss_c = F.cross_entropy(out_c, y)
loss_c.backward()
check("forward matches CPU autocast", close(out, out_c), f"rel={rel(out, out_c):.4f}")
worst = max(rel(pa.grad, pb.grad) for pa, pb in zip(m.parameters(), c.parameters()))
check("param grads match CPU autocast", worst < 0.08, f"worst rel={worst:.4f}")
check("conv weight grad is bf16-accurate", rel(m.conv.weight.grad, c.conv.weight.grad) < 0.02)

# 3. several optimizer steps keep fp32 state and track CPU --------------------
for name, mk in (("SGD", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
                 ("Adam", lambda p: torch.optim.Adam(p, lr=0.01))):
    md, mc = copy.deepcopy(base).to("xdna"), copy.deepcopy(base)
    od, oc = mk(md.parameters()), mk(mc.parameters())
    for _ in range(4):
        step(md, od, xd, yd, "xdna")
        step(mc, oc, x, y, "cpu")
    dtypes_ok = all(p.dtype == torch.float32 for p in md.parameters())
    state_ok = all(
        t.dtype == torch.float32
        for st in od.state.values() for t in st.values() if torch.is_tensor(t) and t.is_floating_point()
    )
    worst = max(rel(a, b) for a, b in zip(md.parameters(), mc.parameters()))
    check(f"{name}: fp32 params/state after steps", dtypes_ok and state_ok)
    check(f"{name}: params track CPU autocast", worst < 0.05, f"worst rel={worst:.4f}")

# 4. disabled / nested contexts ------------------------------------------------
lin = nn.Linear(8, 4).to("xdna")
a = torch.randn(3, 8).to("xdna")
with ac("xdna"):
    o1 = lin(a)
    with ac("xdna", enabled=False):
        o2 = lin(a)
    o3 = lin(a)
check("nested disabled context is fp32", o2.dtype == torch.float32, str(o2.dtype))
check("outer context restored", o1.dtype == BF16 and o3.dtype == BF16)
check("no autocast outside context", lin(a).dtype == torch.float32)

# 5. weight-cast cache invalidated by an optimizer update ----------------------
lin = nn.Linear(8, 4).to("xdna")
ref = copy.deepcopy(lin).cpu()
opt = torch.optim.SGD(lin.parameters(), lr=0.5)
a_c = a.cpu()
with ac("xdna"):
    before = lin(a)
with ac("xdna"):
    lin(a).float().sum().backward()
opt.step()
ro = torch.optim.SGD(ref.parameters(), lr=0.5)
with ac("cpu"):
    ref(a_c).float().sum().backward()
ro.step()
with ac("xdna"):
    after = lin(a)
with ac("cpu"):
    after_c = ref(a_c)
check("update visible in next autocast context", not torch.equal(before.cpu(), after.cpu()))
check("post-update output matches CPU", close(after, after_c), f"rel={rel(after, after_c):.4f}")

# 6. non-contiguous input, gradient accumulation --------------------------------
w = nn.Linear(8, 8).to("xdna")
wc = copy.deepcopy(w).cpu()
nc = torch.randn(8, 5).to("xdna").t()
check("input is non-contiguous", not nc.is_contiguous())
with ac("xdna"):
    o = w(nc)
with ac("cpu"):
    oc_ = wc(nc.cpu())
check("non-contiguous forward", close(o, oc_), f"rel={rel(o, oc_):.4f}")
for _ in range(2):
    with ac("xdna"):
        w(nc).float().sum().backward()
    with ac("cpu"):
        wc(nc.cpu()).float().sum().backward()
check("accumulated grads match", close(w.weight.grad, wc.weight.grad, 0.08),
      f"rel={rel(w.weight.grad, wc.weight.grad):.4f}")

# 7. routing under autocast matches a pure-BF16 model -------------------------
# Autocast adds casts, not a different dispatch: an op the NPU GEMM family takes
# in BF16 must not fall back to the CPU because its inputs arrive via autocast.
# 256x768 @ 768x3072 is such a shape (see run_gemm_safety.py).  nn.Linear
# multiplies by a transposed weight view, which still falls back in plain BF16
# too (general dX/dW orientations are roadmap item A1), so it is compared for
# parity rather than required to hit the NPU.
CPU_MM = "aten::mm.cpu_mapped"


def cpu_mm_calls(fn):
    xdna_train.reset_mapped_cpu_stats()
    fn().float().sum().backward()
    return xdna_train.mapped_cpu_stats().get(CPU_MM, {}).get("calls", 0)


a32 = (torch.randn(256, 768) * 0.03).to("xdna")
b32 = (torch.randn(768, 3072) * 0.03).to("xdna")


xdna_train.reset_mapped_cpu_stats()
with ac("xdna"):
    mm_out = a32 @ b32
mm_out.cpu()
fallbacks = xdna_train.mapped_cpu_stats().get(CPU_MM, {}).get("calls", 0)
check("autocast mm forward stays off CPU fallback", fallbacks == 0, f"fallback mm calls={fallbacks}")

lin_ac = nn.Linear(768, 3072, bias=False).to("xdna")
lin_bf = copy.deepcopy(lin_ac).bfloat16()
xl = (torch.randn(256, 768) * 0.1).to("xdna")


def lin_autocast():
    with ac("xdna"):
        return lin_ac(xl)


check("nn.Linear routing parity with BF16",
      cpu_mm_calls(lin_autocast) == cpu_mm_calls(lambda: lin_bf(xl.bfloat16())),
      "autocast and BF16 take the same mm path")
check("autocast gradients fp32", lin_ac.weight.grad.dtype == torch.float32)

# 8. default dtype and explicit float16 handling --------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("error")  # a silent-disable warning must not appear
    with torch.autocast("xdna"):
        od_ = lin(a)
check("bare autocast('xdna') uses bf16", od_.dtype == BF16, str(od_.dtype))

# 9. torch.cat promotes mixed dtypes like CPU (BF16 activations next to an FP32 input)
a_bf = torch.randn(2, 3, 4).bfloat16().to("xdna").requires_grad_()
b_fp = torch.randn(2, 5, 4).to("xdna").requires_grad_()
cat = torch.cat([a_bf, b_fp], dim=1)
check("cat of bf16 and fp32 promotes to fp32", cat.dtype == torch.float32 and cat.shape == (2, 8, 4), str(cat.dtype))
cat.sum().backward()
check("cat gradients return in each input's dtype",
      a_bf.grad.dtype == torch.bfloat16 and b_fp.grad.dtype == torch.float32)
cat_cpu = torch.cat([a_bf.detach().cpu(), b_fp.detach().cpu()], dim=1)
check("cat values match CPU", torch.equal(cat.detach().cpu(), cat_cpu))

# 10. NPU weight gradient + autocast: the FP32 cast of the BF16 dW must not force the
# deferred NPU job (that ended its overlap with backward); values stay right.
conv_x = (torch.randn(40, 256, 18, 18) * 0.1)
conv_ref = nn.Conv2d(256, 128, 3, padding=1, bias=False)
conv_dev = copy.deepcopy(conv_ref).to("xdna")
xdna_train.reset_mapped_cpu_stats()
for _ in range(4):  # >= 3 steps so the async policy has learned the step structure
    conv_dev.weight.grad = None
    with ac("xdna"):
        out_ = conv_dev(conv_x.to("xdna"))
    out_.float().square().mean().backward()
    conv_dev.weight.grad.cpu()  # the reader (optimizer step) pays the wait
dw_runs = xdna_train.mapped_cpu_stats().get("npu_conv3x3w.dw.run", {}).get("calls", 0)
conv_ref.weight.grad = None
with ac("cpu"):
    out_c = conv_ref(conv_x)
out_c.float().square().mean().backward()
check("conv dW ran on the NPU under autocast", dw_runs >= 4, f"dw runs={dw_runs}")
check("deferred FP32 weight gradient matches CPU autocast",
      close(conv_dev.weight.grad, conv_ref.weight.grad, 0.04) and conv_dev.weight.grad.dtype == torch.float32,
      f"rel={rel(conv_dev.weight.grad, conv_ref.weight.grad):.4f}")

print()
print("FAILED:" if failures else "ALL OK", failures if failures else "")
sys.exit(1 if failures else 0)
