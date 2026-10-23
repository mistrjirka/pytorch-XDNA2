#!/usr/bin/env python3
"""Stage 6 (XDNA_CONV_DIRECT_INPUT): a conv3x3w reads a host-written rows tensor
as its input directly, and the NPU dW reads X and dY from such rows, with no
pack.  The bytes it reads are the ones a pack would write, so one training step
must give bit-equal gradients with the feature off (a child process).

Covers a ResNet block chain (BatchNorm backward output -> dX conv), plus a cat
whose activation backward is not a rows op yet (two-segment value), so its
slices must keep using the packed path.

Run with the hardware Python: PYTHONPATH=<repo> python tests/run_conv_direct_input.py
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")
os.environ.setdefault("XDNA_CONV3X3W_DW_MIN_MACS", "0")
os.environ.setdefault("XDNA_NPU_MODE", "2")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import xdna_train

xdna_train.register_xdna_device()


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        return F.leaky_relu(self.norm2(self.conv2(y)) + x, 0.2)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(Block(128), Block(128))
        self.a = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.b = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.c = nn.Conv2d(192, 64, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.c(F.leaky_relu(torch.cat([self.a(h), self.b(h)], 1), 0.2))


def grads():
    torch.manual_seed(0)
    net = Net().to("xdna", torch.bfloat16)
    x = torch.randn(4, 64, 64, 64).bfloat16().to("xdna")
    for _ in range(2):  # the second step runs with every stream built
        for p in net.parameters():
            p.grad = None
        xdna_train.reset_mapped_cpu_stats()
        net(x).float().square().mean().backward()
        out = {n: p.grad.cpu() for n, p in net.named_parameters()}
    phases = {k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()}
    return out, phases


if os.environ.get("_DIRECT_CHILD"):
    torch.save(grads()[0], os.environ["_DIRECT_CHILD"])
    sys.exit(0)

failures = []


def check(name, ok, detail=""):
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        failures.append(name)


got, phases = grads()
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "eager.pt")
    env = dict(os.environ, XDNA_CONV_DIRECT_INPUT="0", _DIRECT_CHILD=path)
    subprocess.run([sys.executable, __file__], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ref = torch.load(path)
direct = phases.get("npu_conv3x3w.direct_flush", 0)
# dX convs reading directly: the 4 block convs (the stem's dX is not needed;
# the cat slices are packed).
check("dX convs read their input directly", direct == 4, f"direct runs {direct}")
check("NPU dW ran for every conv", phases.get("npu_conv3x3w.dw.run", 0) >= 8,
      str({k: v for k, v in phases.items() if "dw" in k}))
for name in ref:
    check(f"{name} grad equals packed path", torch.equal(got[name], ref[name]))
print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
