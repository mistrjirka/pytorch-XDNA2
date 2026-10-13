#!/usr/bin/env python3
"""Quick local correctness check (~1 min on the XDNA2 machine).  Run it before
pushing backend changes:

    PYTHONPATH=<repo> python tests/quick_check.py

1. Training matches CPU: a small conv net (NPU forward, dX and dW, BatchNorm,
   upsample, skip concat) trained with default Adam on CPU bf16 and on XDNA
   from the same seed must reach the same loss.  This catches anything that
   silently corrupts training, e.g. NPU convs running on stale weights.
2. The fast scripts: view safety, GEMM safety, import side effects, LSTM/GRU islands,
   conv packing.
"""
import os
import subprocess
import sys
import time

# Exercise the NPU paths deterministically: build NPU streams before their
# first call (instead of running on the CPU while they build) and skip the
# step autotuner, which may pick CPU only for a net this small.
os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_NPU_MODE", "2")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")  # NPU call counters

import torch
import torch.nn as nn
import xdna_train

HERE = os.path.dirname(os.path.abspath(__file__))
xdna_train.register_xdna_device()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))
        self.inp = block(3, 64)
        self.down = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.mid = block(128, 128)
        self.up = nn.Upsample(scale_factor=2)
        self.dec = block(128 + 64, 128)       # NPU-sized: B16 x 192 -> 128 @ 36x36
        self.out = nn.Conv2d(128, 3, 1)

    def forward(self, x):
        a = self.inp(x)
        b = self.mid(self.down(a))
        return self.out(self.dec(torch.cat([self.up(b), a], 1)))


def train(dev, steps=40):
    torch.manual_seed(0)
    model = Net().bfloat16().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(16, 3, 36, 36, generator=gen).bfloat16()
    y = (x.flip(1) * 0.5 + 0.25).bfloat16()        # learnable target
    x, y = x.to(dev), y.to(dev)
    losses = []
    for _ in range(steps):
        loss = nn.functional.l1_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def training_matches_cpu():
    t0 = time.time()
    xdna_train.reset_mapped_cpu_stats()
    lx = train("xdna")
    npu_calls = {k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()
                 if k.startswith("npu_conv3x3w.") and k.endswith(".run")}
    lc = train("cpu")
    first, last_c, last_x = lc[0], sum(lc[-5:]) / 5, sum(lx[-5:]) / 5
    learned = last_c < 0.5 * first
    # Correct runs land within ~3 % of the CPU loss (bf16 rounding); NPU convs
    # on stale weights ended ~70 % higher.
    close = abs(last_x - last_c) <= 0.2 * last_c
    used_npu = sum(npu_calls.values()) > 0
    ok = learned and close and used_npu
    print(f"training vs CPU: loss {first:.3f} -> cpu {last_c:.4f}, xdna {last_x:.4f}; "
          f"npu calls {npu_calls} ({time.time() - t0:.0f} s) {'OK' if ok else 'FAIL'}")
    if not used_npu:
        print("  no NPU conv ran: check the XRT install / XDNA_CONV3X3W settings")
    return ok


def script(name):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, name)],
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"{name}: {last} ({time.time() - t0:.0f} s)")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
    return r.returncode == 0


ok = training_matches_cpu()
for name in ("run_view_safety.py", "run_gemm_safety.py", "run_tensor_semantics.py",
             "run_deferred_dw.py", "run_import_scope.py", "run_cpu_islands.py",
             "run_conv3x3w_pack.py", "run_coherency_lifetime.py",
             "run_set_storage_pending.py", "run_pending_writes.py",
             "run_cpu_island_higher_order.py",
             "run_set_storage_variants.py", "run_gru_dropout_grad.py",
             "run_runtime_hygiene.py", "run_dw_floor.py", "run_autotune_builds.py",
             "run_norm_backward.py", "run_loss_warnings.py"):
    ok &= script(name)
print("QUICK CHECK OK" if ok else "QUICK CHECK FAILED")
sys.exit(0 if ok else 1)
