#!/usr/bin/env python3
"""Per-op profile of one training step, CPU vs XDNA, for finding backend overhead.

Runs a model from benchmarks/model_zoo.py (default: nis_mynet with the NIS VGG
perceptual loss) on fixed data, profiles a few steps with torch.profiler and
prints self-time per op for each device side by side.  Ops whose XDNA self
time exceeds CPU point at dispatch, allocation, copy or layout overhead.

    PYTHONPATH=/path/to/torch-xdna2 python benchmarks/profile_step.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _nis_full():
    """NIS MyNet(3,3,5) + the script's ImageGenLoss (VGG19 perceptual + L1)."""
    sys.path.insert(0, "/home/jirka/programovani/magistr/nis/lab02_cnn")
    from models import MyNet
    from train import ImageGenLoss

    def build(device):
        model = MyNet(3, 3, 5)
        loss = ImageGenLoss(4.0, 6.0, device=device)
        return model, loss
    return build, lambda g: (torch.randn(40, 3, 36, 36, generator=g),
                             torch.rand(40, 3, 36, 36, generator=g))


def worker(model_name: str, device: str, steps: int) -> dict:
    from torch.profiler import ProfilerActivity, profile
    if device == "xdna":
        import xdna_train
        xdna_train.register_xdna_device()
    dt = torch.bfloat16
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    if model_name == "nis_full":
        build, make_batch = _nis_full()
        model, loss_mod = build(device)
        model = model.to(device, dt)
        loss_mod = loss_mod.to(device, dt)
        loss_fn = loss_mod
    else:
        from model_zoo import MODELS
        model, make_batch, loss_fn = MODELS[model_name]()
        model = model.to(device, dt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x, y = (t.to(device).to(dt) if t.is_floating_point() else t.to(device) for t in make_batch(gen))

    def step():
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()

    for _ in range(4):
        step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(steps):
            step()
    out = defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        out[e.key][0] += e.self_cpu_time_total / 1e3 / steps
        out[e.key][1] += e.count / steps
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nis_full")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--_worker", nargs=2)
    args = ap.parse_args()
    if args._worker:
        print("PROFILE=" + json.dumps(worker(args._worker[0], args._worker[1], args.steps)))
        return
    res = {}
    for dev in ("cpu", "xdna"):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", args.model, dev,
                            "--steps", str(args.steps)], capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("PROFILE=")), None)
        if line is None:
            sys.exit(f"{dev} worker failed: {p.stderr[-2000:]}")
        res[dev] = json.loads(line[8:])
    keys = set(res["cpu"]) | set(res["xdna"])
    rows = sorted(keys, key=lambda k: -abs(res["xdna"].get(k, [0])[0] - res["cpu"].get(k, [0])[0]))
    print(f"{'op':58s} {'cpu ms':>8s} {'xdna ms':>8s} {'diff':>8s} {'cpu n':>6s} {'xdna n':>6s}   (self time per step)")
    for k in rows[: args.top]:
        c, x = res["cpu"].get(k, [0, 0]), res["xdna"].get(k, [0, 0])
        print(f"{k[:58]:58s} {c[0]:8.2f} {x[0]:8.2f} {x[0]-c[0]:+8.2f} {c[1]:6.0f} {x[1]:6.0f}")
    tc = sum(v[0] for v in res["cpu"].values())
    tx = sum(v[0] for v in res["xdna"].values())
    print(f"{'TOTAL self time':58s} {tc:8.2f} {tx:8.2f} {tx-tc:+8.2f}")


if __name__ == "__main__":
    main()
