#!/usr/bin/env python3
"""Train the NIS generator on real patches and record the loss per step.

Run twice with the same seeds, with and without --fuse-loss (the opt-in
xdna_train.regions loss), then compare the curves:

    NIS_ROOT=.../lab02_cnn python benchmarks/loss_curve_compare.py a.json --steps 400
    NIS_ROOT=.../lab02_cnn python benchmarks/loss_curve_compare.py b.json --steps 400 --fuse-loss
    python benchmarks/loss_curve_compare.py --compare a.json b.json

Same data order, init, optimizer (Adam 1e-4, wd 1e-5, cosine) and NPU mode 2.  The
NPU path is not bitwise reproducible run to run (async dW timing), so the
question is whether the fused curve stays within the run-to-run spread of the
eager one: run the eager side twice (a.json, a2.json) to see that spread.
"""
import argparse
import json
import os
import statistics
import sys
import time


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    n = min(len(r["loss"]) for r in runs)
    print(f"steps compared: {n}")
    w = max(10, n // 10)
    print(f"{'window':>12s}" + "".join(f"{os.path.basename(p)[:14]:>16s}" for p in paths))
    for lo in range(0, n - w + 1, w):
        print(f"{lo:5d}-{lo + w - 1:<6d}" + "".join(f"{statistics.mean(r['loss'][lo:lo + w]):16.3f}" for r in runs))
    print(f"{'first 5':>12s}" + "".join(f"{str([round(x, 1) for x in r['loss'][:5]]):>16s}"[:16] for r in runs))
    for p, r in zip(paths, runs):
        print(f"{os.path.basename(p)}: fuse={r['fuse']} median step {statistics.median(r['step_ms']):.1f} ms, "
              f"final-window loss {statistics.mean(r['loss'][-w:]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--fuse-loss", action="store_true")
    ap.add_argument("--fuse-bn-act", action="store_true")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    import torch
    from torch.utils.data import DataLoader
    nis = os.environ["NIS_ROOT"]
    sys.path.insert(0, nis)
    from data import DatasetPatches
    from models import MyNet
    from train import ImageGenLoss
    import xdna_train
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)
    torch.manual_seed(100)
    model = MyNet(3, 3, 5).to("xdna", torch.bfloat16).train()
    loss_fn = ImageGenLoss(4.0, 6.0, device="xdna").to("xdna", torch.bfloat16)
    if a.fuse_bn_act:
        from nis_adapters import fuse_nis_norm_act
        assert fuse_nis_norm_act(model) > 0
    if a.fuse_loss:
        from xdna_train import regions
        assert regions.fuse_perceptual_loss(loss_fn), "loss does not match the region pattern"
    ds = DatasetPatches(os.path.join(nis, "data/VID20261009180405/train/input"),
                        os.path.join(nis, "data/VID20261009180405/train/target"),
                        patch_size=36, num_patches=200000)
    torch.manual_seed(131)
    loader = DataLoader(ds, batch_size=40, shuffle=True, num_workers=1, pin_memory=True,
                        generator=torch.Generator().manual_seed(331))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)
    losses, step_ms = [], []
    it = iter(loader)
    for i in range(a.steps):
        t = time.perf_counter()
        x, y = next(it)
        x, y = x.to("xdna", torch.bfloat16), y.to("xdna", torch.bfloat16)
        val = loss_fn(model(x), y)
        opt.zero_grad()
        val.backward()
        opt.step()
        sched.step()
        losses.append(val.item())
        step_ms.append((time.perf_counter() - t) * 1e3)
    json.dump({"fuse": a.fuse_loss, "fuse_bn_act": a.fuse_bn_act, "loss": losses, "step_ms": step_ms, "steps": a.steps}, open(a.out, "w"))
    print(f"done {a.steps} steps, median {statistics.median(step_ms[20:]):.1f} ms, last loss {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
