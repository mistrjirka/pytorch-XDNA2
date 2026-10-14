#!/usr/bin/env python3
"""End-to-end speed gate: a whole training step is faster on XDNA than on the CPU.

Trains the benchmark generator with its VGG19 perceptual loss (benchmarks/style_net.py,
BF16, batch 40, 36x36 patches, Adam) in separate processes per device, alternating
CPU / XDNA / XDNA+fused-loss in ABBA order so heat and background load do not favour
one side, and reports the median step time of each.  Default backend settings are used
(the step autotuner picks the NPU mode during warmup), so this is what a user gets.

    python benchmarks/end_to_end_speed.py                 # prints a table, exits 1 on a miss
    python benchmarks/end_to_end_speed.py --rounds 3 --min-speedup 1.3 --min-speedup-fused 1.5

Run on AC power with nothing else using the NPU or the CPU.  Hardware test: not in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def worker(variant: str, steps: int, warmup: int, threads: int) -> None:
    import torch
    sys.path.insert(0, HERE)
    from style_net import StyleLoss, StyleNet, SyntheticPatches
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    amp = variant.endswith("-autocast")
    dev = "cpu" if variant.startswith("cpu") else "xdna"
    if dev == "xdna":
        import xdna_train
        xdna_train.register_xdna_device()
    dt = torch.float32 if amp else torch.bfloat16  # autocast: FP32 master weights, BF16 compute
    torch.manual_seed(0)
    model = StyleNet(3, 3, 5).to(dev, dt).train()
    loss_fn = StyleLoss(4.0, 6.0, device=dev)
    if not amp:
        loss_fn = loss_fn.to(dev, dt)
    if variant == "xdna-fused":
        from xdna_train import regions
        assert regions.fuse_perceptual_loss(loss_fn), "loss does not match the region pattern"
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    data = SyntheticPatches(36, 200000)
    xs = torch.stack([data[i][0] for i in range(40)])
    ys = torch.stack([data[i][1] for i in range(40)])
    xs, ys = xs.to(dev, dt), ys.to(dev, dt)
    times, losses = [], []
    for i in range(warmup + steps):
        t = time.perf_counter()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        val = loss.item()  # materialises everything, like a logging script does
        if i >= warmup:
            times.append((time.perf_counter() - t) * 1e3)
        losses.append(val)
    print("RESULT=" + json.dumps({"variant": variant, "median_ms": statistics.median(times),
                                  "first_loss": losses[0], "last_loss": losses[-1]}), flush=True)


def run(variant: str, args) -> dict:
    env = dict(os.environ, XDNA_AUTO_SETUP="0")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", variant,
                        "--steps", str(args.steps), "--warmup", str(args.warmup), "--threads", str(args.threads)],
                       capture_output=True, text=True, env=env, timeout=1800)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT="):
            return json.loads(line[7:])
    raise RuntimeError(f"{variant} failed:\n{p.stderr[-2000:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=40, help="the step autotuner needs ~21 steps to settle")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cool", type=int, default=20, help="seconds between runs")
    ap.add_argument("--min-speedup", type=float, default=1.2, help="XDNA default vs CPU")
    ap.add_argument("--min-speedup-fused", type=float, default=1.4, help="XDNA with the fused loss vs CPU")
    ap.add_argument("--autocast", action="store_true",
                    help="also time torch.autocast training (FP32 master weights) on CPU and XDNA")
    ap.add_argument("--min-speedup-autocast", type=float, default=1.1)
    ap.add_argument("--json", default="")
    ap.add_argument("--_worker")
    args = ap.parse_args()
    if args._worker:
        worker(args._worker, args.steps, args.warmup, args.threads)
        return 0
    order = ["cpu", "xdna", "xdna-fused"] + (["cpu-autocast", "xdna-autocast"] if args.autocast else [])
    runs = {v: [] for v in order}
    for r in range(args.rounds):
        for v in (order if r % 2 == 0 else order[::-1]):
            res = run(v, args)
            runs[v].append(res)
            print(f"round {r + 1} {v:11s} {res['median_ms']:7.1f} ms/step   loss {res['first_loss']:.3f} -> {res['last_loss']:.3f}",
                  flush=True)
            time.sleep(args.cool)
    med = {v: statistics.median(x["median_ms"] for x in runs[v]) for v in order}
    sp, spf = med["cpu"] / med["xdna"], med["cpu"] / med["xdna-fused"]
    print(f"\nmedian step: cpu {med['cpu']:.1f} ms | xdna {med['xdna']:.1f} ms ({sp:.2f}x) | "
          f"xdna+fused loss {med['xdna-fused']:.1f} ms ({spf:.2f}x)   [{1000 / med['cpu']:.1f} / "
          f"{1000 / med['xdna']:.1f} / {1000 / med['xdna-fused']:.1f} it/s]")
    ok = sp >= args.min_speedup and spf >= args.min_speedup_fused
    if args.autocast:
        spa = med["cpu-autocast"] / med["xdna-autocast"]
        print(f"autocast: cpu {med['cpu-autocast']:.1f} ms | xdna {med['xdna-autocast']:.1f} ms ({spa:.2f}x)")
        ok = ok and spa >= args.min_speedup_autocast
    # the loss must also train sensibly on every device (same first loss, no divergence)
    for v in order:
        for x in runs[v]:
            if not (x["last_loss"] == x["last_loss"]) or x["last_loss"] > 1.5 * x["first_loss"]:
                ok = False
                print(f"FAIL: {v} loss went {x['first_loss']:.3f} -> {x['last_loss']:.3f}")
    if args.json:
        json.dump({"median_ms": med, "speedup": sp, "speedup_fused": spf, "runs": runs}, open(args.json, "w"), indent=2)
    print("PASS" if ok else f"FAIL: need >= {args.min_speedup}x (default) and >= {args.min_speedup_fused}x (fused)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
