#!/usr/bin/env python3
"""Chrome trace of one NIS-shaped training step with named phase spans.

SYNTHETIC PROBE: fixed random inputs/targets (B40, 36x36), Adam without the
application's weight decay, no data loader, scheduler, logging, validation or
checkpointing.  It reproduces the shape of the NIS step (MyNet(3,3,5) + VGG19/L1
loss, BF16), not the exact real-data application.

Settings are explicit and saved next to the trace (<out>.meta.json):

    --mode 2|1|0   fixes the NPU mode (default 2: all NPU paths); "auto" keeps
                   the step autotuner, with enough warmup for it to settle.
    --threads N    torch intra-op threads (default 10).

Wall time is measured on separate, unprofiled steps (profiling with
record_shapes slows the step and keeps tensor references), so quote
`wall_ms_median` from the meta file, not the span of the trace.

    NIS_ROOT=.../lab02_cnn XDNA_CONV3X3W_SYNC_BUILD=1 \\
        python benchmarks/step_timeline.py out.trace.json --mode 2

Phase spans in the trace: step:generator_fwd, step:loss_fwd, step:backward,
step:optimizer (user annotations).  step_timeline_buckets.py reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", default="2", choices=["0", "1", "2", "auto"])
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=0, help="default: 12, or 40 for --mode auto")
    ap.add_argument("--wall-steps", type=int, default=20)
    ap.add_argument("--no-trace", action="store_true", help="only the unprofiled wall time")
    ap.add_argument("--fuse-loss", action="store_true",
                    help="route the perceptual loss through xdna_train.regions (opt-in region)")
    a = ap.parse_args()
    nis_root = os.environ.get("NIS_ROOT", "")
    if not os.path.isdir(nis_root):
        sys.exit("set NIS_ROOT to the NIS lab02_cnn directory")
    sys.path.insert(0, nis_root)
    from models import MyNet
    from train import ImageGenLoss

    import xdna_train
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    xdna_train.register_xdna_device()
    if a.mode != "auto":
        xdna_train.set_npu_mode(int(a.mode))
    dt, dev = torch.bfloat16, "xdna"
    torch.manual_seed(0)
    model = MyNet(3, 3, 5).to(dev, dt)
    loss_fn = ImageGenLoss(4.0, 6.0, device=dev).to(dev, dt)
    fused = False
    if a.fuse_loss:
        from xdna_train import regions
        fused = regions.fuse_perceptual_loss(loss_fn)
        if not fused:
            sys.exit("--fuse-loss: the loss module does not match the region pattern")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 3, 36, 36, generator=g).to(dev).to(dt)
    y = torch.rand(40, 3, 36, 36, generator=g).to(dev).to(dt)

    losses = []

    def step(spans: bool = False) -> float:
        t = time.perf_counter()
        rf = record_function if spans else (lambda _n: _Null())
        with rf("step:generator_fwd"):
            out = model(x)
        with rf("step:loss_fwd"):
            loss = loss_fn(out, y)
        opt.zero_grad()
        with rf("step:backward"):
            loss.backward()
        with rf("step:optimizer"):
            opt.step()
        losses.append(loss.item())
        return (time.perf_counter() - t) * 1e3

    class _Null:
        def __enter__(self): return self
        def __exit__(self, *e): return False

    warm = a.warmup or (40 if a.mode == "auto" else 12)
    for _ in range(warm):
        step()
    xdna_train.reset_fallback_stats()
    xdna_train.reset_mapped_cpu_stats()
    wall = [step() for _ in range(a.wall_steps)]
    stats = {k: {"calls": v["calls"] / a.wall_steps, "ms": v["total_ms"] / a.wall_steps}
             for k, v in xdna_train.mapped_cpu_stats().items()}
    fb = {k: {"calls": v["calls"] / a.wall_steps, "ms": v["total_ms"] / a.wall_steps}
          for k, v in xdna_train.fallback_stats().items()}
    if not a.no_trace:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            step(spans=True)
        prof.export_chrome_trace(a.out)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        commit = ""
    meta = {
        "probe": "synthetic NIS-shaped step (fixed random B40 36x36, Adam without weight decay)",
        "commit": commit, "torch": torch.__version__, "threads": a.threads,
        "fuse_loss": fused, "first_losses": losses[:3], "last_loss": losses[-1],
        "requested_mode": a.mode, "npu_mode_now": xdna_train.npu_mode(),
        "autotune_result": xdna_train.autotune_result(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("XDNA_", "OMP_", "MKL_"))},
        "warmup_steps": warm, "wall_ms_median": statistics.median(wall),
        "wall_ms_all": wall, "async_dw": xdna_train.async_dw_stats(),
        "mapped_cpu_per_step": stats, "fallback_per_step": fb,
    }
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"wall median {meta['wall_ms_median']:.1f} ms  mode {meta['npu_mode_now']}  "
          f"async_dw {meta['async_dw']}")


if __name__ == "__main__":
    main()
