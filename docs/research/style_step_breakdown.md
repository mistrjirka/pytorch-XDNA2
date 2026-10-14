# style-transfer training step: where the 108 ms go (experimental BFP-input kernel, not merged, 2026-10-13)

Baseline for the offload plan (handoff removal vs. CPU-tail offload).  One
steady-state step, mode 2 (`XDNA_NPU_MODE=2`, `XDNA_CONV3X3W_SYNC_BUILD=1`), B40
36x36, StyleNet(3,3,5) + VGG19/L1 loss, BF16, 10 threads, AC.  Wall 107.8 ms
unprofiled; the profiled step below is 110 ms, so the shares are representative.

**Measure with `XDNA_NPU_MODE=2` pinned.**  Without it the step autotuner may
still be sampling or sit on CPU-only mode, the NPU convs silently never run (no
`TRACE` lines, 158-189 ms steps) and every per-op profile is of the CPU fallback.

Reproduce: `benchmarks/step_timeline.py <out.json>` (chrome trace of one step),
then `benchmarks/step_timeline_buckets.py <out.json> <loss_start_ms> <bwd_start_ms>`
(phase boundaries are read off the trace: first VGG conv, `PowBackward`).

## Main-thread wall time by kind of work (ms, exclusive time)

| | generator fwd | VGG loss fwd | backward | total |
|---|---|---|---|---|
| CPU convs (oneDNN) | 8.8 | 7.6 | 26.1 | **42.5** |
| NPU conv calls (host side: pack, wait, unpack, dispatch) | 14.1 | - | 11.0 | **25.1** |
| other elementwise / autograd / loss math | 1.4 | 3.8 | 6.9 | 12.1 |
| copies | 0.7 | 2.4 | 3.2 | 6.3 |
| BatchNorm | 2.4 | - | 3.9 | 6.2 |
| cat / resize | 2.5 | 1.8 | 1.2 | 5.5 |
| max-pool | - | 3.7 | 0.3 | 4.0 |
| activation | 1.4 | 0.1 | 2.4 | 3.9 |
| optimizer | - | - | 1.0 | 1.0 |
| **total** | 31.3 | 19.3 | 56.0 | 106.6 |

## CPU convs by layer (42.5 ms)

| layer | ms | note |
|---|---|---|
| residual 128->128 @9x9, 10 convs | 5.3 fwd + 13.0 bwd = **18.3** | 0.5 ms fwd, 0.65 ms per bwd call; ~1 TFLOPS |
| VGG prefix (3->64, 64->64 @36, 64->128 @18), target + generated | ~8.5 fwd + ~4 bwd | dX only, frozen weights |
| encoder stride-2 64->128 @36 | 1.3 fwd + 5.2 bwd | |
| other encoder / 128->128 @18, final 1x1 | ~4 | |

## NPU conv calls (trace, per call)

pack 8.9 ms + unpack 6.4 ms + wait 4.4 ms + sync 1.4 ms over 6 calls (3 fwd, 3
dX).  The one large wait (3.16 ms, first forward conv, 18x18) is the hardware
context switch away from the dW program; the other five waits are 0.14-0.35 ms.
dW (async, helper thread, off the main thread): 14.8 ms of NPU run+sync time per
step, hidden under the 26 ms residual/encoder backward tail.

## Reading (weakened after review)

What the trace supports: main-thread host time by operator kind in one profiled
step.  It does not give a helper-thread or NPU-device timeline, and the "NPU conv
calls" row is the exclusive time of `aten::*convolution*_overrideable`, a native
dispatch wrapper that also runs for the CPU-fallback branch; read it as "dispatch
wrapper incl. NPU pack/wait/unpack", not as hardware-proven NPU time.

- Largest recorded blocks: CPU convs 42.5 (serial forward/loss work AND the
  backward tail; residual 18, VGG ~13, encoder ~8), dispatch wrapper 25 (pack 8.9 +
  unpack 6.4 + wait 4.4 + sync 1.4 over the 6 traced calls), BN + activation + cat/resize 15.6.
- The 15.6 ms is not all decoder handoff glue: it includes residual-block BN and
  activation and the VGG feature concatenation (1.8 ms in loss forward).  Do not
  add "15 ms conversions + 15 ms glue" and call 30 ms removable.
- Estimates, not measurements: NPU activity "~20 ms", "0.05 ms compute" for a 9x9
  conv, "10-15 ms realistic saving".  Hidden dW has little *exposed* waiting
  here, not zero cost (bandwidth, program switches).
- The 3.2 ms wait on the first forward conv is observed.  That it is a context
  switch away from the dW program, and that pre-triggering it wins, is
  unverified: prewarming is a separate opt-in experiment (own on/off A/B of
  complete steps, gradients and energy), not a promised 3%.

## Capture settings (P0 closeout)

`step_timeline.py` now takes `--mode 2|1|0|auto`, `--threads`, saves
`<out>.meta.json` (commit, torch, env, requested/actual NPU mode, autotune
result, async-dW counters, native counters per step, unprofiled wall times), and
writes named `step:*` phase spans that `step_timeline_buckets.py` reads (no manual
timestamps).  The bucket script keeps one stack per (pid, tid).  Wall time comes
from separate unprofiled steps.  It is still a synthetic probe (fixed random B40
inputs, Adam without weight decay, no loader/scheduler/validation); a warmed
real-data run is still to be confirmed.

Re-measured with the new script (fixed mode 2, 10 threads): 106.2 ms median
unprofiled.  Automatic policy: the autotuner selected mode 2 (samples 108-136 ms,
noisy; median wall 114 ms in that run).

## Prewarm experiment (opt-in `XDNA_PREWARM_FWD=1`, default off)

When the dW helper thread finishes the last queued dW of a step it runs the smallest ready
forward stream once on scratch buffers (about 3.4 ms of NPU time, in the background), so the
NPU's switch back to the forward/dX program happens during the CPU backward tail.

- Mechanism verified: the first forward conv's host wait falls from 2.8 ms to 0.4 ms
  (`XDNA_CONV3X3W_TRACE`), so that wait was the program switch.
- Full-step effect NOT established.  Separate-process ABBA (warm 40, 40 timed steps, fused
  loss, 8 runs each): median 92.55 off vs 92.25 on (-0.3 ms; mean without the outlier -1.1).
  In-process alternation (toggle `set_prewarm_fwd`): steady blocks about -2 to -5 ms, but a
  toggle leaves a transient of more than a block (first block after switching is 5-15 ms
  slower), thermal drift hit the late blocks, and the clean pairs are few.  Run-to-run noise
  is about +-2 ms.  Best reading: a gain of roughly 0-3 ms, not shown to beat noise.
- Not a promised 3%.  To settle it: a long run with many separate-process pairs, or find the
  toggle transient (not understood).
