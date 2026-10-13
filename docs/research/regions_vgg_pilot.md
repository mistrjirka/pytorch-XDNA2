# Training regions: frozen-VGG perceptual loss pilot (branch `bfp-input`, 2026-10-13)

Status: working, tested, opt-in.  Not on main.

## What it is

`xdna_train/regions.py` runs the pattern "frozen conv/relu/maxpool prefix, mean
squared error on selected taps" as one autograd node (`frozen_feature_mse`).
Eager PyTorch clones each tap, concatenates, subtracts, squares, means, and runs
the matching backward over the 16.6 MB concatenation.  The region never builds the
clones or the concatenation: native fused kernels (`_native/region_kernels.cpp`)
do the tap reduction, tap gradient, ReLU backward and 2x2 max-pool forward/backward
in one pass each.  The convolutions still go through the normal backend.

Opt in: `regions.fuse_perceptual_loss(loss_module)` (duck-typed on `perceptual_vgg.model.features`,
`.feature_layers`, `.norm`).  Eligibility is semantic (op types, bf16, XDNA device, frozen weights,
x requires grad); otherwise the module's own method runs.  `XDNA_REGIONS=0` disables.

Contract, numerics and saved state are in the module docstring.  Nothing is NIS or
B40 specific; the NIS adapter lives in `benchmarks/step_timeline.py --fuse-loss`
and `benchmarks/loss_curve_compare.py --fuse-loss`.

## Measurements (AC, mode 2, 10 threads, synthetic NIS-shaped probe, wall time unprofiled)

| Variant | Step ms (median of paired runs) |
|---|---|
| eager loss | 106.5 (first ABBA x6), 109.0 (second, noisy: 103-122) |
| fused loss, backend cost model places convs | 94.5, 93.4 |
| fused loss, measured per-conv placement | 92.9 (within noise of the line above) |

Every fused run beat its neighbouring eager run in both comparisons.  Loss forward
span dropped 24.6 -> 13.6 ms in the profiled trace.

Kernel timings at the tap size (3.3M bf16 elements, 10 threads): tap error sum 0.06 ms,
relu+maxpool 1.1 ms (torch max_pool2d alone 2.3), relu backward + tap 0.25, max-pool backward
0.10 (was 1.8 before restructuring), add-tap 0.2.  The earlier versions were scalar:
GCC would not vectorize loops with short-circuit NaN tests, `if` argmax chains, an early
return in the rounding helper, or float compares under trapping-math (see the notes in
`region_kernels.cpp`).  The forward pool row kernel now has an AVX2 intrinsics path (runtime-dispatched, scalar fallback for
odd widths): relu+maxpool 1.1 -> 0.18 ms at 40x64x36x36 (torch max_pool2d 1.9).  Full step after the change:
fused median 90.7 ms (88.7-95.2, 4 runs) vs eager 104.8 ms; the pool's own share (~1-2 ms) is not separable from noise.

Placement of the region's own convolutions (`XDNA_REGION_PLACEMENT=measure|cost|npu|cpu`):
in isolation the NPU saves 0.6-0.8 ms per call on 64->64@36 and 64->128@18 (fwd and dX);
the 3-channel first conv is faster on CPU (0.3 vs 1.3 ms).  B80 (target+generated batched)
does not help.  The placer picks accordingly (NPU for those two), but the full-step gain over
the cost model is not distinguishable from noise, so the default stays measure only because
it is harmless; do not quote it as a win.

## Training-quality check

`benchmarks/loss_curve_compare.py`, real NIS patches, 400 steps, same seeds and data order:
two eager runs are bit-identical (loss per step); the fused curve is within 0.1% in every
40-step window (32.806 vs 32.837 ... 11.223 vs 11.213 final window).  First losses equal.
This is a short run, not a convergence study.

## Not done

- Real-data end-to-end timing on the actual application loop (scheduler/logging/validation).
- Target features early (while the generator runs), NPU-resident chain (needs a layout that
  keeps relu/pool/taps on device; all VGG intermediates are loss taps, so residency alone saves nothing here).
- Prewarming of the NPU program switch: separate opt-in experiment, not started.
