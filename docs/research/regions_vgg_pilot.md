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

## Fused training BatchNorm + LeakyReLU (`regions.fuse_norm_activation`, 2026-10-13)

Explicit pairing API in the style of `torch.ao.quantization.fuse_modules`: the caller declares that the
activation is applied directly to the norm output.  A module walk cannot know that: in MyNet's `ResNetBlock`
`norm2` goes through an add before the activation, and a blind pairing would apply LeakyReLU twice (slope
0.2 twice is not 0.2).  The NIS declarations live in `benchmarks/nis_adapters.py` (norm in ConvBlock /
UpsampleConvBlock, `norm1` in ResNetBlock; the shared `relu` module is safe because it only passes through
tensors the fused norm produced).  Both modules keep parameters, buffers and state_dict keys.

Native kernels (`region_kernels.cpp`): per-(n,c) plane mean and sum of squared deviations in fp32, planes merged
per channel in double (Chan), one apply pass with the activation, running-stat update (momentum, unbiased
variance) in the same call; backward = one reduction pass (dbias, dweight with the activation mask `y > 0`)
and one dx pass.  Eligibility: training mode, affine, tracked statistics with a float momentum, bf16
contiguous NCHW on XDNA, more than one value per channel, and at least 4 MB per tensor
(`XDNA_REGION_NORM_ACT_MIN_BYTES`); everything else runs the eager pair unchanged.

Accuracy (`tests/run_regions_bn.py`, 7 shapes, 3 steps each, vs an fp32 CPU reference): the fused pair is closer to
fp32 than the eager bf16 pair on output, input / weight / bias gradients and both running statistics in every case
(e.g. 40x64x36x36: dx 0.0149 vs 0.0181, dW 0.0089 vs 0.0138).

Speed (microbenchmark, reliable): fwd+bwd, eager -> fused: 13.3 MB tensor 3.46 -> 2.29 ms, 6.6 MB 1.18 -> 0.82 ms; 3.3 MB and
smaller: no gain (0.40 -> 0.43, 0.15 -> 0.14), hence the size threshold.  Profile (clean, cooled): BatchNorm +
activation 9.9 ms of host time eager -> ~1.2 ms, with the fused op itself ~7 ms, net -3 to -5 ms op time.  Full
step (wall): NOT established.  My pairs suggested -1 to -3 ms, but they needed outliers dropped (the machine throttles
after a few minutes of back-to-back runs) and an external audit of the same configuration (loss-only 93.5 / 96.2 ms vs
loss+BN 108.5 / 92.3 ms, nothing discarded) could not separate the two; run-to-run variation of identical runs is larger
than the claimed gain.  What is solid: the microbenchmark gain on the two largest layers (~1.2 + 0.4 ms) and the
op-time profile.  Do not count the BN fusion as a confirmed full-step gain.  It is also far below the 6-8 ms I
estimated: eager BatchNorm on the large tensors is already vectorised, and only the two 36x36 128-channel layers
have enough bytes to gain.

Training check (400 real-patch steps, loss + BN fused vs eager): windows within +-1% (first window +0.5%, last
-0.9%), same trend; the fused arithmetic rounds differently so the curves are not bit-equal.

Backend bug found and fixed on the way: eager `batch_norm` (forward and backward) on a non-contiguous bf16 XDNA
input (e.g. a transposed view) returned wrong, run-to-run varying results (6-75% error vs CPU).  The handlers passed
a strided CPU alias to PyTorch's `native_batch_norm[_backward]_out`, which picks its memory format from the strides
and mis-indexes a permuted view.  Such inputs are now made contiguous first (`batch_norm_layout_safe`); test in
`run_regions_bn.py`.

## Contract bugs found by an external audit (fixed 2026-10-14, `tests/run_regions_contract.py`)

All five reproduced before the fix:
- `copy.deepcopy` of a model with a fused norm/activation pair, or of a fused loss, kept using the ORIGINAL module's
  parameters (the wrappers were closures over the original): the copy's batch counter did not move, the original got
  the weight gradient, and a copied loss with zeroed feature weights still returned 0.17.  The wrappers are now module
  level functions bound with `types.MethodType` (deepcopy rebinds them to the copy) and read everything from `self`.
  Paired modules must be copied together (the pairing is stored on the norm).
- Feature weights that became trainable after `fuse_perceptual_loss` silently lost their gradients.  Eligibility is
  re-checked on every call (conv weights / biases must not require grad) and the chain is rebuilt if the layers or
  weights were replaced; otherwise the module's own method runs.
- The "already activated" mark was never consumed, so a second `act(y)` on the activated tensor was skipped (15.7%
  error).  The mark now covers exactly one activation call.
- A changed `negative_slope` was ignored on the fused path (27% error); it is read on every call.

Not changed (documented): pickling a whole fused module with `torch.save(model)` is not supported (bound methods in
the instance dict); save `state_dict()` instead.

## Using the fusions in the real NIS script

The unchanged coursework `style_transfer.py` defaults to CPU and calls neither adapter; the benchmarks switch them on
explicitly.  After `model.to('xdna', bf16)` and building the loss on XDNA the script would need, for example:

    from xdna_train import regions
    regions.fuse_perceptual_loss(loss_fn)            # loss_fn = ImageGenLoss(...) on xdna
    sys.path.insert(0, '<repo>/benchmarks'); from nis_adapters import fuse_nis_norm_act
    fuse_nis_norm_act(model)                         # optional; gain not established
