# Fusion analysis: lazy pointwise values and the op graph (2026-10-23)

Goal: run cheap per-element ops (BatchNorm apply, activations) inside the next
op that reads the data anyway, instead of as separate passes over the tensor.
Long term the same records move into the NPU conv prologue and the tensors stay
in the NPU layout between ops. This document records stage 1 and the graph
analysis that chooses the next stages.

## Stage 1: lazy pointwise values (done)

- `leaky_relu` and training `BatchNorm2d` on contiguous NCHW bf16 return a lazy
  value `act(source * scale + shift)` per channel (`XdnaVirtualValue` in
  `torch_xdna_device.cpp`). A LeakyReLU on a lazy BatchNorm output composes into
  one value.
- Consumers that apply it while reading: the conv3x3w forward pack, the NPU dW
  pack, and `leaky_relu_backward` (mask from the BatchNorm input).
- Every other access writes the bytes first (`materialize_virtual`), also
  through aliases, resizes and `set_`. A source keeps a list of its dependents
  and materializes them before any access, so a later write cannot change them.
- Exactness: bit-equal to eager XDNA and to CPU for forward, all gradients and
  the running mean (`tests/run_lazy_pointwise.py`). The lazy BatchNorm
  reproduces ATen's rounding: bf16 mean and variance, invstd from the bf16
  variance, affine from the saved values. The running variance can differ by
  one bf16 step (TODO in `lazy_batch_norm_training`, eval mode only).
- `XDNA_LAZY_POINTWISE=0` disables it. `xdna_train.allocator_stats()` reports
  `lazy_created`, `lazy_consumed` and `lazy_materialized`.

Gotcha found on the way: the first kernels used branches (NaN check,
LeakyReLU select) and did not vectorize. They ran at 0.4 GB/s per core and made
the lab04 step 35 % slower (585 -> 785 ms). Branch-free bit blends made them
7-11x faster. Any kernel on the lazy path runs as often as ATen's vectorized
loops it replaces, so check `-fopt-info-vec` for it.

### Measured (ABBA, all-NPU mode, median ms per step)

| Network | Eager | Lazy |
|---|---|---|
| lab04 baseline (B64, 64x64) | 585, 592 | 599, 577 |
| resnet18 (zoo) | 48.0, 59.4, 40.5, 38.9, 53.5 | 39.8, 41.8, 41.2, 40.2, 44.1 |
| unet | 57.2, 57.0 | 60.7, 56.3 |
| style_net | 101.5, 108.3, 143.3, 133.7, 110.8 | 134.0, 104.1, 135.3, 142.1, 113.1 |
| conv_autoencoder | 19.8, 19.6 | 18.9, 20.6 |
| vgg11_128 | 159.2, 158.0 | 161.2, 155.2 |

No regression. Gains are within noise (style_net is bimodal in both modes).
The graph below shows why: most lazy values meet a reader that cannot apply
them.

## Op graph (benchmarks/record_graph.py)

One training step recorded below autograd for lab04 (baseline and composite),
style_net, resnet18, unet, conv_autoencoder and transformer.

What reads the output of each forward pointwise or norm op (lab04 baseline):

| Producer | Count | Readers |
|---|---|---|
| BatchNorm | 13 | LeakyReLU only (composes) |
| BatchNorm | 7 | residual `add` |
| LeakyReLU | 8 | NPU conv only (applied in the pack) |
| LeakyReLU | 6 | `clone` (ResNet `x_in = x.clone()`) |
| LeakyReLU | 5-6 | `cat` (U-Net skips) |
| LeakyReLU | some | CPU convs (stride 2, 1x1, RGB): need bytes |

Counters per lab04 step: 40 created, 18 consumed, 29 materialized. The
materializations are writes eager also did, moved later, so they cost nothing
extra. The other networks show the same shape:

- resnet18, unet: most convs run on the CPU (stride 2, 1x1, 7x7, cost model).
  Every lazy value is materialized (20 of 20, 10 of 10).
- torchvision uses `relu_` (in place) and `relu`, which are not lazy yet.
- transformer: no conv, no BatchNorm. The pointwise ops sit between matmuls
  and layer norms (`add -> native_layer_norm`, `gelu -> addmm`).

## Next stages, by what they unblock

1. **More lazy producers**: `relu`, `relu_` (in place on a lazy value just
   adds the activation to the record), `add` of a lazy value and a tensor
   (residual), `clone` of a lazy value (a new record on the same source).
   Unblocks resnet18-style networks and the ResNet blocks of lab04.
2. **Layout consumers**: `cat` and nearest `upsample` that keep lazy inputs
   lazy, with the conv pack reading several sources. The removed
   `XDNA_LAZY_DECODER_VALUES` did this for cat + upsample and gained nothing
   because the conv still repacked. It pays only together with stage 4.
3. **BatchNorm statistics in the conv unpack**: the unpack touches every
   output element. Computing per-channel sums there removes the statistics
   pass (20 per lab04 step).
4. **Keep the NPU layout between ops (Option A)**: the conv writes its output
   in the input layout, and lazy values record "bytes are in the packed
   buffer". This targets the 147 ms of pack and unpack per lab04 step, the
   largest item. Stages 1-3 are its building blocks: the lazy record is the
   tag that says which form a tensor is in.
5. **NPU prologue/epilogue**: move the pointwise record into the conv kernel
   once the input is already packed. Budget measured on 2026-10-22: the
   forward conv core uses 2.8 of 16 KB program memory and about 55 of 64 KB
   data memory. Per-channel scale/shift (up to 4 KB) and statistics (1 KB) fit,
   a streamed residual input (32 KB double-buffered) does not.

## Full graph recording (way 1): not needed yet

Every stage above is local: the consumer decides when it sees a lazy input.
A full recorded graph is needed only to pick per-network engine variants
(which add-ons to compile in) or to plan buffers across ops. Revisit at stage
5, when kernel variants exist. `benchmarks/record_graph.py` already gives the
graph offline.
