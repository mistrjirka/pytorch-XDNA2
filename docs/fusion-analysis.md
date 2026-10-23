# Fusion analysis: lazy pointwise values and the op graph (2026-10-23)

Goal: run cheap per-element ops (BatchNorm apply, activations) inside the next
op that reads the data anyway, instead of as separate passes over the tensor.
Long term the same records move into the NPU conv prologue and the tensors stay
in the NPU layout between ops. This document records stage 1 and the graph
analysis that chooses the next stages.

## Result of stages 1-4 (lab04, B64, 64x64, all-NPU mode)

ABBA, 25 steps each, `XDNA_LAZY_POINTWISE=0` vs default: median 604, 606, 584
vs 524, 518, 515 ms; min 568, 571, 536 vs 488, 502, 495 ms. About 14 % per
step, with every forward value and most gradients bit-equal to eager.

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

Second gotcha (stage 1b): the extension builds for baseline x86-64 (SSE2),
while ATen dispatches to AVX-512. A lazy value that a CPU op materializes
was 2x slower than eager BN + ReLU. The hot loops now carry
`target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")`, and their
helpers are `always_inline`. Without that, GCC stopped inlining inside the
clones, and affine + activation stayed at baseline speed. One core, 8M
elements: 7.5 -> 2.1 ms. BN -> ReLU -> CPU reader is now faster lazy than
eager (16x64x56x56: 0.40 vs 0.70 ms). The conv pack and unpack loops
(`npu_conv3x3w.pack` / `.unpack`, about 150 ms per lab04 step) are still
baseline-only, so they are the next candidate for the same treatment.

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

## Stage 1b: ReLU, clone and the residual add (done)

- `relu` is lazy. It is its own activation, not LeakyReLU with slope 0: ATen's
  relu is `clamp_min(x, 0)`, which maps -inf to 0, while `-inf * 0` is NaN.
- `relu_` on a lazy affine value (torchvision's `BatchNorm -> ReLU(inplace)`)
  adds the activation to the record. On any other tensor it is eager.
- `threshold_backward` (relu's backward reads the relu result) takes the mask
  from the value's input with a select, so an infinite gradient stays 0.
- `clone` of a lazy value is a new lazy value of the same source.
- `add` and `add_` with alpha 1 on same-shape contiguous bf16 compute one or
  two lazy operands while adding. A plain operand is only read, so the lazy
  values that read it stay lazy.
- Tests: the `res` block in `tests/run_lazy_pointwise.py` (conv -> BN ->
  relu_ -> clone -> conv -> BN, + skip -> relu -> conv) is bit-equal to eager
  XDNA and consumes 4 lazy values in forward with no materialization.
- lab04: 81 % of lazy values are now consumed without a write (stage 1: 45 %).
- Measured (ABBA, median ms; eager vs lazy): lab04 603, 563 vs 579, 609;
  resnet18 46.5, 37.6, 52.3 vs 36.7, 40.4, 43.9; unet 52.7, 67.1, 56.8 vs
  64.0, 55.1, 100.7; style_net, conv_autoencoder and vgg11_128 equal within
  noise. The machine noise (up to 2x on unet) is larger than the effect. The
  per-op profile of unet shows no slower op.

## Stage 2: lazy cat and upsample (done)

- A lazy value is now a list of channel segments. Each segment has its own
  source, per-channel scale/shift/activation and nearest-upsample factor
  (`XdnaLazySegment`, `conv_pack.h`).
- `cat` along channels of contiguous bf16 NCHW tensors and nearest 2x
  `upsample_nearest2d` return lazy values. Lazy inputs keep their segments;
  plain inputs become segments read as they are.
- One pack kernel (`xdna_conv3x3w_pack_segments_bf16`) reads any segment list,
  so the conv packs `upsample(cat(leaky(bn(a)), b))` straight from `a` and `b`.
  An activation on a cat or an upsample moves into every segment.
- Elementwise consumers (add, the activation backwards) still need a single
  segment and otherwise read written bytes.
- lab04 (ABBA, median ms; eager vs lazy): 605, 576 vs 557, 510; min 557, 536
  vs 520, 487. About 9 % per step: the decoder no longer writes its cat and
  upsample outputs (up to 134 MB each).

## Stage 3: BatchNorm statistics in the conv unpack (done)

- The conv3x3w forward unpack adds every output value and its square per
  channel (double) while the rows are in cache, and attaches the sums to the
  output tensor with its write epoch (`XdnaChannelStats`). Not with a conv
  bias: eager adds it after the unpack.
- Lazy BatchNorm takes mean and variance from them when they still match, so it
  no longer reads its input. `allocator_stats()["bn_stats_reused"]` counts it.
- Results stay bit-equal to eager XDNA in the tests: the double sums give the
  same bf16-rounded mean and variance as the statistics kernel.
- lab04 (ABBA, median ms; eager vs lazy): 565, 565 vs 530, 531. BatchNorm
  forward: 16.8 -> 1.3 ms per step; the unpack: unchanged (81 vs 82 ms).

## Stage 4: keep the NPU layout between ops (done)

A conv3x3w output no longer has to be unpacked to NCHW. It stays in the rows
the NPU wrote (`XdnaConvRows`: blocks of `[grid rows][128]` on the padded
strip grid), and the tensor is a lazy value of that kind
(`XDNA_LAZY_CONV_ROWS`, on by default).

- Forward: lazy BatchNorm takes the stage 3 sums (now from a stats-only pass
  over the rows), LeakyReLU/ReLU/cat/upsample build records on the rows tensor,
  and the next conv packs straight from the rows: channels are contiguous
  there, so no transpose.
- Backward: the input-gradient conv also keeps its output in rows. LeakyReLU
  and ReLU backward, BatchNorm backward, upsample backward, the residual add
  and the gradient accumulation compute on rows of one grid
  (`xdna_rows_*` in `conv_pack.cpp`) and return rows, so the chain
  conv -> BN -> act -> conv stays in rows in both directions. Channel slices of
  a rows tensor (the cat backward) are read in place; an add of a rows slice
  and an NCHW tensor unpacks only the slice into the sum.
- Anything else unpacks the tensor once (`materialize_virtual`), which eager
  did in the conv itself. With `XDNA_PROFILE_MAPPED_CPU=1`,
  `lazy.write[<op>]` in `mapped_cpu_stats()` shows which ops do that, so where
  a chain breaks.
- Exactness: forward, activation backward, upsample backward and the adds are
  bit-equal to eager. BatchNorm backward sums in double where ATen sums in
  float: in the tests about 4 values per million differ by more than one bf16
  step (relative error 7e-6).
- Gotcha: the unpack used to run while the NPU computed the next piece. Without
  it the NPU time shows (`dx.run` +19 ms, `fwd.run` +7 ms per lab04 step).
  The gain is the unpack minus that.
- lab04 per step (averages of 3 ABBA runs): unpack 90 -> 28 ms, BatchNorm
  backward 58 -> 44 ms, net about -37 ms. Step min 568, 564, 570 vs 533, 530,
  518 ms (medians noisier: 578, 607, 626 vs 566, 602, 547).
- Not covered yet: the pack itself (still about 70 ms per lab04 step; from rows
  it runs at about 10-16 GB/s against 49 GB/s for a copy), NCHW inputs from the
  CPU convs (stride 2, 1x1), and convs with a bias (eager adds it after the
  unpack, so they keep unpacking).

## Stage 5: NPU prologue/epilogue (evaluated, not built)

Superseded by stage 6, which measured the direct read on hardware; kept for the
numbers.

Measured on lab04 with stages 1-4 in place:

- Prologue (BN affine + activation on the core, on the input window): the
  pointwise is 5 % of a host pack (rows 8.71 ms, rows + affine 9.17 ms for a
  134 MB input, 16 threads). The pack is memory-bound, so moving the
  pointwise to the NPU saves about 3-4 ms per step. It also costs NPU time,
  which is no longer hidden behind the unpack, and the core must re-zero the
  padding rows (affine(0) = shift).
- Reading the previous conv's rows directly on the NPU (no host pack): the
  next input is the previous output shifted by W + 3 rows, so a DMA pattern can
  read it, but only for one-strip images (W <= 36). Wider images need
  neighbour columns from other strips. For lab04 that covers the 16x16 and
  32x32 convs, about 12-24 ms of pack per step. The deferred dW reads the
  forward's packed copy, so without a host pack it would pack again in backward,
  unless the dW kernel gets the same prologue.
- Epilogue (BatchNorm sums on the core): the host stats pass over the rows is
  about 7 ms per step.
- Exactness: the AIE has no IEEE fp32 multiply; x * scale + shift would not
  round as the host does, so results would differ from eager by bf16 steps.
- Risk: a wrong DMA pattern or lock protocol hangs the array (see the rules in
  `_designs/conv3x3w.py`), and recovery may need a driver reload.

Verdict: at most about 5 % of the step for a new kernel variant, a dW variant
and hang risk. Not built now. Larger items first: the host pack itself (about
70 ms per step at 10-18 GB/s, against 49 GB/s for a copy) and the NPU time
that the removed unpack no longer hides. Revisit when the pack is near copy
speed, or if the NPU kernel is rebuilt for other reasons.

## Stage 6: faster pack, 64-wide window, direct conv input

Each part measured on lab04 (B64, 64x64, `XDNA_NPU_MODE=2`), ABBA, 25 steps.

- Pack stores (`store_packed_row`): a full 64-channel packed row is two
  whole cache lines, written with streaming stores. A normal store first reads
  the line it overwrites, so the memory-bound pack moved about 1.5x the bytes
  of a copy. Pack kernel alone: 7.2 -> 6.1 ms for 134 MB, the speed of a
  parallel `memcpy`. lab04: about 4 %.
- 64-wide window (`conv3x3w.MAX_WIDTH` 36 -> 64): 64x64 convs ran as two strips
  of 34 columns, so the pack wrote halo columns and the NPU computed 8 % more
  rows. The window is now 198 x 64 bf16; with the double-buffered weights, the
  output tile and the stack it leaves about 2 KB of the 64 KB core memory.
  Packed `GROUP_ROWS` 2126 -> 2182 (the dW reads that layout). NPU time per
  pass: 64x64 -2 to -6 %, small shapes +1 to +5 % (more window rows per core).
  lab04: about 3 %.
- Direct input (`XDNA_CONV_DIRECT_INPUT`, default on). Host-written rows on a
  one-strip grid go to an NPU buffer laid out as the conv reads its input:
  NHWC, buffer row q = padded-grid row q (a zero margin of wp + 1 rows in
  front), padding pixels and a 198-row tail zero. The conv reads them through
  an `--in-cin R` stream (`[chunks, 4 cores, 198 rows, 64]` with strides
  `[64, 64 R, R, 1]`; windows overlap in place, no per-group halo copy). The
  NPU dW reads X and dY the same way (`--x-rs`, `--d-rs`). Bytes read are the
  bytes a pack would write: gradients are bit-equal with the feature off
  (`tests/run_conv_direct_input.py`).
  - Verified first on hardware outside torch (32x32 and 64x64, C 128/256):
    same NPU time and results as the packed input, also with the output
    written as strided NHWC rows and for conv -> conv with no host step.
  - Who qualifies: only host-written rows (BatchNorm/activation backward,
    adds, upsample backward, cat-backward slices of them), C a multiple of 64.
    In lab04 forward every conv input is a lazy `LeakyReLU(BN(...))`, which
    needs the BN and activation on the NPU core (not built), so only dX
    convs read directly.
  - The CPU dW of small convs read dY as NCHW, which ends the rows chain: the
    16x16 trunk (14 convs, 2.4 GMAC dW each, under the 3 GMAC NPU dW gate)
    never reached a direct read. The gate is now 2 GMAC (`device.py`): lab04
    median 423-441 vs 533-557 ms per step with the trunk dW on the NPU
    (mostly not the direct read: the CPU dW competed with host work). 1 GMAC
    was measured before as 20-45 % slower for 1.2-1.3 GMAC convs (program
    switch), so the gate stops above those. Zoo (noisy, ABBA): style_net and
    conv_autoencoder faster, resnet18, unet and vgg11 within noise.
  - Removing a pack exposes the NPU time it overlapped: lab04 pack -6 to -10
    ms, dX NPU wait +6 ms. Net 1-3 % with the trunk dW on the NPU.
- Layout: the rows kernels take block and row strides per operand
  (`XdnaRowsRef`): NPU outputs stay in 128-channel blocks, host rows are NHWC,
  so a channel slice of host rows is just an offset.

Stage 6 total (lab04, ABBA vs commit 806ac70, 25 steps): median 440, 421, 478
vs 634, 564, 615 ms; min 407, 407, 416 vs 540, 508, 525 ms.

Not covered yet: the activation backward of a lazy cat (`leaky(cat(...))` in a
decoder) is not a rows op, so cat-backward slices stay NCHW and are packed.

Gotchas:
- Host rows buffers have their own pool: the CPU writes them, so they must
  never become NPU outputs (a dirty cache line could land over an NPU result).
  Writers fill only valid rows; a buffer is zeroed when its geometry changes.
- A direct read flushes the whole buffer each time (dX and dW both do).
- Direct reads need 4-byte aligned rows (even row stride and channel offset)
  and a row stride below 1024 (the dW group stride must stay under 2^20
  words).

## Next stages, by what they unblock

1. ~~More lazy producers~~: done (stage 1b).
2. ~~Layout consumers~~: done (stage 2). Unlike the removed
   `XDNA_LAZY_DECODER_VALUES`, the conv pack reads the inputs directly, so it
   gains without stage 4.
3. ~~BatchNorm statistics in the conv unpack~~: done (stage 3).
4. ~~Keep the NPU layout between ops~~: done (stage 4).
5. ~~NPU prologue/epilogue~~: evaluated, not built (stage 5 above).
6. ~~Faster pack, 64-wide window, direct conv input~~: done (stage 6).
7. Forward direct input: BN scale/shift + activation + padding mask on the
   conv core (and on the dW core, which reads the same X). Needs a fourth
   input stream for scale/shift and a tolerance instead of bit-equality.

## Full graph recording (way 1): not needed yet

Every stage above is local: the consumer decides when it sees a lazy input.
A full recorded graph is needed only to pick per-network engine variants
(which add-ons to compile in) or to plan buffers across ops. Revisit at stage
5, when kernel variants exist. `benchmarks/record_graph.py` already gives the
graph offline.
