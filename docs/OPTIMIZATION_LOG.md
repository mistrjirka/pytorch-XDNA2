# Optimization log: what was tried, what was kept, what was rejected

Target: `lab02_cnn/style_transfer.py` (B40, 36x36, MyNet(3,3,5) + VGG19 perceptual loss) on
the XDNA2 NPU, compared against **CPU BF16** (not FP32). Laptop, run-to-run noise ±5-8 %,
so decisions need >= 2 ABBA rounds (`benchmarks/nis_style_transfer.py --quick --rounds 2`).
**Benchmark only on AC power**: on battery the CPU drops to ~1-2 GHz and CPU-bf16 became
842 ms/step instead of ~172 (`benchmarks/results/nis_ab_battery_invalid_2026-10-10.json`).

Evidence column: file in `benchmarks/results/`, test in `tests/`, or a debug flag you can rerun.

## Progress (ms/step, CPU BF16 vs XDNA, AC)

| Stage | CPU | XDNA | Speed-up | Evidence |
|---|---|---|---|---|
| Starting point (old conv path) | ~233 | ~519 | 0.45x | AGENTS.md "Verified" |
| conv3x3w fwd + dX | 171 | ~166 | 1.03x | `nis_conv3x3w_ac.json` |
| + NPU dW (bfp16) | 171.4 | 147.7 | 1.16x | `nis_conv3x3w_dw_ac.json` |
| + cost model, packed-input reuse, fast cat/upsample | 175.6 | 147.2 | 1.19x | `nis_round3_ac.json` |
| + pipelined piecewise dispatch | 171.8 | 128.8 | **1.33x** | `nis_round4_ac.json` |

## Kept (with proof)

| Change | Why it works | Proof |
|---|---|---|
| Whole-array implicit-GEMM window conv (conv3x3w), bf16 x bfp16 weights | Reads each activation once, no im2col; ~1.2 % rel. error vs FP32 | `tests/run_conv3x3w_hw.py`, `tests/run_conv3x3w_torch.py` |
| NPU dW with bfp16 operands (exact bf16 was 63 ms, bfp16 5.6 ms per pass) | AIE2P has no native exact-bf16 matmul (emulated ~16 MAC/cycle) | `tests/run_conv3x3w_dw_hw.py`; 300-step training gate `benchmarks/results/bfp16_gate*.log`: window gaps within ±0.5 % (one 1.4 % outlier) |
| Deferred dW: all NPU dWs run back to back at optimizer time | Avoids a fwd<->dW context switch per layer (~2.6 ms each) | measured with `ctxswitch.py` (scratch); research note `docs/research/xdna2_coherency_ctxswitch_dma.md` |
| Small dWs (< 1 GMAC) on CPU, overlapped with NPU dX | NPU dW dispatch overhead exceeds CPU time | per-pass trace (`XDNA_CONV3X3W_TRACE=1`) |
| Cost model (padded MACs/5.5e9 + 0.15 ms + bytes/20e6 vs real MACs/0.9e9) | RGB 3-channel convs were 5x slower on NPU (padding) | `tests/run_conv3x3w_costmodel.py` |
| dW reuses forward/dX packed inputs | Saves ~6.4 ms/step of repacking | phase profile (`XDNA_PROFILE_MAPPED_CPU=1`) |
| Fast `cat` / `upsample_nearest2d` (+backward) | upsample 5.6 -> 1.2 ms/step | `tests/run_fast_layout_ops.py` |
| Pipelined piecewise dispatch (<= 4 pieces, device sub-buffers, async start) | Host pack of piece i+1 / unpack of i-1 overlap NPU piece i | `nis_round4_ac.json` |
| Dedicated, flushed NPU output buffers | NPU DMA is not cache coherent; dirty CPU lines in recycled buffers were written back over NPU results (1-70 % wrong first dW) | bug reproduced then fixed; research note |
| `posix_spawn` for compiler launches | `fork`/`std::system` marks pinned DMA pages copy-on-write | research note |
| SSE 8x8 transposes in pack/unpack | Fewer instructions; bit-exact | `tests/run_conv3x3w_pack.py` (pack 1.36 -> 1.33 ms standalone, unpack 0.42 -> 0.32 ms; in-situ pack 7.2 -> 6.2 ms/step, unpack unchanged) |
| `XDNA_CONV3X3W_TRACE=1` | Per-conv wait/pack/unpack/sync and per-pass dW times | used for the analysis below |

## Rejected or not worth it (with proof)

| Idea | Verdict | Proof |
|---|---|---|
| Updating mlir-aie / toolchain to dodge the NPU hang | Rejected (user preference); hangs were our design bugs | Root causes below |
| NPU writes channel-major output to make unpack a row copy | Not worth it | Host pack/unpack is memory-bound: `XDNA_HOST_THREADS` 6 vs 10 same, 20 is 2x worse; same bytes move either way; unpack reads cold (post-clflush) DRAM. Standalone unpack 0.3 ms/layer |
| One combined xclbin for fwd + dX + dW to remove context switches (~5 ms/step) | Not feasible | Firmware >= 6.12 switches contexts between any two xclbins (research note). dW needs 6x4 cores with a different dataflow vs the 8x4 forward design; two dataflows cannot share one xclbin and L1 |
| 64-wide-output (Cout <= 64) variant | Not worth it | Would be a separate xclbin -> extra context switches (> savings). dW tile mapping is fixed by column/row broadcast, not changeable via instruction stream. Estimated gain ~4 ms/step. Per-layer trace: conv11 fwd 4.5 ms, VGG 64->64 1.8 ms |
| Stride-2 convs (conv1/conv2) on the NPU via space-to-depth | Not worth it | Embedding 2x2 taps in a 3x3 kernel wastes 5x MACs: ~0.85 ms NPU vs ~1.07 ms CPU, before host overhead |
| RGB / 3-channel convs on NPU | Rejected by cost model | ~7 ms total fwd+dX+dW on CPU, 5x slower on NPU |
| More host threads for pack/unpack | Rejected | `XDNA_HOST_THREADS=20`: 288 ms/step; 6 ≈ 10 |
| `xrt::runlist` batching | Deferred | Open reliability reports (research note); dispatch overhead is ~0.15 ms |
| 10 resnet 9x9 convs: dedicated handling | Break-even only | Trace: 0.49 ms total per op, 0.32 ms NPU wait, ~0.15 ms compute; CPU 0.48 ms |
| Fixed-shape patch conv family (B40 256->128 @ 18/36, `XDNA_ENABLE_PATCH_CONV`) | **Removed** (see `MODES_AUDIT.md`): shadowed by conv3x3w; as the only NPU conv family it ran NIS at 199 ms vs 191 ms with no NPU conv | `ab_patch` runs in `MODES_AUDIT.md` |
| `XDNA_BO_ARENA_MB` sub-buffer arena | **Removed**: 0.09805 vs 0.10194 it/s | `docs/xdna-device-architecture.md` |
| `XDNA_LAZY_DECODER_VALUES` / `_DW` (virtual cat + upsample fed to the next conv) | **Removed**: no gain on `lab02_seg_full` (+2 % / +8 % step time, noise-level), large correctness surface | `MODES_AUDIT.md` |
| `XDNA_DEFER_CPU_DW=1` (CPU dW queued on a worker, held until NPU dW starts) | **Removed** (see `MODES_AUDIT.md`): full-frame ABBA 0.1117 vs 0.1377 it/s, +50 % step time on `lab02_seg_full`; NIS phase timing was bwd 63 -> 56 ms but fwd +3 and opt +1.5 ms, net 0-3 ms, within noise |

## Hang / correctness root causes (do not repeat)

- Chunk-switch hang: IRON `WorkerRuntimeBarrier.wait_for_value` needs a matching `release_with_value`, else later dispatches reuse stale RTPs (header of `conv3x3w.py`).
- Random hangs: shim DMA task queue depth is 4; more queued tasks per channel are silently dropped. One fill per group+column, `slots` <= 4.
- Shim taps must always have 4 entries; shim stride limit 2^20 words; aligned vector loads drop low address bits (`load_unaligned_v`).
- "NPU wedged after reboot" was a stale experimental xclbin used as health check; a TDR hang does not wedge the device.
- Shared packed-input buffer with stale borders across shapes: per-buffer shape tag with re-zeroing.

## Where the ~129 ms goes (AC, `PHASES=1 XDNA_PROFILE_MAPPED_CPU=1 split.py xdna`)

fwd ~49, bwd ~60, opt ~17 ms. NPU conv ops: forward ~24 ms (incl. host work), dX ~22 ms, dW ~12 ms
(per pass 2.2 ms; first pass after a switch ~2.5 ms vs 0.74 ms for the same shape). Context
switches ~5 ms. CPU conv fallbacks ~14 ms (RGB, stride-2, 1x1), elementwise (BN, LeakyReLU, cat,
upsample, mul) ~25-30 ms. Profile is flat; remaining wins are each 1-3 %.

## Open items

- Older NPU conv families still write outputs into recycled allocator memory (same coherency bug risk).
- `w.data.add_` does not bump the version counter (weight cache may go stale; optimizers are fine).
- Global `set_swap_module_params_on_conversion(True)` breaks LSTM `load_state_dict`; LSTM cell unoptimized (0.29x in model zoo).
- NIS venv still imports the old installed package; reinstall needs user approval.
- Re-run `benchmarks/model_zoo.py` for regressions with the new conv paths.

## Generalization work (2026-10-10, branch `conv3x3w-tiling`)

- **Width tiling** (images wider than 36 px): an image is cut into S vertical strips of <= 34 output
  columns, each packed as a virtual image with one real neighbour column per side; only the
  middle columns of each strip are valid. Host pack/unpack only: no new kernel, no new xclbin,
  streams stay keyed `(groups, wp, chunks)`. dW packs dY with zeroed halo columns. Proof:
  `tests/run_conv3x3w_pack.py` (bit-exact vs numpy, incl. group pieces and halo modes),
  `tests/run_conv3x3w_torch.py` (W = 37, 56, 64, 112, 224: fwd/dX/dW ~1 % vs FP32).
- **Shape sweep** (`benchmarks/conv_shape_sweep.py`, `results/conv_shape_sweep*.json`, 3 channel
  configs x 9 sizes x 13 batches, fwd+bwd vs CPU BF16): NPU wins at 31-36 px with batch >= 16
  (1.4-3.0x, 256->128 B100: 2.99x); loses at <= 18 px or batch <= 8. Wide images with 128+
  channels: 1.8-2.7x (48-112 px); 64-channel convs break-even or slower (half of the 128-wide
  output block is wasted). Max error 1.2 %, no hangs. Isolated tests include ~5 ms of dW
  context switching that real models amortize.
- **Cost model fixes**: NPU fixed overhead 0.15 -> 0.5 ms (traces: tiny ops 0.8-1.0 ms), padded rows use
  whole 2048-row groups, break-even ops need `XDNA_CONV3X3W_MARGIN` (0.8) predicted gain: whole-model
  runs showed break-even NPU convs lose ~10 % (resnet18 at 224 px, conv path on: 100-120 ms vs 92 off).
- **Coverage tool** `benchmarks/conv_coverage.py` (no hardware): fraction of conv MACs eligible per
  torchvision model, with and without tiling (VGG11@224 49 % -> 99 %, ResNet18@224 57 % -> 83 %).
- **Not covered**: 1x1 convs (52-91 % of MACs in ResNet50/MobileNet/EfficientNet), depthwise, stride-2,
  7x7. A 1x1 variant needs a tap-count parameter in the core loop and weight DMA (tap offsets are
  already runtime parameters), the most valuable next widening.
- Whole-model caveat: `vgg11_128` is dominated by its 130 M-parameter classifier/optimizer, so it
  is a poor conv benchmark.

## Universality round (2026-10-12): every model at least as fast as CPU BF16

Starting point (`zoo_round3_ac.json`, CPU first then XDNA): geomean **0.68x**, with only NIS
(1.59x) and the transformer (1.09x) faster than CPU, and the MLP at 0.33x and the LSTM at
0.26x. The losses came almost entirely from the host side, not from the NPU.

| Change | Why it works | Proof |
|---|---|---|
| Backward on the calling thread (`torch.autograd.set_multithreading_enabled(False)` in `register_xdna_device`) | PyTorch runs a non-CPU device's backward on its own worker thread with its own OpenMP pool; the forward pool's spinning threads compete with it | `a*2 .sum() .backward()` 131 -> 46 us (CPU 43); MLP 8.0 -> 3.8 ms (CPU 3.8), embedding 11.3 -> 5.9 (5.3), mobilenet 60 -> 49; NIS proxy neutral (95-100 ms both) |
| Step autotuner (`device._StepAutotuner`, global optimizer step hook) | Per-op cost models cannot see per-step costs: NPU dW lives in a second xclbin, 2 x ~2.6 ms switches per step. UNet: 67-74 ms with NPU convs, 56-59 without NPU dW, 51-53 CPU only (CPU 56) | UNet picks CPU only (53.8 ms); NIS picks all NPU (2 of 2 runs, `nis_round5_ac.json` style runs: 122.5 / 123.2 ms vs 125.5 / 127.2 old behaviour) |
| ABBA order inside the tuner | First version measured modes in a fixed order; the mode measured last looked 30 % slower (resnet18 CPU-only 65 ms vs 47-54 in fixed-mode runs) | fixed-mode A/B runs |
| CPU islands for `aten::lstm.input` / `gru.input` (`xdna_train/cpu_islands.py`) | CPU picks the fused oneDNN RNN only for CPU tensors; on XDNA the op decomposed into ~5000 per-timestep ops per step | `tests/run_cpu_islands.py` (bit-identical outputs and all gradients vs CPU); LSTM 48 -> 18.5-20.8 ms (CPU 17.2-18.3) |
| Fused optimizer default (`_prefer_fused_optimizer`) | PyTorch defaults PrivateUse1 optimizers to `foreach` (~10 CPU-fallback list ops per step); the fused kernel runs through the zero-copy fallback | MLP 3.7-4.2 -> 2.9-3.5 ms; transformer fused 75-80 vs 78-100 ms |
| `model_zoo.py --rounds` (ABBA) and 24 warmup steps on both devices | The zoo ran CPU then XDNA, so XDNA always ran on a warmer chip: transformer 0.85x single-order vs 1.01x ABBA | `zoo_round4_ac.json` vs ABBA rerun |

Measured and not adopted:

| Idea | Verdict | Proof |
|---|---|---|
| `OMP_WAIT_POLICY=PASSIVE` instead of the thread fix | Rejected: slows the CPU itself | LSTM CPU 19.7 -> 31.8 ms, `a*2` bwd CPU 43 -> 103 us |
| Better per-op NPU rate in the cost model for 1-chunk (Cin <= 64) convs | Not the UNet problem | standalone NPU rate 3.8 GMAC/ms at 1 chunk vs 4.6-7.7 at 2-4 chunks (`rate.py`, B16 32x32, 10 groups); UNet's dX waited 3.3 ms in situ vs ~0.9 ms standalone = context switch |
| Channels-last for the CPU baseline | Note, not a backend change | CPU BF16 with `channels_last`: UNet 57 -> 49, resnet18 53 -> 47, NIS proxy 155 -> 133 ms. XDNA's CPU conv fallback already gets this (UNet NPU-off 49.6 ms with a contiguous model) |

### What the documentation offers for small shapes (not yet implemented)

NPU side, ranked by expected value:

1. **Reconfigure inside one hardware context** (`aiex.configure` blocks in a full-ELF runtime
   sequence; `mlir-aie/test/npu-xrt/reconfigure_loadpdi`, `..._persistent_memtile`,
   `ctrl_packet_reconfig*`). The fwd/dX and dW designs could live in one ELF and switch with a PDI
   load in the instruction stream instead of a firmware context switch (2.6 ms). The cost of
   that PDI load is unmeasured. It needs a NOOP microbenchmark before porting the conv runtime to the
   full-ELF flow (open bugs mlir-aie#3654/#3656, research note). It would remove the main reason
   the tuner rejects NPU dW for mid-sized models.
2. **`xrt::runlist` / command chains** (`test/npu-xrt/add_one_two_runlist`, driver
   `aie2_cmdlist_multi_execbuf`). It submits several dispatches with one ioctl and one wait. The
   fixed ~0.5 ms per op makes 9x9 / 4x4 convs break-even; chaining the convs of a block would
   amortize it.
3. **Small spatial sizes: im2col GEMM instead of the halo grid.** A 4x4 image becomes a 6x6
   padded grid (2.25x MACs) and still occupies a whole 2048-row group. Host im2col
   (K = 9*Cin) has no halo waste, and batch*H*W rows fill the 2048-row groups densely.
4. **1x1 convs** are plain GEMMs (52-91 % of MACs in ResNet50 / MobileNet / EfficientNet): a
   tap-count parameter in conv3x3w (tap offsets are already runtime parameters).

CPU side: CPU kernels that PyTorch selects by device type are reachable through CPU islands
(the LSTM pattern) or through zero-copy fallback with the right flag (fused optimizers).
Candidates still on the slow path: `nn.MultiheadAttention`'s fused CPU path
(`_native_multi_head_attention` only for CPU in eval/no-grad), `embedding_bag`.

**Result** (`benchmarks/results/zoo_round5_ac.json`, ABBA, 2 rounds, AC): geomean **1.24x**, every model
>= 1.04x: resnet18 1.14, mobilenet_v3 1.04, unet 1.13, transformer 1.05, mlp 1.16, lstm 1.21,
conv1d_audio 1.18, conv_autoencoder 1.20, embedding 1.78, nis_mynet 1.72. First losses are identical
to CPU. Last losses can drift slightly because the fused optimizer rounds differently in bf16 (NIS proxy after 34
steps: 0.5195 CPU vs 0.5625 XDNA, within the zoo's 10 % check). The real NIS script: 122.5-123.2 ms/step
vs CPU BF16 184 ms.

## 2026-10-12: modes cleanup (removals)

Removed after the audit in `docs/MODES_AUDIT.md`; the rows above keep their history.
Each removal is its own commit on the pruning branch (`9e1a5de..HEAD`):

- Fixed-shape patch conv family (`XDNA_ENABLE_PATCH_CONV`, `XDNA_PATCH_CONV_ARTIFACT_DIR`, `XDNA_CONV_OUTPUT_NCHW`); the `conv3x3_patch40_128` artifacts.
- Full-frame NPU dW experiment (`XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_DW`, `XDNA_FULLFRAME_DW_ARTIFACT_DIR`); no artifact existed.
- Deferred CPU dW worker queue (`XDNA_DEFER_CPU_DW`, `XDNA_DEFER_CPU_DW_WINDOW`, `XDNA_HOLD_CPU_DW`) and `AsyncHostWorkQueue`.
- BO sub-buffer arena and tensor BO flag (`XDNA_BO_ARENA_MB`, `XDNA_TENSOR_BO_FLAG`).
- Lazy decoder values (`XDNA_LAZY_DECODER_VALUES`, `XDNA_LAZY_DECODER_DW`), virtual values, fused cat2+upsample2 forward; their two tests.
- Shadowed legacy full-frame streams (res146 / dX292 / dX584 of the legacy family), `configure_fast_dx584` and the `XDNA_ENABLE_FAST_DX584` / `XDNA_FAST_DX584_ARTIFACT_DIR` aliases; the `XDNA_FULLFRAME_TM` environment channel (tile M is now passed as an argument).
- Unused C pack entry points in `conv_pack.cpp`.

## 2026-10-13: early start of the NPU weight gradients

Timeline. In the profiled NIS step (126 ms) the last NPU dX call ends at 81 ms. The
CPU then works alone for 31 ms (five 9x9 residual blocks and the encoder convolutions)
while the NPU idles. The six dW passes (11.9 ms) started only at the optimizer, at 112 ms.

Change. The NPU dW passes are still deferred, but they can now start earlier. Once two
training steps show the same number K of NPU dX calls in backward, a helper thread
starts the deferred dW passes right after the K-th dX call of the following steps. They
run while the CPU finishes backward. A reader still waits for its gradient, and runs the
job itself if the helper has not started it. The first two steps, and models whose dX
count per step is unstable, keep the lazy behaviour. `XDNA_ASYNC_DW=0` turns the feature
off. The default is on.

Measurement. Ryzen AI 9 365 on AC power. NIS style-transfer training step: MyNet(3,3,5),
batch 40, 36x36, BF16, Adam, 10 CPU threads. Four runs per setting, in ABBA order, on the
same tree. Both settings give the same loss (22.9).

| Setting | Run 1 | Run 2 | Run 3 | Run 4 | Mean |
|---|---|---|---|---|---|
| `XDNA_ASYNC_DW=0` (ms) | 124.3 | 120.3 | 119.9 | 125.7 | 122.6 |
| `XDNA_ASYNC_DW=1` (ms) | 113.3 | 108.9 | 110.6 | 109.3 | 110.5 |

The step is about 10 percent faster with the feature on.

Limits. The feature helps only when backward has CPU work after the last NPU dX call.
If the learned K is wrong, because the structure changed, the early start can overlap
with later NPU dX calls. That costs hardware-context switches, but the results stay
identical. The test is `tests/run_async_dw.py`. With gradient accumulation or several
optimizers, the dX count may never stabilise, and the feature then stays lazy. A process
that exits while a started dW is running is not handled specially. That window is a few
milliseconds after backward.

Candidates not done. Running the target-image VGG forward during the forward NPU waits
can save at most 5.6 ms, and only while the host is blocked. Overlapping the target and
prediction VGG passes gains about 0, because both use the same CPU cores. Both were
measured as small or zero, so they were not implemented.
