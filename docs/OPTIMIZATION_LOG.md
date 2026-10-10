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
| `XDNA_DEFER_CPU_DW=1` as default (CPU dW held until NPU dW starts) | **Unproven, opt-in** | Phase timing: bwd 63 -> 56 ms but fwd +3 and opt +1.5 ms, net 0-3 ms, within noise. Results identical (`gradchk`: loss 149.0, grad sum equal). Needs a 2-round AC ABBA |

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

- Re-run the A/B for `XDNA_DEFER_CPU_DW=1` on AC power (>= 2 ABBA rounds).
- Older NPU conv families still write outputs into recycled allocator memory (same coherency bug risk).
- `w.data.add_` does not bump the version counter (weight cache may go stale; optimizers are fine).
- Global `set_swap_module_params_on_conversion(True)` breaks LSTM `load_state_dict`; LSTM cell unoptimized (0.29x in model zoo).
- NIS venv still imports the old installed package; reinstall needs user approval.
- Re-run `benchmarks/model_zoo.py` for regressions with the new conv paths.
