# conv3x3w with bfp16 activations (branch `bfp-input`, 2026-10-13)

Status: correct and tested, kernel 1.8x faster, **no end-to-end gain yet** on the
NIS training step. Not for main until the host side catches up.

## What changed

The forward/dX conv3x3w kernel used to receive bf16 activations and convert each
8x8 block to bfp16ebs8 inside the inner loop, 8 times per block per tap (once per
pair of output-channel blocks). Now the host converts the activations to bfp16
once while packing (`xdna_conv3x3w_pack_input_strips_range_bfp16`, AVX-512 with a
scalar fallback, bit-identical to `conv3x3w_host.bfp16_encode`), and the kernel only
does block loads and `mac_8x8_8x8T`.

- Layout: `[groups][chunks][8 channel blocks][2128 rows][9 B]`, one block = one
  pixel's 8 channels. A tap is a byte offset of `9 * (dy*(W+2)+dx)`.
- Hardware rule found on the way: a bfp16 load stream holds two 64-byte lines, so
  a pair of 72-byte pops is exact only when it starts at most 48 bytes into its
  line (a start at 56..63 loses the last block). The core keeps a second copy of
  its window shifted by 16 bytes (`prep_window`) and reads late-starting pairs
  from it. See the comment in `conv3x3w_kernel.cc`.
- The deferred dW kernel still uses the bf16 layout. When dW will run on the NPU,
  the forward/dX packing writes that copy in the same pass.
- `matmul_window` must set `conv_even` rounding: without it, the 36 bf16 stores of
  the running sum truncate and the error grows with the number of taps (6.4% instead
  of 1.2%).

## Measurements (Ryzen AI 9 365, AC)

Kernel only, B40 36x36 Cin256 Cout128 (g29_w38_c4), median of 27 dispatches:

| Variant | ms |
|---|---|
| old kernel (bf16 in, in-kernel conversion) | 3.11 |
| old, conversions removed (wrong results, diagnostic) | 1.91 |
| old, data movement only | 1.63 |
| **new kernel** | **1.74** |
| new, without the shifted copy (wrong results, diagnostic) | 1.58 |
| new, data movement only | 1.37 (~46 GB/s, near the DRAM ceiling) |

Accuracy vs FP32 (`tests/run_conv3x3w_hw.py`): 1.24% rel. RMS, same as before;
emulated bfp16 x bfp16 math gives 1.03%. `tests/run_conv3x3w_torch.py`: all shapes
OK (fwd 0.7-1.2%, dx 1.0-1.2%, dw 0.9%).

NIS training step (`profile_training.py`, ABBA, 4 rounds each): old 111.4 ms, new
111.5 ms; identical loss. Per-call trace (`XDNA_CONV3X3W_TRACE=1`): the host now
waits only 0.1-0.4 ms per conv for the NPU (was 0.8-3.5), and fwd+dX calls take
~3.5 ms less per step, but packing/unpacking on the CPU is ~15 ms per step and the
rest of the step is CPU work (VGG loss, BN, activations, CPU convs).

## Open questions

1. Host pack/unpack dominates the NPU calls now: unpack of dX at B40 36x36 is ~2.1 ms,
   forward pack ~2.4-2.8 ms (bfp + the dW bf16 copy). Can consecutive NPU convs skip
   the NCHW round trip, or can pack/unpack move off the critical path?
2. The cost model (`XDNA_CONV3X3W_NPU_GMACS`, 5.5) still describes the old kernel;
   the new one does ~10 GMAC/ms on large shapes. Retune it, then re-check which
   convs move to the NPU (VGG features 0..5, ResNet blocks at 9x9).
3. Input DMA: each core fetches its own 144-row window (2.25x the 64 rows it
   computes). Sharing the halo through the memtile, and keeping the weights in the
   memtile instead of re-reading them per group, would cut DRAM traffic ~40%.
4. `prep_window` costs ~0.16 ms per pass; the memtile could deliver the shifted
   copy instead.

## Follow-up after the external audit (docs/research/bfp-input-notes, 2026-10-13)

Applied:
- Non-finite inputs: the scalar, AVX-512 and numpy encoders now agree (Inf
  saturates to 127/-128, NaN encodes as 0). Before, scalar and AVX-512 produced
  different bytes. A NaN or Inf therefore does not propagate through an NPU conv.
- The shifted window copy is placed from each buffer's own address
  (`shift_base`), instead of a pointer difference between two buffers that
  assumed equal placement modulo 64. The row-pair addresses are computed once
  per row pair, outside the output-channel loop.
- The dW copy of the input is written with streaming stores (full aligned rows)
  and synced when the dW runs (`Conv3x3wPacked::sync_pending`), not at the end
  of the forward/dX call.

Measured (NIS step, ABBA x4): old 108.2, new 109.4 (one 117 ms outlier, median
107.5), new with `XDNA_CONV3X3W_NPU_GMACS=10` 104.9 ms. Rate 10 won every round
there. But in the model zoo (XDNA only, 2 rounds each) it made `nis_mynet`
(MyNet without the VGG loss) ~12% slower (76.1/79.4 -> 85.6/89.0 ms), and other
rows were mixed or noise. So the default stays 5.5. The cost model needs per-shape
NPU times (small 9x9 convs are overhead bound), not one higher rate.
