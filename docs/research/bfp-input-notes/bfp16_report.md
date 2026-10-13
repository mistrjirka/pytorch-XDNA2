# bfp16ebs8 activations for conv3x3w: layout, existing kernels, host cost, numerics, output

Read-only study. Nothing run on the NPU. CPU measurements are marked [M]. Facts from code/docs are marked [F]. Inferences are marked [I]. Paths are under the project roots named below.

Abbreviations: KD = `/home/jirka/programy/torch-xdna2-async-dw/xdna_train/_designs/`, AIE = `/home/jirka/.cache/uv/archive-v0/iObbsIgTNTG5hg7S/mlir_aie/include/aie_api/`.

## 0. Summary

- [F] Forward taps never change the channel grouping. A bfp16 block = (one pixel, 8 consecutive channels), and a tap shifts only the pixel index. One encoding of X therefore serves all 9 taps.
- [F] The kernel converts 4608 8x8 tiles per 64-channel chunk per core (9 taps x 8 z x 8 i x 8 j-pairs). The window has only 142 tiles (142 rows x 8 channel blocks / 8 rows per tile). Converting the window once is about 32x less conversion work.
- [F] The AIE API gives byte-granular block streams (`seek_byte`, `pop_seek_byte`). So an arbitrary 9-byte-block start is expressible, with no 64-byte alignment needed. Its cycle cost and minimum alignment are not documented; that needs an NPU micro-benchmark later.
- [F] No existing kernel in the trees does tap-shifted reads of bfp operands. The local kernel that caches converted activations (dense pipeline) was measured slower.
- [M] Scalar C on one Zen 5 core converts bf16 to bfp16ebs8 at 3.0 GB/s of bf16 input. numpy is 0.2-0.3 GB/s.
- [F] Activations are already rounded to bfp16 (RNE) inside the kernel, so a host or per-window encode gives the same numbers if the hardware converter matches the host rule.
- [I] Recommended: convert each window once on the core, in place, into bfp16 blocks (proposal P1b below). The DMA and host stay unchanged. Expected kernel time goes from 3.1 ms toward the measured 1.9 ms floor, if the stream seeks are cheap.

## 1. Layout and unaligned block access

Current layout [F]:
- The window is bf16 `[8 ch-blocks][142 rows][8 ch]` (`conv3x3w_kernel.cc:20-27`, `conv3x3w.py:105-106,123`). Its size is 142 x 64 x 2 = 18,176 B.
- A tap offset is a row offset `tap_rows = dy*(W+2)+dx`, applied as a pointer shift (`conv3x3w_kernel.cc:135-136`). A pixel is 16 B, so the offset is a 16 B multiple. Unaligned loads are used (`:55-64`).
- Each `A00 = load_unaligned_v<64>(p, 8)` returns 8 rows x 8 channels of one channel block. Its 8 blocks, one per row, are converted by `::to_v64bfp16ebs8` (`:61-62,69-70,77-78,85-86`). So a block is (pixel, 8 channels), as the question assumes.

Why the block boundaries never move [F]: a tap changes only the pixel index. The 8-channel group stays fixed, so one encoding serves every tap.

Block-stream API [F]:
- `block_vector_input_buffer_stream<bfp16ebs8,64>` is the only way to read a v64bfp16ebs8. The docs say block vectors cannot be loaded with the standard load APIs (`AIE/aie_doc.hpp:944-946`, footnote at `:256`).
- `aie_doc.hpp:952-955` says the FIFO interface allows "larger, unaligned pops".
- `seek(n)` moves by n x 72 B. `seek_byte(n)` moves by n bytes (`AIE/detail/aie2p/array_helpers.hpp:162-205`). `pop_seek_byte(n)` pops and then moves by n bytes in one operation (`:255`). Each seek resets the FIFO state (`:162-170`), so the next pop refills.
- The intrinsics include a `_1d_byte` form with a byte offset for bfp16 (`llvm-aie/.../clang/21/include/aie2p/aie2p_ldst.h:425-434`). The pointer type is `v64bfp16ebs8_unaligned` (`AIE/detail/aie2p/block_vector_native_types.hpp:30`).
- Block size: 9 B per 8 values (`aie_doc.hpp:232`). A v64 is 72 B, which is a multiple of 4 B, so DMA can move it.
- Not documented in these headers: the minimum byte granularity of a stream start, and the cycle cost of a seek or refill. Those need an NPU test.

Layout proposals, L1 per core (64 KB). Baseline [F]: A bf16 18,176 + B 2 x 9,216 (`b_l1_depth=2`) + C 16,384 + stack 3,328 = 56,320 B.

| Proposal | What it does | L1 per core | Conversions per chunk | Input DMA | Verdict |
|---|---|---|---|---|---|
| P1b. Convert window in place on core | Keep the bf16 DMA. After the window arrives, rewrite it in place as `[i][row][9 B]`. Write address 9b trails read address 16b, so an in-place forward pass is safe. Taps read via `seek_byte(9*(i*142+tap_rows+8z))`, then sequential pops over z (72 B stride). | 56,320 (window 10,224 used inside the 18,176 buffer) | 142 (was 4608) | unchanged, 16 B per 8 values | Recommended |
| P1. Host pre-encodes window | Host writes `[i][row][9 B]` per chunk. DMA moves 9 B blocks as 72 B runs. | 48,368 (A = 10,224); depth 2 fits at 58,592 | 0 | -44% (9/16) | Second step, after P1b works. Adds a host encode per layer (section 3). |
| P2. Three dx-shifted copies, row stride padded to a multiple of 16 px | Makes dy shifts 16 B aligned (9 x 16 = 144 B). | 30,672 resident, or ~58.6 KB with a 2-deep ring | 142 x 3 | 3 x 1.125 B per value vs 2 B now (worse) | Reject. |
| P4. Per-tap scratch of converted tiles | Convert the 64 tiles of each tap once into a 4,608 B scratch, reuse across 8 j-pairs. Avoids unaligned bfp reads. | 60,928 (single buffer) | 576 (8x fewer) | unchanged | Fallback only. A similar 8x-fewer-conversion variant was measured slower (section 2). |
| P3. Pad each block to 16 B | Aligns every block. | 18,176 | 0 | unchanged | Not usable: the stream pops a contiguous 72 B v64 and needs the 9 B packing. |

## 2. Existing kernels with bfp16 operands

- [F] `/home/jirka/programy/mlir-aie/aie_kernels/aie2p/mm_bfp.cc:76-146`: pure bfp16 x bfp16. A and B are both pre-encoded, read as tile-major streams (`seek(z*colA)`, lines 104-113), and the output is bfp. Its `matmul_vectorized_2x2_bfp16` is the closest match to the target MAC loop. It has no tap offsets and no measured speed in the repo.
- [F] `/home/jirka/programy/mlir-aie/aie_kernels/aie2p/mm_bfp_mixed.cc:75-96`: bf16 A converted per use. A1 goes through `mul_elem_64` as a second conversion path (`:84-85`).
- [F] `/home/jirka/programy/llama.cpp-xdna2-bfp/tools/xdna/nanbeige-resident/mm_bfp_single_preencoded.cc:8-44`: encodes 8 K8 blocks once into a local scratch area, then both operands stream. This is the local-cache pattern.
- [F] `.../nanbeige-resident/mm_bfp_small_m.cc`: converts A inside the k-loop (`::to_v64bfp16ebs8(av)`). Mixed.
- [F] `/home/jirka/programy/xdna-atb-peano/mm_atb_peano_preconvert_generic.cc` and `mm_bfp_mixed_exact_q0.cc`: B is bfp, A is bf16 and converted in the kernel (mixed).
- [F] Many files in `xdna-superoverlay-probe` use `bfp16ebs8*` as an opaque carrier for bf16 bits (for example `stream_attention_online_generic.cc:8`, `as_bf16`). They do no block arithmetic.
- [F] Peak probe `/home/jirka/programy/xdna-tops-20260920/peak-initial.cc:19-27` (PEAK_MODE 1) converts operands before the loop and keeps them in registers. Measured on 32 cores: 54.85-58.16 TOPS for preconverted bfp (`peak-measured-1-v2.json`). The bf16 matrix API control gives 1.18 TOPS (`peak-measured-2.json`). These are register-resident numbers, not streamed GEMM.
- [F] `XDNA2_TOPS_AND_EXECUTION_PROFILE.md:32-37`: the packed path needs operands in grouped shared-exponent form before the inner loop.
- [F] `XDNA2_DENSE_PIPELINE_RESULT.md:66`: a prototype cached converted activation tiles. It cut conversions from 256 to 32, but was slower: 16.60 ms vs 13.73-14.15 ms. The extra local writes and reads cost more than the conversions saved. P4 is the same kind of design, so treat it with caution.
- [F] `XDNA2_DENSE_PIPELINE_RESULT.md:70`: a three-stream variant failed code generation with an "alignment-immediate" error. Expect compiler constraints on stream seeks.
- [F] `XDNA2_BFP_BUG_AND_GPU_PLAN.md:7,33`: a BFP context kernel gives wrong results under `-O2` pipelining. The local workaround disables the pipeliner for that kernel. Re-verify any restructured bfp kernel under the same flags.
- [F] `XDNA2_NANBEIGE_RESEARCH_NOTES.md:637`: AMD's ATB config2 (pure BFP16) is quoted at 31.3 TFLOPS on Strix. Not reproduced here.
- [I] No kernel in these trees reads bfp operands at tap-shifted offsets. conv3x3w would be the first.

## 3. Host conversion cost

Rounding rule [F]:
- `conv3x3w_host.py:57-68` (`bfp16_encode`): blocks of 8 consecutive values. Exponent byte = max biased exponent of the block. lsb = 2^(E-133). Mantissa = RNE(x/lsb), clipped to [-128, 127]. Byte layout is exponent first, then 8 int8 mantissas.
- `torch-xdna2-auto-setup/benchmarks/bfp16_training_gate.py:29-40` (`bfp_quant`) uses the same rule, written as floor(log2 max) - 6.
- The docstring at `conv3x3w_host.py:58-60` says AMD's helper truncates.
- The kernel sets `aie::swap_rounding(conv_even)` (`conv3x3w_kernel.cc:132`), so its activation conversion is RNE if the hardware honours the rounding mode.
- [I] The weights use `bfp16_encode` and the tests compare it with the AMD helper, so the hardware exponent and mantissa layout are believed to match. Bit-exactness of `to_v64bfp16ebs8` against `bfp16_encode` has not been checked on NPU. That needs one test (encode a known tile both ways).

Speed [M] on this machine (Ryzen AI 9 365, AVX-512, 20 threads):
- `bfp16_encode` as written (float64 path): 0.24 GB/s of bf16 input, 1 thread.
- Vectorized numpy in float32 (same results, checked exact on 65,536 values): 0.33 GB/s, 1 thread. Too slow to use.
- Scalar C built with `gcc -O3 -march=znver5` (exponent by bit max, RNE via `rintf`): 3.0 GB/s of bf16 input (1.5 Gval/s), 1 thread. Benchmark binary in the scratchpad only.
- [I] A hand-written AVX-512 version should reach several GB/s per core. Across 20 threads it should be memory-bound, roughly 20-40 GB/s. Not measured.
- Per layer, for a 1x64x128x128 input (padded 130x130 grid, 1.08 M values, 2.2 MB bf16): about 0.7 ms on one core, about 0.1 ms on 8 threads [I].
- [F] Earlier repo measurements: CPU Q4_1 to BFP converter, 20 threads, about 0.67 GB/s of source bytes (`XDNA2_NANBEIGE_RESEARCH_NOTES.md:629-631`). Not the same operation.

## 4. Numerics

- [F] Today the kernel converts bf16 window values (exact bf16, loaded into fp32 accumulators) with RNE. Host encoding of the same values with the same grouping gives identical inputs to the MAC. So pre-conversion (P1) and per-window conversion (P1b) are numerically equivalent to today, provided the hardware rule matches `bfp16_encode`.
- [F] The grouping must match. Forward blocks are (pixel, 8 consecutive channels). The host's flat NHWC rows (`conv3x3w_host.py:32-46`) already group that way. The L1 block order `[i][row]` is a permutation of whole 9 B blocks, so the host or the in-place pass must place blocks, not values.
- [M] Error on Gaussian data (`bfp16ebs8` RNE vs bf16 input): 0.67% relative RMS. Truncation: 1.06%. bf16 alone: 0.17%. For a K=576 dot product with bfp activations, the error is 0.67% relative RMS (`/tmp` script, not saved).
- [F] Forward running sum is bf16: `matmul_2x2_window` reloads C as bf16 and stores bf16 after every tap call (`conv3x3w_kernel.cc:48-51,105-108`). That is 9 x chunks roundings per output.
- [M] Rounding error of that running sum vs fp32 accumulation: 0.37% relative RMS for 9 partials, 0.72% for 36. This is about the same size as the bfp activation error, so it is worth fixing too. The dW kernel keeps fp32 for this reason (`conv3x3w_dw_kernel.cc:5-6`).
- [F] dX: no dX kernel is in `_designs`. [I] It is probably the same conv3x3w family with dY as activation and flipped weights. Its forward-like grouping over channels means the same encoding applies.
- [F] dW (`conv3x3w_dw_kernel.cc:33-54,67-68`) groups bfp blocks along pixels (`transpose(...,8,8)`). A one-pixel shift changes block membership, so X needs one encoding per dx. Pre-encoding X would cost 3 x 1.125 B per value of DMA, versus 2 B today. The same idea does not transfer cleanly to dW. dY is converted once per call (32 tiles), so little is gained there.

## 5. Writing the output in bfp

- [F] Output is a bf16 C tile in L1, `C_l1_ty = (64,128)` = 16,384 B (`conv3x3w.py:109,144`), stored with `to_vector<bfloat16>` (`conv3x3w_kernel.cc:105-108`).
- [F] An accumulator 8 pixels x 8 output channels, converted with `to_v64bfp16ebs8`, gives blocks of (pixel, 8 output channels). That matches the next layer's input grouping. A 64 x 128 bfp tile is 9,216 B, which frees 7 KB.
- [I] Blocks of 72 B (8 pixels x 8 channels) are DMA-able, since 72 is a multiple of 4.
- What breaks:
  1. Bias, BN and activation must run on fp32 accumulators before conversion. Per-output-channel scale and bias are lane-wise and cheap. Training-mode BN needs batch statistics over all pixels first, so it needs a separate reduction or two passes.
  2. The padded border must be explicitly zeroed after bias, or the next layer sees garbage.
  3. Residual and skip paths in bfp: quantization error 0.67% vs 0.17% for bf16 (section 4).
  4. Anything on the CPU (PyTorch loss, pooling, norms) needs bf16. Decoding bfp to bf16 costs a pass of 9 B to 16 B per block [I].
  5. Forward output already has the bf16 running-sum error (section 4).
- Chaining layers in bfp avoids host conversion between them, and is the largest saving (P1b plus bfp output).

## Recommended design

1. Phase 1 (no host change, no DMA change): P1b. In the matmul, after the window is acquired, convert the bf16 window in place to `[8 i][142 rows][9 B]` using 142 `to_v64bfp16ebs8` calls per chunk. Write address 9b trails read address 16b, so this is a single forward pass with no overlap hazard. Read taps with `block_vector_input_buffer_stream<bfp16ebs8,64>`: `seek_byte(9*(i*142+tap_rows+8*z))` at the start of each (i, tap), then pop sequentially over z (72 B stride). Keep `AIE_PREPARE_FOR_PIPELINING` only after checking the pipeliner bug (section 2).
   - Expected: conversions drop from 4608 to 142 per chunk per core. Kernel time about 1.9 ms plus about 0.04 ms residual [I], assuming seeks are cheap.
   - Unknown, must measure on NPU when free: minimum byte alignment of the stream start, seek and refill cost, and whether `pop_seek_byte` is a single instruction.
2. Phase 2 (only if phase 1 works and input DMA is the bottleneck): P1. Host encodes each window with `bfp16_encode` on the flat rows and permutes 9 B blocks to `[i][row]`. Input DMA falls 44%. Host encode costs about 0.1 ms per layer on 8 threads in C [I].
3. Phase 3: bfp output with a fused fp32 epilogue (bias, activation, zero border), fed straight into the next layer's phase-1 window conversion or a P1-style layout. This removes host conversion between layers. Keep the skip path in bf16 if it is needed.
4. Fallback if byte-offset stream starts are unusable: P4 (8x fewer conversions). Prior measurement of a similar design was slower, so measure before adopting.
5. dW: keep the current in-kernel conversion. Pre-encoding X needs three encodings and more DMA.
6. Independent of bfp: fix the forward bf16 running sum, for example by keeping an fp32 C buffer (32 KB, which does not fit with the current L1 budget) or by reducing the number of bf16 roundings.

Expected saving [I]: phase 1 removes about 97% of the in-kernel conversion work (3.1 ms to about 1.95 ms kernel time, if seeks are cheap). Phase 2 cuts input DMA by 44%. Phase 3 removes host conversion between layers.

Open items needing NPU time (not run here): byte-granular stream starts and their cost; one bit-exact test of `to_v64bfp16ebs8` against `bfp16_encode` on a known tile under `conv_even`; the `-O2` pipeliner check on any restructured kernel.
