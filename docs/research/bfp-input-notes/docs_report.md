# AIE2P (XDNA2 / npu2) bfp16 mac_8x8_8x8T research report

Scope: local docs and headers only. Nothing was run on the NPU. "[inferred]" marks derivations, not documented facts.
Abbreviations: ATB = `programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/`; HDR = `/home/jirka/.cache/uv/archive-v0/iObbsIgTNTG5hg7S/mlir_aie/include/` (the other archive `ZeAsGIM0078C2ZOD` has the same bfp headers). MLIR = `/home/jirka/programy/mlir-aie/`. PROJ = `/home/jirka/programy/torch-xdna2-async-dw/`.

## Summary of leads (most actionable first)

1. Pipelining macros are probably no-ops under Peano. `aie_kernels/aie_kernel_utils.h` only defines `AIE_PREPARE_FOR_PIPELINING` / `AIE_LOOP_MIN_ITERATION_COUNT` for `__chess__` and `__AIECC__`. The Peano compile flags in the installed `python/aie/utils/compile/utils.py` (~L320-345) add only `-DNDEBUG` and `-D__AIE_API_AIE_ADF_HPP__`. No file in the tree defines `__AIECC__`. [inferred] Your kernel (`use_chess=False` in `conv3x3w.py`) gets an empty macro, so the loops are not pipelined.
2. Your inner K loop has 2 trips (K_STEP=64 gives colA=8, `i += 4`), and each (z,j) block does 4 bf16 C loads and 4 stores around only 16 mac instructions. C is round-tripped through bf16 once per matmul call, and there are 9*chunks calls per output group (`conv3x3w.py` L175-181). [inferred] This overhead, not the mac itself, probably dominates. The ATB reference has the same structure but reaches higher throughput.
3. Native bf16 mmul does not need conversion. Default (non-emulated) bf16 uses `mac_8x8_8x4_bf16` and `mac_4x8_8x8_bf16`, each 256 MAC per instruction (half of bfp's 512). [inferred] If bf16 issues at ~1 instruction/cycle, that is 256 MAC/cycle against your measured ~102 MAC/cycle. This needs B unpacked to bf16 (2x L1 footprint), so it changes the weight format.
4. Your conv3x3w_dw_kernel.cc:33-35 and OPTIMIZATION_LOG.md:26 say "AIE2P has no native exact-bf16 matmul". The headers show a default native bf16 path. Worth re-checking that claim.

## 1. Peak throughput

- One `mac_8x8_8x8T` = 8x8x8 = 512 MAC. Peak is 512 MAC/cycle/core if it issues every cycle. Sources: `conv3x3w_dw_kernel.cc:35` ("bfp16ebs8 blocks run at 512"); bf16 exact path "~16 MAC/cycle" at `OPTIMIZATION_LOG.md:26`. The 1 instr/cycle issue rate is your own assumption, not in the docs.
- bf16 default native mmul (HDR `aie_api/detail/aie2p/mmul_bf16_bf16.hpp`): 8x8x4 = 256 MAC (L21-41, `mac_8x8_8x4_bf16`); 4x8x8 = 256 MAC (L158-177, `mac_4x8_8x8_bf16`); 4x8x4 = 128 MAC (L43). The 8x8x8 default (L136-156) is two `mac_4x8_8x8_bf16` calls. The 8x8x8 bf16 "emulated" form (L63-133) is only compiled with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, which is not defined by default. `python/iron/kernels/linalg.py:84-92` and `aie_kernels/aie2p/mm.cc:~290` agree.
- Clock: not documented in any local file (grepped mlir-aie docs/programming_guide and PROJ docs). [inferred] Two independent estimates:
  - Marketing "50 TOPS" INT8 (quoted in `/home/jirka/programy/XDNA2_NANBEIGE_RESEARCH_NOTES.md` ~L147) over 32 cores at 512 MAC/cycle gives ~1.5 GHz.
  - Your 5 cycles/mac and the cost-model rate 5.5 GMAC/ms (`OPTIMIZATION_LOG.md:29`, = 5.5 TMAC/s whole array) give 5.5e12 / (32 * 102) ≈ 1.7-1.8 GHz. Consistent with your 19% figure only if nominal peak is ~29 TMAC/s.
- Whole-array bfp16 GEMM, documented (ATB `README.md`, "Performance", Strix Point NPU, 32 cores, turbo mode, min over iterations):
  - config1 bf16 A x bfp16 B (tile 128x64x128, rho 4): 24.3 TFLOPS (Krackan 27.33)
  - config2 pure bfp16 A and B (192x128x96, rho 6): 31.3 TFLOPS (Krackan 31.58)
  - config3 pure bfp16 with bf16 C (128x64x128, rho 4, same shape as config1): 28.5 TFLOPS
  - Per core: 24.3 TFLOPS = 12.2 TMAC/s / 32 = 380 GMAC/s per core. [inferred] At 1.77 GHz that is ~215 MAC/cycle (~42% of 512), i.e. ~2.4 cycles per 512-MAC instruction, vs your 5.
- `block_datatypes/matrix_multiplication/README.md` (Performance table, 8 cols, self-described as not representative): bfp16ebs8 in/out 10.2 TFLOPS; bf16 with bfp16 emulation 4.8; bf16 native 1.6.
- Your own cost model (`OPTIMIZATION_LOG.md:29, 124`): 3.8-7.7 GMAC/ms whole array for NPU convs.

## 2. bf16 -> bfp16 conversion

- The only conversion path is accumulator to block. `to_v64bfp16ebs8(accum<accfloat,64>)` (HDR `detail/aie2p/mmul_bfp16_bfp16.hpp` L68-95, `mmul_bf16_bf16.hpp` L99-131); or `acc.to_vector<bfp16ebs8>()` (`aie_kernels/aie2p/mm_bfp_mixed.cc`, ~L60-100). The bf16 vector must first be upshifted into an accumulator (`accum = vec`). There is no direct vector->bfp intrinsic in aie_api.
- No native bf16 x bfp16 mixed mac. The bf16 A operand is always converted with `to_v64bfp16ebs8` before `mac_8x8_8x8T` (`mmul_bfp16_bfp16.hpp` L81, L90). The accumulator table in `aie_doc.hpp` ~L1029-1032 lists bfloat16 x bfp16ebs8 with accfloat, which is the accumulator type, not a separate mac.
- `mm_bfp_mixed.cc` (aie2p) converts A1 with `mul_elem_64(A1, ones)` instead of the plain `accA1 = A1` upshift used for A0. [inferred] This spreads work across two units. Not documented.
- Cost: no cycle count in any local header or doc. The `block_datatypes/matrix_multiplication/README.md` "Impact of conversion unit" section says accumulator conversions are "inexpensive from the kernel perspective" (its plot `res/conversion_unit.png` has no readable cycle data).
- Measured proxy (ATB, same 128x64x128 tile and rho 4): mixed bf16 A (config1, 24.3 TFLOPS) vs pure bfp A (config3, 28.5). Conversion costs about 15% of whole-array throughput. [inferred for the cause; shapes and C type differ slightly.]
- Each A block is converted once and used for 2 B blocks (2x2 blocking). Your kernel and ATB both amortize that way.

## 3. Register files

- Local docs describe AIE-ML/XDNA1 only, not AIE2P: `programming_guide/section-4/section-4c/README.md`, "The Vector Unit - Loads" (12 x 512-bit vector registers, 2 load units of 256 b/cycle, load from different banks to avoid conflicts) and "Multiply and Add" (9 x 512-bit accumulator registers; "1 VMAC per cycle" ideal).
- No AIE2P register count is in the headers (checked `aie_api/detail/aie2p/*`, `aie_doc.hpp`).
- Conflict: the ATB and your kernel hold 4 x `accum<accfloat,64>` (256 fp32 lanes = 8192 bits, which is 16 x 512-bit). That does not fit 9 registers, so AIE2P accumulator state must be larger than the AIE-ML doc says. [inferred]
- Register-blocking options (2x2 is proven by ATB and your kernel; 4x2 and 4x4 are not). Size per 8x8 accfloat C block: 64 x 32 bit = 2048 bits (4 x 512-bit). A 2x2 block of C = 16 x 512-bit equivalents, so 4x2 (32) and 4x4 (64) are likely out of reach. [inferred]
- Accumulator cascade width is 512 bits (`include/aie/Dialect/AIE/IR/AIETargetModel.h`, `getAccumulatorCascadeSize`).

## 4. bfp16ebs8 block loads

- Block size and bytes: bfp16ebs8 = 8 mantissas + 8-bit shared exponent, 9 bytes per block (`aie_doc.hpp` L229-233). v64bfp16ebs8 = 8 blocks = 72 bytes = 576 bits (HDR `block_vector_native_types.hpp` L54, L76).
- Standard loads do not work for block vectors. `aie_doc.hpp` L942-947: "Block vector types are unable to be loaded directly using the standard load APIs". Use `block_vector_input_buffer_stream` (`aie_doc.hpp` L942-975).
- Alignment: `block_vector_native_types.hpp` L30 defines `pointer_type = v64bfp16ebs8_unaligned`. The stream is byte-addressed (`seek_byte`), so 64-byte alignment is not required by the API. [inferred; the hardware cost of an unaligned fill is not documented.]
- Stream implementation (`HDR detail/aie2p/array_helpers.hpp` ~L139-240, generic `fifo_buffer_stream` for block/sparse vectors):
  - `pop()` refills the 512-bit fifo only when `load_count_ == 0`; `ld_fill_ratio` = 512 / (576 % 512) = 8 for v64bfp16ebs8, so one refill per 8 pops.
  - `seek(int inc)` moves by `inc * 72` bytes and resets `load_count_` and `state_.pos`, so the next pop refills. [inferred cost: one extra fill per seek.]
  - `pop_seek(int)` / `pop_seek_byte(...)` (~L239-260 and onward) combine the seek and the pop into one operation.
- Your kernel constructs a new `block_vector_input_buffer_stream` and calls `seek(j*colA)` for every (z,j). [inferred] That is one extra refill per 16 mac instructions. Hoisting the stream or using `pop_seek` may help.
- A-side loads: aie_doc.hpp L600-618: aligned `load_v` is undefined behaviour on unaligned pointers; `load_unaligned_v` works but "may incur additional overhead". Your window taps use `load_unaligned_v<64>` on bf16 (`conv3x3w_kernel.cc` L55-82). ATB uses aligned `load_v<sizeA>` (`mm_bfp_mixed_peano_k4.cc` L38-40). This is a possible difference between your kernel and ATB, but the overhead is undocumented.

## 5. Memory sizes and DMA

From `mlir-aie/lib/Dialect/AIE/IR/AIETargetModel.cpp` and `include/aie/Dialect/AIE/IR/AIETargetModel.h` (NPU2 = `BaseNPU2TargetModel`, 8 cols x 6 rows: 1 shim, 1 memtile, 4 core rows; L781-830):
- Core tile data memory (L1): `getLocalMemorySize() = 0x10000` = 64 KB (AIE2TargetModel, ~L568). ATB README says "L1 is 63 KB" usable.
- Core tile program memory: 16 KiB PMEM (`/home/jirka/programy/XDNA2_NANBEIGE_RESIDENT_HANDOFF.md` L5). This is program memory, not data.
- Memtile: `getMemTileSize() = 0x80000` = 512 KB (~L608).
- Locks: core 16, memtile 64 (`getNumLocks`). BDs: core 16, memtile 48 (`getNumBDs`).
- DMA channels (`AIETargetModel.cpp` ~L872-935, switchbox connection counts for the DMA bundle):
  - Core tile: 2 DMA connections per switchbox direction. [inferred: 2 MM2S + 2 S2MM]
  - Memtile: 6 DMA connections per direction. [inferred: 6 MM2S + 6 S2MM]
  - Shim: 2 DMA. `docs/research/xdna2_coherency_ctxswitch_dma.md` L57 cites AM020 as 2 MM2S + 2 S2MM per shim tile.
- Stream width per channel: 32 bit (same research note, L57).
- Shim burst lengths: 64/128/256/512 B encodings (`AIETargetModel.cpp` ~L1602).
- Measured DRAM-side effective BW of ~50 GB/s whole array (`docs/research/xdna2_coherency_ctxswitch_dma.md` L54, citing arXiv 2512.13282; not NPU-local).

## 6. Software pipelining (Peano vs Chess)

- `aie_kernels/aie_kernel_utils.h`:
  - Chess (`__chess__`, L10-32): `AIE_PREPARE_FOR_PIPELINING` -> `[[chess::prepare_for_pipelining]]`; `AIE_LOOP_MIN_ITERATION_COUNT(x)`, `AIE_LOOP_MAX_ITERATION_COUNT`, `AIE_LOOP_RANGE`, `AIE_LOOP_UNROLL(x)`, `AIE_LOOP_FLATTEN`, `AIE_TRY_INITIATION_INTERVAL`-equivalents via `AIE_MODULO_SCHEDULING_BUDGET_RATIO`, `AIE_PEEL_PIPELINED_LOOP`.
  - `__AIECC__` (L34-58): `#pragma clang loop min_iteration_count / unroll_count / pipeline_initiation_interval` (`AIE_TRY_INITIATION_INTERVAL` only here).
  - Otherwise (L60+): all empty.
- Peano-specific detail: [inferred] no file in the tree defines `__AIECC__` except this header, and the Peano compile flags (`utils.py` L320-345) do not set it. So under Peano the pipelining annotations are empty unless clang predefines `__AIECC__` (not checked; no Peano binary is installed locally).
- Trip counts: `AIE_LOOP_MIN_ITERATION_COUNT(16)` on the `scale` example (`programming_guide/section-4/section-4c/README.md` ~L100-101, 124-125). ATB `mm_bfp_mixed.cc` uses `min_iteration_count(4)` on the z-loop. `docs/AIEVectorization.md` ~L80-83 shows `chess_loop_range(256,256)` for an 8-wide vector loop.
- Your loop trip counts per matmul call (M_CORE=64, N=128, K_STEP=64; `conv3x3w.py` L59-61): z = 4, j = 8, i = 2. The innermost i loop has 2 trips, so pipelining it has little benefit and the prologue/epilogue dominates. [inferred] Consider `AIE_LOOP_UNROLL(2)` or flattening `z`/`j`. Chess-only `AIE_LOOP_FLATTEN` is used in `mm_bfp_mixed.cc` under `OPT_PERF_ENABLED`.

## 7. Example kernels

ATB config1 (`gemm_asymmetric_tile_buffering/config1/mm_bfp_mixed_peano_k4.cc`), most relevant:
- 2x2 blocking of 8x8 blocks, `r=s=t=8`. Four `accum<accfloat,64>` C accumulators (c00, c01, c10, c11) live across the K loop inside one call.
- Per i iteration (4 k-blocks): 2 aligned `load_v<64>` of bf16 A per row (A0, A1), 2 `pop()` per column from `block_vector_input_buffer_stream`, 8 converts, 16 `mac_8x8_8x8T`.
- Loop order z (rows, step 2) > j (cols, step 2) > i (K, step 4). The stream is re-seeked per j with `seek(j*colA)`.
- C is loaded from bf16 at the start of each (z,j) and stored as bf16 at the end. Each call covers K=64 and M/4 rows, and there are K//64 calls per output tile (`n32_core_peano_k4_rne.py` L173-179).
- Config1 uses rho=4 asymmetric C buffering, so one C buffer covers 4 A sub-tiles. The README says the configs "currently require the chess compiler" and that Peano crashes on bfp16 matmul. But `config1/n32_core_peano_k4_rne.py` uses `use_chess=False` with Peano kernels. Treat that README statement as possibly stale.

Other references:
- `aie_kernels/aie2p/mm_bfp_mixed.cc`: same 2x2 structure. A0 is upshifted with `accA0 = A0`, A1 with `mul_elem_64`. `z` loop has `AIE_LOOP_MIN_ITERATION_COUNT(4)`. The `i` loop is `for (i<colA; ++i)`, not step 4. B from two streams seeked to `j*colA` and `(j+1)*colA`.
- `programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/config2/` and `config3/` `mm_bfp_mixed.cc`: pure bfp16 (no A conversion), best whole-array number in the ATB README (31.3 TFLOPS).
- `programming_examples/ml/block_datatypes/matrix_multiplication/` (`single_core_mixed`, `whole_array_mixed`, `in_core_shuffle`): reference shapes and layouts for bfp, documented as not tuned.
- `programming_examples/ml/block_datatypes/bfp_conversion/kernel.cc`: bf16 -> bfp16 conversion plus bfp matmul on 8x8 blocks (minimal example).
- `aie_kernels/aie2p/mm.cc` (~L274-300): native bf16 4x8x8 and 8x8x8 matmuls, with the note that 8x8x8 bf16 is only used with bfp16 emulation.

## Not found locally

- Core clock frequency, per-instruction latency and issue rate for `mac_8x8_8x8T`, `to_v64bfp16ebs8`, and block stream fills.
- Exact AIE2P accumulator and vector register counts.
- Cycle cost of the bfp16 conversion.
- Whether Peano predefines `__AIECC__`.
