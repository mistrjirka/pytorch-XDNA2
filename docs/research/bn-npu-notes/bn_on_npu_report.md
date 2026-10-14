# BatchNorm and elementwise activations on the XDNA2 NPU: research notes

Date: 2026-10-13. Branch context: `bfp-input`. Scope: training-mode BatchNorm (forward stats, normalize + affine, fused LeakyReLU/ReLU, backward channel reductions), and elementwise activations, to be fused next to `conv3x3w`.

Status: read-only study. Nothing was built, run on the NPU, benchmarked, or edited. This file is the only file created.

Labels used throughout:
- **[VS]** verified in source: I read the file or header, line numbers given.
- **[DOC]** documented: an external doc or web page. Secondary sources (search snippets, WebFetch summaries) are marked as such.
- **[INF]** inferred from the above, not tested.

Abbreviations:
- `MLIR` = `<local path>` (git HEAD `ed23bba`, 2026-07-02). Its `third_party/aie_api` was not checked for content.
- `AIEAPI` = `<local path>` (wheel `mlir_aie-1.4.0`). Used for all `aie_api` line numbers below.
- `LLVMAIE2P` = `<local path>` (intrinsic headers).
- `<repo>` = `<local path>`.

---

## Summary

1. **Stats fit the conv epilogue.** Each core holds its output tile in L1: 64 pixel rows x 128 channels, bf16, 16 KB [VS: `<repo>/xdna_train/_designs/conv3x3w.py:61-63,114`]. A per-core partial of `[2][128]` fp32 values (sum and sum of squares, or a centred Welford/Chan triple) is 1 KB. The 32 cores give about 32 KB per conv dispatch, against a 13.3 MB activation.
2. **No mlir-aie example reduces a normalization across many cores.** Normalization examples (`norm`, `softmax`, `layer_norm`) assign whole rows or tiles to one core. The cross-core reduction examples are a neighbour-chained column reduce (2 cores per column) and a memtile split/join reduce (4 cores). Neither is used for mean/variance. [VS]
3. **aie2p (XDNA2) math primitives:** fp32 accumulators (`accfloat`), a scalar fp32 `invsqrt` intrinsic, and bf16-only vector forms of `invsqrt`, `exp2` and `tanh`. A vector `invsqrt` on fp32 has no native aie2p path and falls back to a per-element loop [VS, see section 2]. BN plus LeakyReLU needs no transcendental function, so this does not block the design.
4. **Three cross-core combine options** (section 3): (A) per-core partials, host finishes the 32 x 256 floats; (B) a 4-core cascade chain per column, using aie2p `put_mcd`/`get_scd` on `v16accfloat`; (C) a memtile join plus a reducer core. Option C needs a 33rd compute tile, and all 32 tiles are already used by `conv3x3w` (`COLS, ROWS = 8, 4`, conv3x3w.py:59).
5. **Numerics:** keep partial statistics in fp32, and merge them with a Chan/Welford-style combine or with shifted sums. Several existing kernels in the repo accumulate or round in bf16, which costs up to 0.39% relative error per rounding (section 4).

### Recommended first prototype (stage 1, forward statistics only)

1. In `conv3x3w_kernel.cc`, add one function run on the bf16 C tile after the last chunk and before `out_c.release(1)` (conv3x3w.py:189). It reads the 8x8-blocked C tile, masks invalid padded pixels, and accumulates per-channel `sum` and `sumsq` (or a Chan-merged `count, mean, M2`) in fp32 into a `[2][128]` float buffer. Masks need 2-3 spare RTP words (RTP_LEN = 16, 11 used: conv3x3w.py:70; the RTP array is read on-core at conv3x3w.py:161-199).
2. Write that buffer to a second output: one ObjectFifo per column, joined from the 4 cores, the same pattern as the existing C join (conv3x3w.py:145-148), drained to a host buffer.
3. On the host, combine the 32 partials in fp64 into mean, var and invstd per channel. Check them against `torch.nn.functional.batch_norm(training=True)` run on the same bf16 tensor.
4. Apply scale/shift and LeakyReLU in the next layer's existing pack step, or on the CPU for the first test.

Why this first: it changes one design output and one kernel function. The bfp path, the weights and the window layout are untouched. The extra L1 is about 1 KB per core. The host work is 32 x 256 floats. It isolates the numerics before any NPU-side apply exists.

Limits of stage 1: the CPU still does the NCHW unpack and the BN apply, because autograd saves NCHW BN inputs and outputs (`<repo>/docs/research/bfp-input-notes/dataflow_report.md`, section 1). The part removed is the CPU stats pass. The BatchNorm row in `<repo>/docs/research/style_step_breakdown.md` (2.4 ms forward, 3.9 ms backward, 6.2 ms total per step) is the upper bound on what BN-related work could save; the stats pass is only part of that.

Next stages, if stage 1 is accurate:
- Stage 2: move the apply to the consumer. Fold scale/shift and LeakyReLU into the bf16-to-bfp16 encode that the next conv already does on its window (`prep_window`, conv3x3w_kernel.cc:PREP_ONLY block).
- Stage 3: backward. See section 3.4. It needs the forward input, which is the harder part.

---

## 0. Existing notes read and reused

Read first, as asked. I reused these facts and did not re-derive them:

- `<repo>/docs/research/bfp-input-notes/dataflow_report.md`: per-step costs (wait, pack, unpack, sync), the CPU BN/LeakyReLU chain around each conv, the fact that every conv boundary goes through NCHW bf16 on the host, the padded-grid geometry (section 2: "pad columns and between-image rows hold garbage" in the output), the redundant unpack/pack pairs.
- `<repo>/docs/research/bfp-input-notes/bfp16_report.md`: the bfp16ebs8 layout, the 56 KB L1 budget (section 2), the fp32-versus-bf16 running-sum error in the forward C (section 4, 0.37-0.72% RMS), the output-in-bfp idea (section 5), and the warning that BN needs batch statistics before any output encode.
- `<repo>/docs/research/bfp-input-notes/docs_report.md`: ISA and API facts (`accfloat`, `to_v64bfp16ebs8`, `aie_doc.hpp` table at line 1032-1034, the `invsqrt` note in section 2 of that report), the memory sizes (L1 64 KB, memtile 512 KB, 16 locks per core, 16 BDs per core), and the pipelining-macro caveat (Peano may leave `AIE_PREPARE_FOR_PIPELINING` empty).
- `<repo>/docs/research/bfp-input-notes/projects_report.md`: the measured peak probe and the notes on which accumulator counts are unknown. Its "not found locally" list still applies (clock, instruction latency, accumulator register count).
- `<repo>/docs/research/bfp16_input_kernel.md` and `<repo>/docs/research/style_step_breakdown.md`: kernel timings and the step breakdown (BatchNorm 6.2 ms/step, activation 3.9 ms/step).
- Not repeated here: the bfp16 conversion rules, the DMA/shim findings, and the context-switch analysis.

Nothing in these notes covered normalization, reductions, or cross-core combines. Those are new.

---

## 1. Existing kernels and examples (mlir-aie, local checkout)

No batch-normalization kernel exists in `MLIR` for training or inference. A grep for `batchnorm|batch_norm|variance|welford` finds only README text: `MLIR/programming_examples/ml/bottleneck/README.md:22,41,52` (BN and ReLU are folded into the conv for inference), and `MLIR/programming_examples/ml/resnet/README.md:40`. No LeakyReLU exists anywhere in `MLIR` [VS, grep].

| Pattern | File (line) | Data types | Reduction scope | Notes |
|---|---|---|---|---|
| LayerNorm, one-pass | `MLIR/aie_kernels/aie2p/layer_norm.cc` L13-53 | bf16 `T`; fp32 `sumsq` | one row in one core | `sum_acc` is a **bf16** vector (L22, L28: `aie::add` on `vector<T,N>`), so the running sum rounds to bf16 on every add. Variance is `E[x^2]-mean^2` (L38). `aie::invsqrt(float)` at L39. |
| LayerNorm, Welford | `MLIR/aie_kernels/aie2p/layer_norm.cc` L55-99 (instantiated as float, N=16, L106-108) | fp32 | **per column over rows**, vectorised over 16 columns | Welford update per row (L72-82). `inv_count = 1/(r+1)` (L76). Vector `invsqrt` on float (L87). This is the closest existing match to per-channel statistics: rows = pixels, columns = channels. |
| RMSNorm | `MLIR/aie_kernels/aie2p/rms_norm.cc` L13-~60 | bf16 `T`; fp32 accum | one row | `mul_square` to float (L26), `reduce_add` (L30), scalar `aie::invsqrt(float)` (L44). `inv_rms` is broadcast as bf16 (`static_cast<T>(inv_rms)`, after L44), so the scale carries a bf16 rounding. |
| Norm design | `MLIR/programming_examples/ml/norm/norm.py` L65, 72, 83-96 | bf16 | 8 cores, whole rows each | No cross-core reduction. `n_cores = 8`; each core handles `sequence_length // 8` rows. |
| Softmax | `MLIR/aie_kernels/aie2p/softmax.cc` L12, 59-89 | bf16 in/out; accfloat sums | one 1024-element tile | Three passes: max (L59-64), exp2 (L76), sum in accfloat (L84). **The reciprocal is rounded to bf16** (`col_sum_inv`, L85). |
| Softmax design | `MLIR/programming_examples/ml/softmax/softmax.py` L40-42, 74 | bf16 | per 1024 tile, no cross-tile reduction | Test tolerance `rtol=0.128` (L74), a sign of the bf16 error. |
| tanh-approx GELU | `MLIR/aie_kernels/aie2p/gelu.cc` L1-75 | bf16; tanh via float input | element-wise | `aie::tanh<bfloat16>(x.to_vector<float>())`. |
| SiLU | `MLIR/aie_kernels/aie2p/silu.cc` L15-45, tanh at L36 | bf16 | element-wise | Intermediate `half_x`, `tanh`, and `sigmoid` are each rounded to bf16 vectors. |
| SwiGLU | `MLIR/aie_kernels/aie2p/swiglu.cc` L1-78 | bf16 | element-wise, two weight streams | Same tanh-approx structure as SiLU. |
| exp via exp2 | `MLIR/aie_kernels/aie2p/bf16_exp.cc` L1-45 | bf16 | element-wise | `exp(x) = exp2(x * log2e)`; the `exp2` result is bf16. |
| ReLU | `MLIR/aie_kernels/aie2/relu.cc` (XDNA1 directory) | bf16 `v32bfloat16` | element-wise | `max(input, zeroes)`. No aie2p copy exists in the checkout. The IRON kernel selects its source by arch (`mlir_aie/python/aie/iron/kernels/_common.py:146-149`). |
| Elementwise add/mul | `MLIR/aie_kernels/aie2/add.cc`, `mul.cc` (XDNA1 directory); `MLIR/programming_examples/ml/eltwise/eltwise.py` | bf16 | element-wise | Same arch caveat as ReLU. |
| Scale + shift | `MLIR/aie_kernels/aie2/scale_shift.cc`; `MLIR/programming_examples/ml/scale_shift/scale_shift.py` | bf16 | element-wise | Time-multiplexed with an RTP (`rtp = 1` multiply, `rtp = 0` add). Not a per-channel broadcast. |
| Per-column bias add | `MLIR/programming_examples/basic/row_wise_bias_add/kernel.cc` | fp32 | per-column bias, looped over rows | Holds one 32-wide bias vector in a register and streams all rows of that column block (`for j` / `for i` loops). This is the traversal to copy for a per-channel affine apply. |
| Reduce, single core | `MLIR/programming_examples/basic/vector_reduce_add/`; `MLIR/python/aie/iron/algorithms/reduce.py` | int32 / bf16 | whole input in one kernel call | `reduce.py` says the input is handed to the kernel in one call; no tiling. |
| Reduce max, 8 cores, neighbour chain | `MLIR/programming_examples/basic/vector_reduce_max/multi_column_designs/col_wise_vector_reduce_max.py` L68 (`cores_per_col = 2`), L134-153 | i32 / bf16 | 8 cores, partials chained to the next core in the column | A core takes its neighbour's partial through an ObjectFifo (`fifo_args.append(of_outs[i - cores_per_col].cons())`, L150-153). The last core writes the final value. |
| Reduce max, memtile join | `MLIR/programming_examples/basic/vector_reduce_max/single_column_designs/vector_reduce_max_memtile.py` L79-94 | i32 | 4 cores; split/join through memtile | `of_in.cons().split(...)` (L89) and `outC.prod().join(...)` (L94). Core 0 reduces the 4 joined scalars. |
| Cascade matmul | `MLIR/programming_examples/basic/matrix_multiplication/cascade/cascade.py` L264 (`CascadeFlow(workers[row][col], workers[row-1][col])`) | bf16/i32 | vertical chain of cores | Uses `aie2` kernel `MLIR/aie_kernels/aie2/cascade_mm.cc` (`put_mcd`/`get_scd` on `v16int32`, L31-65). Not an fp32 example. |
| Magika LayerNorm-like | `MLIR/programming_examples/ml/magika/kernels/group2.cc` L60-130 | bf16 operands, fp32 `acc0/acc1`, bf16 mean | one core, one vector | Computes mean (L94) and variance (L98-103) on one core, then a LUT-based inverse sqrt. **Chess-only** (`chess_prepare_for_pipelining`). **Licence: `LicenseRef-AMD-Proprietary`** (file header). Read for ideas only; do not copy. |

Local projects outside `MLIR` (not mlir-aie, but our own code on this machine):
- `<local path>`: RMSNorm over a 3072-wide row, one core per token block. Its helpers are the most useful prior art here: `logical()` (L6-16) is the 8x8 tile address map used for the mmul output (same as our C tile, section 3); `load_float8`/`store_float8` (L27-35) move bf16 to and from fp32 accumulators; `sums` and mean are fp32 (L101-109); `aie::invsqrt` on fp32 (L109); `swap_rounding(conv_even)` (L68, L144, L185); and `bf16_times_float` (L49-58) splits an fp32 scale into three bf16 parts (hi/mid/lo) for a more exact multiply (`NB_MIXED_NORM`, L123-127).
- `<local path>`: the same RMSNorm with gamma. It rounds `inv_rms` to bf16 (L27), which adds a 0.39% relative error to every output element (section 4).

Not found: any fp32 cross-core reduction over more than 4 cores; any training BN; any LeakyReLU.

---

## 2. Math primitives for aie2p (XDNA2)

Arch: `XDNA2 = 21`, `AIE_MLv2 = 22` [VS: `AIEAPI/detail/config.hpp:120-130`]. Code paths for XDNA2 are selected by `arch::is(arch::XDNA2)`.

### 2.1 Accumulators and fp32 arithmetic

- `accfloat` is an accumulator tag supported on XDNA2 [VS: `AIEAPI/aie_doc.hpp:511-512`; the default accumulator table `aie_doc.hpp:1031-1034` lists XDNA2 with `%accfloat` for bf16 products].
- `aie::accum<accfloat,16>` is 16 fp32 lanes, 512 bits. This is the working width for per-channel statistics [VS: `layer_norm.cc:23` uses `vector<float,N>` with N=16; `LLVMAIE2P/aie2p_streams.h:378` puts `v16accfloat` on the cascade].
- `aie::mul_square(bf16 vec)` returns an accumulator, and `to_vector<float>` gives the fp32 squares [VS: `AIEAPI/aie.hpp:5879-5908`; used at `rms_norm.cc:26`].
- Vector sum: `aie::reduce_add(vector<float>)` returns fp32 [VS: `AIEAPI/aie.hpp:4260-4269`]. `aie::reduce_add<bfloat16>(acc)` returns a bf16 result [VS: `aie.hpp:4294-4299`, documented example at `aie.hpp:4286-4290`]. Use the fp32 form.
- Bf16 rounding: `aie::swap_rounding(aie::rounding_mode::conv_even)` sets RNE for later conversions [VS: `<repo>/xdna_train/_designs/conv3x3w_kernel.cc:105`; `block_norm.cc:68`]. The default rounding mode for `to_vector<bfloat16>` is the current mode [INF: not checked in the headers].

### 2.2 Inverse square root, square root, reciprocal

- **Scalar fp32 `invsqrt`:** `invsqrt(float)` maps to `__builtin_aie2p_invsqrt` [VS: `LLVMAIE2P/aie2p_nlf.h:19`]. Also `inv(float)` (L25) and `_sqrtf(float)` (L17). The aie_api scalar entry point is `aie::invsqrt(E)` [VS: `AIEAPI/aie.hpp:7097-7100`], with the aie2p float scalar specialisation at `AIEAPI/detail/aie2p/elementary.hpp:477-486`.
- **Vector `invsqrt` on bf16:** a native path exists [VS: `AIEAPI/detail/aie2p/elementary.hpp:275-277`, `elementary_vector_bits_impl<InvSqrt,16,bfloat16,bfloat16,N>`; used at `layer_norm.cc:87` only through the float instantiation].
- **Vector `invsqrt` on fp32:** no aie2p specialisation was found for `(InvSqrt, 32, float, float)`. The call `aie::invsqrt(vector<float>)` (`AIEAPI/aie.hpp:7253-7257`) therefore goes to the generic `elementary_vector_bits_impl` (`AIEAPI/detail/elementary.hpp` ~L57-90), which loops over the elements and calls the scalar form [VS for the fallback loop; INF that the float vector case takes it, since no specialisation was found]. **Consequence: an fp32 vector `invsqrt` costs one scalar call per lane.** For per-channel scales (128 values per core) that is negligible. Do not put it in the per-element loop. Compute it once per channel.
- `block_norm.cc:109` calls `aie::invsqrt` on a `vector<float,8>` and works; its cost was not measured.

### 2.3 exp, tanh, LeakyReLU

- `exp2` (vector): the aie2p intrinsic returns `v16bfloat16` only [VS: `LLVMAIE2P/aie2p_nlf_vector.h:13`; the `aie.hpp:8110-8118` requires `XDNA2` with `TR = bfloat16`]. The fp32 `exp2` path with `exp2_bf20` is for `arch == 22` (AIE-MLv2) only [VS: `AIEAPI/detail/aie2p/elementary.hpp:351-369`, `#if __AIE_ARCH__ == 22`]. **On XDNA2, exp2 and therefore exp output is bf16.**
- `tanh`: `aie.hpp:8092-8100` requires XDNA2 or AIE-MLv2 and returns `TR`, default bf16. Intrinsic `v16bfloat16` [VS: `aie2p_nlf_vector.h:14`].
- **LeakyReLU** has no existing kernel. It needs only `max` and `mul` on bf16 vectors. The max is used in `relu.cc` (`max(input, zeroes)`, `aie_kernels/aie2/relu.cc`, the XDNA1 directory). LeakyReLU = `max(x, a*x)` for `0 < a < 1`, which can use the same `aie::max` [INF: the aie2p `max` on `v32bfloat16` is the same API, not compiled here].

### 2.4 Vector widths and loops

- 512-bit vector registers: `v32bfloat16`, `v16accfloat` [VS: `relu.cc` uses `v32bfloat16`; `aie2p_streams.h:378` cascade word is `v16accfloat`].
- Loop annotations: `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_MIN_ITERATION_COUNT` are defined only under `__chess__` or `__AIECC__` (`MLIR/aie_kernels/aie_kernel_utils.h`). Our conv uses Peano (`use_chess=False`), so these are probably empty. Already noted in `docs_report.md` section 6. Not re-checked here.

### 2.5 Not found

- No aie2p instruction for a fp32 vector inverse square root.
- No aie2p fp32 exp2 on XDNA2.
- No Welford intrinsic. Welford is plain arithmetic (`layer_norm.cc:72-82`).
- No documented cycle counts for `invsqrt`, `exp2` or `tanh`.

---

## 3. Per-channel reduction on our layout

### 3.1 Our layout

- Each core owns `M_CORE = 64` output rows and all `N = 128` output channels [VS: `conv3x3w.py:61-63`, and the design header at lines 9-12].
- The core's output tile `C_l1` has type `(64, 128)` bf16 = 16 KB [VS: `conv3x3w.py:114`]. The kernel writes it as 8x8 mmul blocks: block `(z, j)` is stored at element offset `(z*colB + j) * 64`, `colB = 16`, `z` over 8 pixel blocks [VS: `conv3x3w_kernel.cc:51-93`, `matmul_window_bfp`, stores at the `aie::store_v(pC1, ...)` lines].
- The same 8x8-block address map is written out in `block_norm.cc:6-16` (`logical()`): `c + (row/8)*1024 + (row%8)*8 + (col/8)*64 + col%8`. A per-channel loop must use this map, not row-major indexing.
- The memtile DMA converts the blocked tile back to row-major on the way to DRAM: `c_l2l3_dims = [(M_CORE//r, r*N), (r, t), (N//t, r*t), (t, 1)]` with `r = t = 8` [VS: `conv3x3w.py:104, 129`].
- Padded grid: output rows are padded-grid pixels; pad columns and between-image rows contain garbage [VS: `dataflow_report.md` section 2, from the kernel geometry]. **Stats must mask these rows.** The core currently gets only `groups, chunks` and 9 tap offsets from the RTP block (`conv3x3w.py:161-199`, RTP_LEN = 16). The mask needs `W`, `H` and the core's global row base, which go into the 5 spare RTP words, or into a precomputed mask buffer.
- Memory: L1 is 64 KB, and about 56 KB is in use per the bfp16 notes (section 2). About 8 KB is free. Partial stats need 1 KB per core.
- 32 cores: `COLS, ROWS = 8, 4` [VS: `conv3x3w.py:59`]. Each column has 4 vertically adjacent cores, which matters for the cascade option.

### 3.2 Design A: per-core partials, host finishes (recommended first)

- Core: while its C tile is still in L1, after the last chunk and before `out_c.release(1)`, compute `sum[c]` and `sumsq[c]` (fp32) over its valid rows (masked), for all 128 channels. Vector work: 8192 values, 512 accumulator updates of 16 lanes each [INF on cycles, expected microseconds].
- Output: `[2][128]` fp32 = 1 KB per core, 32 KB per dispatch. Routed through a new ObjectFifo per column, joined from its 4 cores (the existing C join pattern, `conv3x3w.py:145-148`), then drained to a host buffer.
- Host: sum the 32 partials in fp64, or Chan-merge them. Output mean, var and invstd per channel (128 values).
- Data movement: +32 KB output per dispatch. The C output path is unchanged.
- Risk: one more ObjectFifo per column and one more shim drain. The conv already has 4 outstanding shim tasks per channel, and the shim limit (4 tasks per DMA channel, from the conv3x3w.py header) must be respected. [VS: `conv3x3w.py:23-24` describes the limit.]

### 3.3 Design B: 4-core cascade chain per column

- `CascadeFlow(workers[row][col], workers[row-1][col])` between vertically adjacent cores, as in `cascade.py:264`.
- Each core adds its partial and forwards it: `get_scd`, add, `put_mcd`. A `[2][128]` fp32 partial is 256 floats = 16 cascade words of `v16accfloat` [VS: `LLVMAIE2P/aie2p_streams.h:20` (`get_scd_v16accfloat`), `:378` (`put_mcd(v16accfloat)`)]. Three hops per column.
- Output: 8 column results (8 KB) to the host, or one drain per column.
- Pros: the host reads 8 partials instead of 32; the combine happens on the array.
- Cons: the cascade example in the repo uses `aie2` kernels only (`cascade_mm.cc`, `v16int32`). The fp32 cascade intrinsics exist in the headers, but nothing in the repo uses them [INF: not compiled]. A cascade also serialises the 4 cores in a column (each waits for the previous one), which can stall the existing ObjectFifo schedule. Medium-high risk. Not the first prototype.

### 3.4 Design C: memtile join and a reducer core

- Each column's 4 partials go through the memtile join (like `vector_reduce_max_memtile.py:94`), and one core reduces them.
- **Blocker:** all 32 compute tiles are already busy with conv cores [VS: `conv3x3w.py:59`, `Worker.grid(ROWS, COLS, ...)` at L191-196]. There is no spare core unless the conv uses 31 cores or time-multiplexes a reducer. The memtile itself has no compute.
- Verdict: not viable without giving up a core. Keep as a fallback.

### 3.5 Backward (training)

Backward needs `sum(dy)` and `sum(dy * xhat)` per channel, then the input gradient (per-channel coefficients times dy and x) [DOC: standard BN backward; no mlir-aie source].

- `dy` is the bfp input to the dX conv. `x` (the forward input, `xhat = (x - mean) * invstd`) is not on the dX core. It is in the bf16 dW layout (`x16`, `dataflow_report.md` section 1), which is taken in backward for the dW kernel.
- Algebraic option: `sum(dy * xhat) = invstd * (sum(dy*x) - mean * sum(dy))`. This needs `sum(dy*x)` uncentred, which has the cancellation problem in section 4.
- Option: compute the backward reductions in the dW pass. dW already reduces products over pixels per channel, and `sum(dy)` is a dW-style reduction with a ones vector [INF].
- dX itself is per-channel coefficient times `dy`, plus per-channel times `x`. That is the same broadcast shape as the apply, so the `row_wise_bias_add` traversal (section 1) applies.
- Verdict: backward is stage 3. Its reduction inputs are harder than forward.

### 3.6 Variant of Design A: ones-vector reduction

[INF] Column sums of the 64 x 128 tile can be computed by a matrix product with a ones row, using the existing mac/mmul path with fp32 accumulators. This would replace the scalar-per-lane sums with mmul calls. It needs the squares in bf16 (`x*x` rounded to bf16) for the sumsq term, which adds rounding. Not recommended for the first prototype.

---

## 4. Numerical stability for bf16 training BN

### 4.1 Precision facts

- bf16 has 8 significant bits. Unit roundoff with round-to-nearest is 2^-8 = 3.9e-3. fp32 is 2^-24 = 6.0e-8 [standard arithmetic; the bf16 format is described in [DOC] Kalamkar et al., arXiv 1905.12322, "A Study of BFLOAT16 for Deep Learning Training", <https://arxiv.org/abs/1905.12322>, which reports that bf16 training matches fp32 results when reductions and the other accumulations are kept in fp32 (summary from search, not a full read)].
- The forward C tile is already bf16 and already has a bf16 running sum with 9 roundings per output (`bfp16_report.md` section 4). Stats computed on that C tile include that error. It is small compared with BN's own errors, but it is there.

### 4.2 Sums and variance

- Accumulating N bf16 values in bf16 is the worst case: error grows roughly as N*u_bf16 worst case, about sqrt(N)*u typical. `layer_norm.cc:22,28` does this (`sum_acc` is `vector<T,N>` with T = bf16). **Do not copy this.**
- Accumulating in fp32 (accfloat): error about N*u_fp32 worst case, about sqrt(N)*u_fp32 typical. For N = 51,840 pixels per channel (40 x 36 x 36), sqrt(N)*u_fp32 is about 1.4e-5. This is fine.
- Naive `E[x^2] - mean^2` (`layer_norm.cc:38`, and the sums in design A) loses precision when `mean^2 >> var`. Each factor of 10 in `mean/std` costs about two decimal digits. After ReLU, the mean can be a large fraction of the std, so this matters. Mitigation: centre the data with a shift `K` (for example the previous step's running mean, or the first pixel's value) before squaring. Sum `(x-K)` and `(x-K)^2`, then correct: `var = sum2/N - (sum1/N)^2`.
- Chan/Welford pairwise merge is the robust option. Chan, Golub and LeVeque (1979 Stanford report STAN-CS-79-773; 1983 *The American Statistician* 37(3) "Algorithms for Computing the Sample Variance: Analysis and Recommendations") give the merge: for subsamples of sizes `m` and `n`, `S = S_1 + S_2 + (m/(n(m+n))) * ((n/m)*T_1 - T_2)^2`, where `T` is the sum and `S` is the sum of squared deviations. [DOC secondary: formula as restated in arXiv 2410.21922, <https://arxiv.org/pdf/2410.21922>; not checked against the 1983 paper.]
- PyTorch CUDA BN uses the same Welford-style merge. [DOC: WebFetch summary of `aten/src/ATen/native/cuda/Normalization.cuh` on the `main` branch, <https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Normalization.cuh>. The summary names `batch_norm_collect_statistics_kernel` and `welford_merge_element`, which does `mean = (mean_new*count_new + mean*count)*factor` and `m2n += m2n_new + delta0^2 * count_new * count * factor`. The version is not pinned, and I did not read the file line by line.]
- mlir-aie's `layer_norm_welford` (`layer_norm.cc:55-99`) does a per-column Welford over rows in fp32, which is the same algorithm in a per-channel layout [VS].
- The Apex `syncbn.cpp` binds Welford mean/variance, Welford parallel reduce and BN forward/backward CUDA kernels [DOC: search snippet, <https://huggingface.co/spaces/Taf2023/Open-Sora/blob/fd2030c5b3f2e4b5f3f34436c20faba04d17f72c/apex/csrc/syncbn.cpp> (a mirror of NVIDIA Apex)].

### 4.3 bf16 rounding in the existing code (do not copy)

| Where | What is rounded to bf16 | Effect |
|---|---|---|
| `layer_norm.cc:22,28` | the running sum | error grows with N |
| `softmax.cc:85` | `1/sum` | 0.39% relative error on every output; the test tolerance is 0.128 (`softmax.py:74`) |
| `rms_norm_gamma.cc:27` (probe) and `rms_norm.cc` after L44 | `inv_rms` as the broadcast scale | 0.39% relative error on every output element |
| `magika/kernels/group2.cc:86, 94, 123` | partial sums and `mean` as bf16 | mean error 0.39% relative; variance then computed from a rounded mean |
| `silu.cc:34-39`, `gelu.cc` | intermediates `half_x`, `tanh`, `sigmoid` as bf16 vectors | several roundings in sequence |

Better pattern, from `block_norm.cc`: fp32 scale, applied with `bf16_times_float` (hi/mid/lo split, L49-58) when the scale needs more than bf16 precision (`NB_MIXED_NORM`, L123-127). Output rounded once with conv_even.

### 4.4 Consistency

- BN statistics must match the values the normaliser sees. Compute them on the stored bf16 C tile (what gets written to DRAM), not on the fp32 accumulators. Then stats and normalised output refer to the same numbers.
- The forward C already carries the bf16 running-sum error (section 4.1). That error is in the statistics too. It is not a BN bug, but it should be in the accuracy budget.

### 4.5 Mixed-precision practice

- [DOC] Micikevicius et al., "Mixed Precision Training", arXiv 1710.03740, section 3.3: "Such reductions mostly come up in batch-normalization layers when accumulating statistics and softmax layers." The paper says these reductions do arithmetic in FP32 while tensors stay in FP16. It does not say whether the stored mean and variance are FP32 or FP16 (WebFetch of `https://ar5iv.labs.arxiv.org/html/1710.03740`).

---

## 5. Prior art

Training BN on AMD NPUs or AI Engine arrays: **none found**. Searches covered AIE, Versal and Ryzen AI. Results below are secondary (search snippets or repo files), not full reads.

- **Local:** the BN text in `MLIR/programming_examples/ml/bottleneck/README.md:22,41,52` is about folding BN into the conv for inference. No training-mode BN.
- **Academic one-pass BN for on-device training:** "Memory-Efficient Batch Normalization by One-Pass Computation for On-Device Training", <https://scholar.xjtu.edu.cn/en/publications/memory-efficient-batch-normalization-by-one-pass-computation-for-/> [DOC: title and abstract only].
- **IEEE TCAS-II 2024** (one-pass BN accelerator, 2.0-3.8x less off-chip access than prior designs, from a search snippet; the paper notes that the two-pass BN training computation needs two reads of the input) [DOC: secondary, not fetched].
- **Versal BN accelerator, SoCC 2025** (fabric BN primitive, nine cycles latency, fixed-point) [DOC: secondary, search snippet only]. It is FPGA fabric, not AI Engine tiles.
- **Huawei Ascend CANN design guide** for BN on a tile-based NPU: when channels outnumber H x W, compute BN along channels pixel-wise in parallel; otherwise tile along each channel and ping-pong buffer between global memory and the unified buffer. <https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/tbeaicpudevg/atlaste_09_0006.html> [DOC: search snippet, not fetched]. The second case is the same shape as our layout.
- **NVIDIA Apex** `syncbn.cpp` (Welford BN CUDA kernels, channel-last variants): mirror URL in section 4.2 [DOC].
- **MXNet** `src/operator/nn/batch_norm.cu`, adapted from Torch [DOC: search snippet, <https://apache.googlesource.com/mxnet/+/HEAD/src/operator/nn/batch_norm.cu>].

---

## 6. Open questions and what to measure

Ordered by what blocks stage 1:

1. **Mask and row mapping:** confirm which output rows of the padded grid are valid for each core and group. Use the geometry in `dataflow_report.md` section 2 and `conv3x3w.py` (`geometry()`, L82-89).
2. **Cascade flow with fp32:** a compile-only test of `put_mcd(v16accfloat)` / `get_scd_v16accfloat` in an aie2p kernel. Needed only for design B.
3. **Float vector invsqrt codegen:** compile `aie::invsqrt(vector<float,16>)` for aie2p and check whether the emitted code is a scalar loop. Affects only the per-channel scale, so low priority.
4. **Stats accuracy:** a host-side model first. Take bf16 conv outputs from a real layer, compute stats in fp64, in fp32 naive, and in fp32 shifted or Chan-merged, and compare against `torch.nn.functional.batch_norm(training=True)`. No NPU needed for this step.
5. **Exact rounding of `to_vector<bfloat16>`:** check whether it honours `conv_even` (the conv kernel sets it at L105; `block_norm.cc` sets it explicitly). Needed for the bit-exact apply in stage 2.
6. **Output DMA budget:** a second output per column uses shim/memtile DMA channels. Check the channel count against the per-column use in `conv3x3w.py:133-150`.
7. **Pipelining macros under Peano:** `AIE_PREPARE_FOR_PIPELINING` may be empty (`docs_report.md` section 6). Any new stats loop should be measured with and without it.

Not done in this study: no NPU run, no compile, no benchmark, no numerical test. All performance estimates above are INFERRED.

---

## 7. Sources

Local (read, line numbers in the text):
- `MLIR/aie_kernels/aie2p/{layer_norm,rms_norm,softmax,gelu,silu,swiglu,bf16_exp}.cc`
- `MLIR/aie_kernels/aie2/{relu,add,mul,scale_shift,cascade_mm}.cc`
- `MLIR/aie_kernels/aie_kernel_utils.h`
- `MLIR/programming_examples/ml/{norm,softmax,eltwise_unary,scale_shift,magika,bottleneck,resnet}/...`
- `MLIR/programming_examples/basic/{vector_reduce_max,row_wise_bias_add,vector_reduce_add,matrix_multiplication/cascade}/...`
- `MLIR/python/aie/iron/algorithms/reduce.py` (via the installed wheel; `MLIR/python` path differs in the wheel: `mlir_aie/python/aie/iron/...`)
- `AIEAPI/{aie.hpp,aie_doc.hpp,detail/config.hpp,detail/elementary.hpp,detail/aie2p/elementary.hpp}`
- `LLVMAIE2P/{aie2p_nlf.h,aie2p_nlf_vector.h,aie2p_streams.h}`
- `<repo>/xdna_train/_designs/{conv3x3w.py,conv3x3w_kernel.cc}`
- `<local path>`
- `<local path>`
- `<repo>/docs/research/bfp-input-notes/{dataflow_report,bfp16_report,docs_report,projects_report}.md`, `<repo>/docs/research/{bfp16_input_kernel,style_step_breakdown}.md`

Web (fetched or searched; labelled in the text):
- PyTorch `Normalization.cuh` (WebFetch summary): <https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Normalization.cuh>
- Micikevicius et al., arXiv 1710.03740 (WebFetch of ar5iv): <https://ar5iv.labs.arxiv.org/html/1710.03740>
- Kalamkar et al., arXiv 1905.12322 (search): <https://arxiv.org/abs/1905.12322>
- Chan/Golub/LeVeque restated, arXiv 2410.21922 (search): <https://arxiv.org/pdf/2410.21922>
- Apex syncbn mirror (search): <https://huggingface.co/spaces/Taf2023/Open-Sora/blob/fd2030c5b3f2e4b5f3f34436c20faba04d17f72c/apex/csrc/syncbn.cpp>
- Huawei Ascend CANN BN design guide (search): <https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/tbeaicpudevg/atlaste_09_0006.html>
- One-pass BN, on-device training (search): <https://scholar.xjtu.edu.cn/en/publications/memory-efficient-batch-normalization-by-one-pass-computation-for-/>
- MXNet BN (search): <https://apache.googlesource.com/mxnet/+/HEAD/src/operator/nn/batch_norm.cu>
