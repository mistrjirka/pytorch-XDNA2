# XDNA2 / AIE2P kernel performance: what other projects achieved and what to reuse

Scope: read-only. Nothing was run on the NPU. Local sources under /home/jirka/programy and public web sources. "Measured" means a number the source reports from hardware. "Claim" means a number or statement without a measurement shown.

## 0. Baseline for our kernel

Our conv3x3w kernel (/home/jirka/programy/torch-xdna2-async-dw/xdna_train/_designs/conv3x3w_kernel.cc) is structurally the same as the production llama.cpp dense kernel (mm_bfp_mixed_peano_k4_rne.cc): 2x2 8x8 blocking, 4-step unrolled K, bfp16ebs8 weights, bf16 activations converted per block. The gap is therefore in the surrounding work, not the 2x2 idea.

Per inner step (i += 4) our loop does 4 unaligned A loads, 4 accum-to-bfp16 conversions (`to_v64bfp16ebs8`), 8 B pops, 16 `mac_8x8_8x8T`, and per tile 4 bf16 loads and 4 bf16 stores of C per call. Each converted A block feeds only 2 MACs. At an ideal of 1 vmac per cycle, 5 cycles per MAC gives about 20% of peak, which matches the observed ~19%.

## 1. Local evidence

### 1a. Register-resident arithmetic peak (our own, measured)
- Source: /home/jirka/programy/xdna-tops-20260920/peak-initial.cc, peak-measured-*.json, peak-1-v2.asm; summary in XDNA2_TOPS_AND_EXECUTION_PROFILE.md.
- Setup: 32 cores, operands preconverted to bfp16ebs8 block vectors held in registers, four independent accfloat accumulators (m0..m3), 8x unrolled MAC4 per loop.
- Measured: 54.85 TOPS (65536 repeats) and 58.16 TOPS (262144 repeats), all outputs exact. Ordinary BF16 8x8x8 API without the packed path: about 1.18 TOPS (untuned, expanded instructions).
- Asm: the loop body is 64 consecutive `vmac.f` with operands in ex0/ex2/ex4/ex6/ex8 and no loads or conversions inside the loop. That is about one vmac per issue slot.
- Technique to reuse: the peak kernel has zero conversions and zero memory ops in the inner loop. This is the target shape for ours.

### 1b. Our own dense-path experiments (measured, NPU, complete real blocks)
Source: /home/jirka/programy/XDNA2_DENSE_PIPELINE_RESULT.md, XDNA2_DOCUMENTED_OPTIMIZATION_RESULT.md.
- Two in-flight dense groups instead of one: dense phase 13.644 ms to 11.481 ms (measured, 12 samples). Three gave no further gain.
- Two complete input records per core (ping-pong instead of one record): dense phase 13.49 ms to 8.50 ms, about 37% lower phase latency (measured, 20 samples baseline/new/baseline). Arithmetic kernel unchanged.
- Caching converted activation tiles (convert once, reuse across output-column pairs, conversions 256 to 32): slower, 16.60 ms vs 13.73–14.15 ms controls. Rejected. The extra local packed-buffer writes/reads cost more than the conversions saved.
- Shortening K unroll from 4 steps to 1 step: 17.39 ms vs 14.46–14.70 ms. Rejected. Keep the 4-step unroll.
- Note: our conv kernel already keeps A converted only in registers, which is the variant that is not rejected. The rejected variant was local scratch.

### 1c. Local mlir-aie and llama.cpp kernel sources
- /home/jirka/programy/mlir-aie/aie_kernels/aie2p/mm.cc: blocked 2x2 aie::mmul with rowA/colA/colB template parameters; comment says the 2x2 expansion is chosen for accumulator-register usage and SIMD utilization. No performance numbers in the file.
- /home/jirka/programy/mlir-aie/aie_kernels/aie2p/mm_bfp_mixed.cc: the source our kernel derives from.
- /home/jirka/programy/llama.cpp-xdna2-general-upstream/tools/xdna/coordinator-20261001-consolidate/overlay/gen-fused/:
  - mm_bfp_multirow.cc (matmul_small_m): one A conversion per K step feeds two B accumulators; M8 row tiles looped inside the kernel.
  - mm_bfp_cached_a.cc: pairs two A row tiles into one packed stream and uses a 4-step K pop pattern; conversion of A happens once per call into a scratch region. Carries an opt-in NB_DENSE_CACHE_K1 path. Note the local-scratch caching was measured slower in 1b, so this exact structure is not a win as-is.
  - mm_bfp_single_preencoded.cc: encodes 8 K blocks once, then both N-groups read the same BFP bytes.
  - peak_arithmetic_probe.cc: the same peak probe as 1a, in the llama tree.
- /home/jirka/programy/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/README.md: IRON whole-array GEMM tutorial. No measured TFLOPS in the README (the sweep results are not stored).
- Local conv2d and resnet/bottleneck examples under mlir-aie/programming_examples/ml/ contain no performance numbers in their READMEs.

## 2. Public sources

### 2a. Taka et al., "Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI NPUs", arXiv:2512.13282
Measured (XDNA2 is the ASRock box with Ryzen AI 7 350, not a Strix Point HX; 32 cores in 4x8; 1.8 GHz used for peak):
- int8-int8 GEMM: up to 37.35 TOPS actual vs 39.52 peak reference (kernel 144x72x144, ~95% L1 use).
- bf16-bf16 GEMM: 14.52 TOPS actual (kernel 112x48x96, 4032x4224x4608); 13.67 TOPS (160x40x80). Peak reference 15.81 and 14.30. Peak-sweep max 14.71.
- Techniques credited, with measured effects:
  - Output-stationary mapping: C stays in L1 while K is reduced (Sec. 4.2.1).
  - Single C buffer, double-buffer only A and B: 30.77 TOPS vs 26.1 TOPS with double-buffered C (about 18%, Sec. 5.3.2).
  - Overlap of DMA buffer-descriptor reconfiguration with transfers: without it, int8-int16 drops 30.77 to 22.21 TOPS, a 28% loss (Sec. 4.4, 5.3.3).
  - Column-major B: 8.7% (bf16) to 25.2% (int8-int16) faster than row-major (Sec. 5.2.3).
  - MemTile neighbour-memory sharing for larger tiles (Sec. 4.2.2).
  - kmt = 384 for contiguous A and B reads (Sec. 5.2.2).
- Caveat: these are GEMM, not conv, and the paper says native bfp16 layout handling is future work (Sec. 5.3.4). The bf16 path uses bfp16 emulation.

### 2b. Daniel Estevez, "Getting peak TOPS on a Ryzen AI 7 350 NPU" (8 May 2026), https://destevez.net/2026/05/getting-peak-tops-on-a-ryzen-ai-7-350-npu/
Measured by the author:
- int8 `mac_8x8_8x8` kernel, 32 tiles: xrt-smi gemm validate 51.0 TOPS; benchmark script 56.28 TOPS.
- Trace: 530 cycles per kernel invocation for 512 vmac, i.e. about 96% vmac issue efficiency (539 cycles per call including loop overhead, per the author).
- Structure: two accumulators alternated across two B buffers, so one vmac issues every cycle (avoids the 2-cycle accumulator dependency). Operands streamed from local memory with post-incrementing loads, pipelined ahead of vmac. Not preloaded to registers.
- Claim (not measured): clock about 1.808 GHz inferred from cycles and runtime.

### 2c. Hand-optimized single-tile kernels (project page at uni-jena, "hello xdna", Jan 2026)
- Measured/claimed by the authors: XDNA1 BF16 398 GFLOPS (86% of peak); XDNA2 BFP16 1760 GFLOPS (95% of peak), single tile, 1024 FLOP per cycle peak. This matches roughly one 8x8x8 bfp16 MAC per cycle, which supports the peak-probe reading in 1a.
- No conv numbers, no accumulator counts shown.

### 2d. IRON / mlir-aie (AMD), "Efficiency, Expressivity, and Extensibility in a Close-to-Metal NPU Programming Interface", FCCM 2025
- Could not retrieve the paper text or its numbers. The IRON README and the IRONSmith arXiv:2607.10944 paper say IRON leaves tiling and placement to the designer; no per-kernel numbers were found.

### 2e. AIE4ML, arXiv:2512.15946
- Found, but the PDF body could not be extracted. Performance and kernel details unverified; not used below.

### 2f. Other items checked (no usable numbers)
- AMD Ryzen AI release notes mention conv2d optimizations (7x7, stride-2 3x3 channel augmentation) with no TOPS.
- No public measured bf16 3x3 conv2d TOPS on AIE2P was found.
- TileFuse and FastFlowLM are referenced in our local docs as hybrid-path systems; no kernel-level numbers were extracted for this report.

## 3. Ranked list: five ideas most likely to speed up our kernel

1. **Keep the output accumulators in accfloat across all 9 taps and all K in one kernel call; store bf16 once.** Currently each of the 9 calls loads and stores C as bf16 (rounding each time) and reconverts. Source: Striking the Balance Sec. 5.3.2 (single C buffer, about 18% measured on GEMM); the peak probe and llama mm_bfp kernels keep accumulators across the whole K loop. Expected to remove 8 of 9 C round-trips. Measure before/after on the NPU in your own loop.

2. **Raise reuse of each converted A block from 2 MACs to 4 (2x4 or 4x2 blocking), accumulators in registers.** Our A conversion feeds only two B blocks; the peak probe feeds one converted operand to four accumulators, and the llama multirow kernel feeds one conversion to two. Source: peak_arithmetic_probe.cc / peak-1-v2.asm (measured 54.85 TOPS in the same shape). Caveat from our own measurements: a local-scratch cache of converted A was slower (16.6 vs 14 ms), so keep conversions in registers and only add accumulators.

3. **Overlap input transfer with compute using two complete input records per core (ping-pong).** Source: our XDNA2_DENSE_PIPELINE_RESULT.md, measured 13.49 to 8.50 ms dense phase (about 37%). Apply to the conv window input so the next row-block is in flight while the current one is consumed. Also, two groups in flight gave 16% over one.

4. **Hide DMA buffer-descriptor reconfiguration and make the B layout column-major / contiguous.** Source: Striking the Balance Sec. 4.4, 5.3.3 (28% loss without BD overlap, measured on XDNA2) and Sec. 5.2.2–5.2.3 (kmt = 384 contiguous reads; column-major B 9–25%). For our 9 taps, the per-tap window offset should not need new descriptors each call.

5. **Use the peak probe as a microbenchmark to split the 5 cycles per MAC.** Build variants of the inner loop: (a) with conversions removed (preconverted A in registers), (b) without C load/store, (c) without unaligned loads. Compare against the 1-vmac-per-cycle asm in peak-1-v2.asm. Source: xdna-tops-20260920 probe (measured) and Estevez two-accumulator scheduling trick (measured 96% vmac issue). This tells which of ideas 1–2 pays off before a full rewrite.

## 4. Notes on what not to copy
- The 2-accumulator alternation in Estevez's int8 kernel is only needed when a single accumulator is reused back-to-back. Our loop already rotates 4 accumulators, so the dependency is probably not the cause.
- Reducing K unroll (4 to 1) was measured slower locally; keep the 4-step pop pattern.
- Packed/BFP quantisation of intermediate results changed numerical error (about 6% relative in local attention tests). Any change to conversion placement needs an accuracy check against the bf16 reference.
