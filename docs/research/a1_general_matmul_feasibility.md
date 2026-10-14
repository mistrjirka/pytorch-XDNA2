# A1 general matrix multiplication: feasibility measurements (2026-10-14)

Question: can a general (any M, K, N) matrix multiply on the NPU beat CPU BF16 enough to matter for MLP,
Transformer and 1x1-convolution training?  This note records the first stage only: what the existing and the
upstream kernels deliver on this machine (Ryzen AI 9 365, AC power, nothing else running).  No backend code was
changed; the probe tools are in `benchmarks/gemm_probe/`.

## Method

`benchmarks/gemm_probe/build_gemm.sh` compiles the upstream mlir-aie whole-array GEMM (bf16 inputs, multiply
emulated with BFP16, 4 rows x 8 columns) for one shape and tile; `run_gemm.py` dispatches it through XRT and
reports the min and median of 20 runs after 3 warm-ups.  Time is dispatch to completion with the operands already
on the device (no host packing: bf16 matrices are used as they are).  Error is the relative L2 difference from an
FP32 product of the same bf16 inputs.  CPU: `torch.mm` in BF16, 10 threads.

## Results

| Shape M x K x N | CPU bf16 | existing `train_gemm` | upstream bf16-out | upstream f32-out |
|---|---:|---:|---:|---:|
| 256 x 768 x 3072 | 0.85 ms (1.4 TFLOPS) | 0.64 ms (1.9) | 0.37 ms (3.2), err 2.6% | 0.41 ms (2.9), err 1.5% |
| 512 x 512 x 512 | 0.17 ms (1.6) | - | 0.18 ms (1.5), err 2.1% | 0.22 ms (1.2), err 1.5% |
| 1024 x 1024 x 1024 | 1.18 ms (1.8) | - | 0.39 ms (5.6), err 3.1% | does not fit L1 at 64x64x64 |
| 2048 x 768 x 3072 | 6.4 ms (1.5) | - | does not build at 64x64x96 | 2.6 ms (3.8), err 1.5% |
| 2048 x 2048 x 2048 | 10.1 ms (1.7) | - | 2.8 ms (6.1), err 5.6% | - |
| 4096 x 1024 x 1024 | 4.6 ms (1.9) | - | 1.25 ms (6.9), err 3.2% | 2.2 ms (3.9), err 1.5% |

(`TFLOPS` counts 2*M*K*N; `existing train_gemm` is the fixed-shape BF16-accumulate family now in the backend,
measured earlier at 0.64 vs 0.87 ms forward and slower than CPU for dW.)

## What this says

* A BFP16-multiply GEMM is 2.3x to 3.6x faster than CPU BF16 for products of about 1 GFLOP and more, and at
  parity at 512^3 (dispatch and launch cost about 0.1 to 0.15 ms).  The existing BF16 kernel is only 1.3x.
* Accuracy is the catch.  The upstream design keeps the running sum in a BF16 tile, rounding after every K chunk:
  2 to 6% error, growing with K.  Writing an FP32 result removes the rounding (1.5%, the floor set by BFP16 operands,
  as for the 3x3 convolution) but costs speed (about 1.5x slower), memory (smaller tiles) and a conversion pass
  to BF16 on the host.  The wanted design accumulates in FP32 inside the core and emits BF16 once at the end.
* Every upstream xclbin is built for one (M, K, N): the loop counts are compile-time.  The backend would need an
  xclbin switch (about 2.6 ms) per shape, so the shape must be moved out of the xclbin: tile counts from runtime
  parameters (as the existing `train_gemm` and conv families do) and one instruction stream per shape.

## Next stages (not started)

1. Core with FP32 accumulation and BF16 output on the BFP16 multiply, K-tile and output-tile counts from runtime
   parameters; check it against the numbers above.
2. Shape generalization: padding to tile multiples, instruction-stream generator and cache, transposed A/B/C
   layouts for dX and dW.
3. Backend: `mm`/`addmm`/`bmm`/`linear` routing with a cost model, deferred dW as for convolutions, tests per
   gradient orientation, and the end-to-end gate on an MLP and a Transformer block.
