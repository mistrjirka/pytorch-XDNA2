# Compatibility and coverage

Reviewed on 2026-10-14 by an independent audit of an earlier snapshot; the findings that
matter for this page were fixed afterwards and are marked below. This matrix records **small tested
motifs**, not certification that every model in a family works. The review was
performed on Arch Linux, Ryzen AI 9 365/XDNA2, Python 3.12.13 and
PyTorch 2.14.1+rocm7.2. CPU-reference operations and XDNA operations used the same
rounded inputs and copied initial parameters. No performance claims come from
these coverage runs.

## Read the matrix correctly

- **Correctness coverage** means tested forward values, input/parameter gradients
  and relevant buffers met a stated comparison gate for the sampled case.
- **CPU fallback** can make an XDNA tensor's operation work without NPU arithmetic.
  This does not establish speed or energy savings.
- **NPU acceleration** requires actual dispatch evidence or the configured fixed
  GEMM path with zero CPU-mm fallback calls. A successful tensor on `xdna` is not
  such evidence by itself.
- **Untested** is different from **observed failure**. Neither is a promise that
  an untested combination works.

## Families, missing operators, and where they are used

| Family | Tests performed | Correctness status | NPU status | Missing coverage / limitations |
|---|---|---|---|---|
| Dense 2-D CNNs / style-transfer / residual decoders | Quick-check training comparison on a small residual CNN; BN/view/coherency tests; separate odd-channel and wide-strip conv probes | Small/representative forward and backward passed | Selected BF16 3x3 stride 1 / padding 1 / dilation 1 / groups 1; other shapes use CPU | Not a claim for every size, full training quality, or generic NPU-resident regions. |
| MobileNet / grouped / atrous CNNs | Depthwise+pointwise+SE, grouped/dilated conv, BF16 and FP32 | Main outputs/gradients work; pre-BN bias cancellation is precision-sensitive | These conv varieties primarily CPU | Two raw bias-gradient alarms investigated with FP64; not established as new XDNA bug. No depthwise/1x1 optimized NPU family. |
| 3-D U-Nets / volumetric / video CNNs | Conv3D+BN3+ReLU; InstanceNorm3D; ConvTranspose3D | BF16/FP32 convolution/norm probes passed; FP32 pooling motif passed | CPU fallback | BF16 AvgPool3D failed in both CPU reference and XDNA. No optimized 3-D NPU kernels. |
| GAN / autoencoder / superresolution | Transpose2D+InstanceNorm+Leaky, pixel shuffle, bilinear resize | Tested first-order motifs passed | CPU fallback except eligible ordinary 3x3 parts | Gradient penalties require higher-order derivatives; opt-in fused regions explicitly lack double backward. |
| Diffusion image networks | GroupNorm+SiLU+time projection residual motif | BF16/FP32 motif passed | Mostly CPU; eligible ordinary conv may offload at larger shapes | No full diffusion U-Net/DiT, scheduler, memory or image-quality validation. |
| ViT / dense Transformers / decoder LMs | Tiny encoder, LayerNorm/RMSNorm, SDPA causal/bool/additive/all-masked/GQA cases | BF16/FP32 small forward/backward passed with attention dropout 0 | Attention/norm CPU; only three GEMM shapes have native acceleration | No long contexts, KV cache, fused attention kernels, complex RoPE guarantees, full models or throughput claim. |
| RNN / LSTM / GRU sequence networks | Bidirectional two-layer LSTM/GRU; existing GRU dropout regression; FP32 PackedSequence | Tested cases passed | CPU islands / fallback | State reuse, long sequence stability, BF16 packed paths, projections and complex RNNs not exhaustively tested. |
| Audio / time-series / ASR | Dilated temporal Conv1D+BN/GELU, FP32 CTC, real FFT motif | Tested cases passed | CPU fallback | No optimized temporal/CTC/FFT NPU kernels; full ASR and streaming states untested. |
| Dense embeddings / recommendation | Embedding max_norm/padding and EmbeddingBag mean/sum/max | BF16/FP32 dense-gradient cases passed | CPU fallback | Large tables, sharding and third-party fused embedding kernels untested. |
| Sparse recommenders / sparse GNNs | Sparse Embedding backward and sparse graph multiplication | Explicit errors | No sparse XDNA layout implementation | SparseAdam contract needs actual sparse gradients; no silent dense replacement. |
| Graph message-passing networks | Dense index_select + index_add aggregation and learned linear projection | BF16/FP32 primitive motif passed | CPU fallback | Does not validate PyG/DGL custom extensions, scatter_reduce variants, sparse attention or graph batching. |
| Complex / learned spectral networks | Complex Linear and exact one-element matmul gradient; simple real FFT round-trip | Complex matmul/Linear gradients were wrong in the audited snapshot and are fixed (`tests/run_complex_views.py`); real FFT motif passed | No native complex family (CPU fallback) | Only matmul/Linear/elementwise complex paths are covered. `x.conj()` and `torch._neg_view(x)` return a copy, not a view, so writing through them does not modify `x`. |
| Detection / spatial-transformer networks | FP32 ROIAlign and grid_sample; resize and patch extraction | Tested image/input gradients passed | CPU fallback | ROI-box derivatives not validated; full detection, NMS variants, deformable kernels and third-party ops untested. |
| Scientific / physics-informed models | FP32 SPD Cholesky; pointwise second derivative; jvp and vmap | Limited primitive tests passed | CPU fallback | Not broad higher-order support: custom regions/cpu islands/deferred conv paths need independent gates; complex case fails. |
| State-space / Mamba-like models | Only generic cumsum primitive | Primitive passed; model family not tested | No selective-scan NPU family | Need selective scan, recurrence stability, fused causal conv/scan and long-sequence gradients. |
| Mixture-of-experts / routed networks | No full routed-expert motif | Untested | No general grouped/expert GEMM family | Need router/top-k, duplicate-index scatter, irregular batches, expert gradient accumulation and collective coverage. |
| Quantized / FP8 / integer networks | No quantized-layout or QAT release gate | Untested / not advertised | BFP16 convolution is not PyTorch quantized-tensor support | Need quantized dispatch, fake-quant gradients, observer state and appropriate numerical references. |
| Distributed / sharded training | Single machine/device only | Untested | No verified DDP/FSDP/collective integration | Need process-group behavior, communication, replicated buffer state, mixed precision and multi-process NPU contention. |

## Actual NPU kernel coverage in this review

**General packed convolution:** BF16 3x3, stride 1, padding 1, dilation 1, groups 1.
The fresh hardware tests covered small groups, channel counts 64/65/128 and
output counts 67/128/129, an odd 7x11 image, and wide 7x65 strips. Forward, input and
weight gradients all dispatched on the NPU in the final forced-path test. Relative
L2 error against FP32 using the same BF16-rounded operands was about 1.01-1.10%
forward, 1.05-1.13% dX and 0.90-0.93% dW. These three cases do not define a universal
error bound. The ordinary quick check also verified actual NPU work during its
short training comparison.

**Fixed GEMM:** the source registers `(M,K,N)` shapes `(256,768,3072)`,
`(256,3072,768)` and `(3072,256,768)`. Each direct matrix test reported its CPU-mm
fallback count. The end-to-end Linear training probe also checked forward/dX/dW
values, but included a CPU-mm fallback and must not be described as wholly NPU
backward. Shape, contiguity and graph lowering affect routing.

**Bundled full-frame decoder kernels:** exist and retain earlier targeted tests,
but the complete DRIVE training/validation workload was not freshly rerun in this
release audit. Do not merge their historical performance results with the new
packed-convolution evidence.

**Host regions:** VGG-feature loss, ReLU/pool, BatchNorm+activation and related
reductions run fused on the CPU; their convolutions still dispatch separately.
They are optional, approximate BF16 training optimizations. They are not an
NPU-resident whole-network compiler. The VGG-feature region re-validates the layers (types, attributes, hooks,
parametrizations, parameter identity) on every call and falls back to the module's own method when anything
changed. Fused BatchNorm+ReLU passes gradients for NaN activations like eager ReLU.

## Training workflow interfaces

Three-step CPU-reference comparisons passed for SGD with momentum/decay, Adam,
AdamW with AMSGrad, RMSprop and Adagrad on a small BF16 MLP. That does not cover
all optimizer flags, capturable/differentiable modes, distributed state or sparse
gradients. A tensor/state serialization-to-CPU probe passed; full fused-model
pickling remains explicitly unsupported by the region contract.

Non-reentrant checkpointing of a dense MLP, pointwise higher-order differentiation,
pointwise vmap/jvp and an FP32 PackedSequence LSTM passed. Those are deliberately
narrow results. Fused regions are once-differentiable, and their interaction with
recomputation of stateful training BatchNorm needs a separate test.

`torch.compile(..., backend="eager", fullgraph=True)` executed a small training
example. This checks the tracing frontend only; it does **not** validate Inductor,
AOT-export, an optimized XDNA graph backend or compiled fused-region gradients.
`torch.autocast("xdna")` is implemented (bare `torch.autocast("xdna")` selects BF16;
FP16 is not supported). It reuses PyTorch's own per-op policy lists, so matmul and
convolution family ops run in BF16 and losses, reductions and normalizations stay
FP32. Parameters, gradients and optimizer state stay FP32 and track CPU autocast
(`tests/run_autocast.py`). Mixed-precision BatchNorm (BF16 activations, FP32 affine
parameters) keeps its saved statistics and weight gradients in the parameter dtype.
`torch.cuda.amp.GradScaler` style loss scaling is unnecessary for BF16 and untested here.

## Whole-model workloads

`tests/coverage/model_workloads.py` trains complete small models for six optimizer steps on CPU and on XDNA
from identical weights and data, then compares the loss trajectory and final parameters (and buffers):
a selective state-space scan, a top-2 routed mixture of experts (one expert receives no tokens), a Fourier
operator layer with complex trainable weights, a detection head with ROI Align, and quantization-aware training
with fake quantization and EMA observers. All pass; they run on CPU fallback, `npu_runs` shows no NPU work.
This found and fixed a silent bug: `torch.view_as_real` had no XDNA view kernel, so Adam/AdamW/RMSprop updates of
complex parameters were applied to a copy and lost.

## End-to-end feature models and speed

`tests/coverage/feature_models.py` trains one model per newer feature until it has learned its task and compares
it with the reference: an autocast CNN classifier (FP32 master weights, BF16 compute), the benchmark generator under
autocast (its convolutions run on the NPU: forward, dX and dW), a 3-D CNN with BF16 AvgPool3d, a recommender with
CPU-resident sparse embeddings, and a Fourier operator with complex weights. All reach the same accuracy or loss as
the CPU run. Only the generator uses the NPU; the others run on CPU fallback.

`benchmarks/end_to_end_speed.py` is the speed gate for a whole training step (generator + VGG19 loss, BF16, batch 40,
Adam, ABBA order, separate processes, default settings): on 2026-10-14 it measured CPU 187 to 239 ms,
XDNA 110 to 138 ms (about 1.7x) and XDNA with `fuse_perceptual_loss` 93 to 105 ms (1.8x to 2.3x); with
`--autocast` (FP32 master weights) XDNA was 1.4x faster than CPU autocast. The spread comes from the laptop
heating up between rounds; the gate defaults require 1.2x, 1.4x (fused) and 1.1x (autocast).

## Numerical qualifications

Sparse tensors have no XDNA layout: sparse gradients and sparse graph products fail loudly (`Sparsexdna`). For recommenders, keep the table on the CPU with `xdna_train.cpu_sparse.CpuSparseEmbedding` / `CpuSparseEmbeddingBag` (sparse gradients, `SparseAdam`) and send only the looked-up rows to the XDNA model; `module.to("xdna")` leaves the table in place (`tests/run_cpu_sparse.py`). BF16 AvgPool3D, which PyTorch's own CPU backend lacks, is computed in FP32 and rounded on XDNA (`tests/run_avgpool3d.py`); the matrix keeps `conv3d_volume/bf16` as a known failure only because its CPU reference cannot run. Convolution-only BF16 3D and InstanceNorm3D probes passed.

MobileNet-like conv biases immediately preceding training BN produced large
relative gradient differences because their ideal FP64 gradients were nearzero.
Both CPU BF16 and XDNA BF16 showed cancellation residuals. Output/input-gradient
errors were about 0.5% against FP64. The raw alarms remain in the data; they were
not relabeled as universally safe or used to invent a new backend bug.

Packed BFP16 uses shared exponents. It is not lossless storage of ordinary BF16,
and low aggregate relative error does not guarantee small error for every output
or every low-amplitude value. Explicitly test outliers, cancellation, zero blocks,
long reductions and downstream gradient sensitivity for a new workload. The
supported numerical domain is finite operands. A NaN operand gives NaN outputs for the windows it touches, as on
CPU. An infinite operand also gives NaN outputs (CPU gives signed infinities); it is never silently turned
into a finite result.

## Reproducing and maintaining the evidence

The audit scripts are under `tests/coverage/`. Run in an environment with the repository's
supported PyTorch and installed torchvision for the tests using it. For example:

```bash
PYTHONPATH=. XDNA_AUTO_SETUP=0 python tests/coverage/network_matrix.py --out /tmp/network-results.json
PYTHONPATH=. XDNA_AUTO_SETUP=0 python tests/coverage/training_interfaces.py --out /tmp/interface-results.json
```

The scripts collect failures rather than aborting the full matrix, so their exit 0
is not an all-tests-pass signal. The verdict comes from the gate, which compares
the JSON results with the checked-in expectations in
[`tests/coverage/manifest.json`](../tests/coverage/manifest.json):

```bash
python tests/coverage/gate.py /tmp/network-results.json /tmp/interface-results.json
```

It exits non-zero for an unexpected failure, a changed failure stage/status, a
known failure that now passes (promote the manifest entry to `pass`), a result
the manifest does not list, a dropped case, or fewer NPU runs than recorded.
Each known failure carries an issue id. Bootstrap or refresh entries with
`--init`, then triage by hand; never widen a tolerance just to turn a case green.
The gate's logic is unit-tested without hardware (`tests/test_coverage_gate.py`)
and runs in CI; the hardware matrix itself needs an XDNA2 machine. Do not run
multiple NPU test processes at once.

The manifest was generated from a run on `135e262` and covers 78 network-matrix
results and 13 interface probes. Raw outputs are not tracked in the repository.

### Primary references

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html),
[complex gradients](https://docs.pytorch.org/docs/stable/complex_numbers.html),
[sparse layouts](https://docs.pytorch.org/docs/stable/sparse.html),
[SparseAdam's sparse-gradient requirement](https://docs.pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html),
[new-accelerator AMP integration](https://docs.pytorch.org/docs/stable/accelerator/amp.html), and
[operator registration versus numerical tests](https://docs.pytorch.org/docs/main/library.html).
