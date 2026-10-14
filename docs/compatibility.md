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
| Dense 2-D CNNs / NIS / residual decoders | Core NIS quick check; BN/view/coherency tests; separate odd-channel and wide-strip conv probes | Small/representative forward and backward passed | Selected BF16 3x3 stride 1 / padding 1 / dilation 1 / groups 1; other shapes use CPU | Not a claim for every size, full training quality, or generic NPU-resident regions. |
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
`torch.autocast("xdna")` failed at Linear because the Autocastxdna dispatch path
was missing. Use explicit supported dtypes until a proper autocast policy exists;
manual `.bfloat16()` is not a replacement claim for AMP support.

## Numerical qualifications

Sparse gradients fail loudly (`Sparsexdna`); `torch.autocast("xdna")` is not implemented. BF16 AvgPool3D is unavailable in the tested
CPU reference as well, so it is a shared dtype limitation, not an XDNA-only bug.
Convolution-only BF16 3D and InstanceNorm3D follow-up probes passed.

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

The scripts collect failures rather than aborting the full matrix. Their exit 0
is not an all-tests-pass signal: inspect each JSON status. Do not run multiple
NPU test processes at once.

Raw evidence is provided in the accompanying review archive under `results/`.
This documentation and the scripts were prepared in an isolated worktree and
have not been committed or pushed. The machine-readable matrix is
[network_kernel_coverage.json](network_kernel_coverage.json).

### Primary references

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html),
[complex gradients](https://docs.pytorch.org/docs/stable/complex_numbers.html),
[sparse layouts](https://docs.pytorch.org/docs/stable/sparse.html),
[SparseAdam's sparse-gradient requirement](https://docs.pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html),
[new-accelerator AMP integration](https://docs.pytorch.org/docs/stable/accelerator/amp.html), and
[operator registration versus numerical tests](https://docs.pytorch.org/docs/main/library.html).
