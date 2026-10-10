# torch-xdna2

Experimental first-class PyTorch backend for the AMD XDNA2 NPU in Ryzen AI / Strix systems.

The target interface is ordinary PyTorch:

```python
import torch
import xdna_train

device = torch.device("xdna")
model = MyModel().bfloat16().to(device)
x = x.bfloat16().to(device)

y = model(x)
loss = loss_fn(y, target.bfloat16().to(device))
loss.backward()
optimizer.step()
```

No model rewrite is required for the first-class device path. Unsupported operations execute on the CPU over the same host-visible XRT-backed tensor storage.

> **Status:** experimental / alpha. Current performance work targets BF16 training on XDNA2. FP32 tensors mostly use CPU compatibility paths and are not the performance target.

## Install

### 1. System prerequisites

You need:

- a working AMD `amdxdna` kernel driver;
- AMD XRT development headers and runtime (`xrt/xrt_device.h`, `libxrt_coreutil.so`);
- a C++20 compiler.

On the Arch Linux development machine the XRT packages are:

```text
xrt
xrt-plugin-amdxdna
```

The Python package does **not** install the kernel driver or XRT.

### 2. Install directly from GitHub

With uv:

```bash
uv pip install "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

Install the PyTorch build you want first (for example ROCm on AMD systems). If you are force-reinstalling only this backend and want to preserve that PyTorch build, use:

```bash
uv pip install --reinstall --no-deps "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

Or add it to a uv project:

```bash
uv add "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

With pip:

```bash
pip install "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

The first import builds a small native bridge against the **exact installed PyTorch ABI** and caches it under `~/.cache/torch-xdna2/`. Validated XDNA instruction streams are bundled with the package.

To build/check explicitly:

```bash
python -m xdna_train
# or
torch-xdna2-build
```

If XRT is in a non-standard location:

```bash
export XRT_INCLUDE_DIR=/path/to/xrt/include
export XRT_LIB_DIR=/path/to/xrt/lib
```

## Quick check

```python
import torch
import xdna_train

xdna_train.register_xdna_device()
print(torch.xdna.is_available())

x = torch.ones(1024, dtype=torch.bfloat16, device="xdna")
print(x.device, x.cpu()[:4])
```

## What is implemented

The current backend uses PyTorch `PrivateUse1` renamed to `xdna` and provides:

- `torch.device("xdna")` / `torch.xdna`;
- XRT-BO-backed PyTorch storage;
- a caching XRT allocator;
- host/device coherency tracking;
- CPU execution directly over mapped XRT storage for unsupported operations;
- BF16 rectangular GEMM forward/dX/dW on XDNA2;
- BF16 3x3 Conv forward/dX families for validated training geometries;
- channels-last layout preservation through important compatibility operations;
- stride-aware view support for non-contiguous tensors when PyTorch can represent the view without copying;
- native mapped-storage `add.out` support for autograd gradient accumulation, avoiding a large generic CPU-fallback boundary;
- mapped-storage out-variant coverage for common AdamW/elementwise primitives (`mul`, `div`, `sqrt`, `addcmul`, `addcdiv`, `lerp`, `sigmoid`, `sub`);
- mapped-storage `addmm.out` and `bmm.out` compatibility paths for Linear/attention shapes not yet covered by the fixed NPU GEMM family;
- CPU `dW` overlapped with dependency-critical NPU `dX` when that is faster;
- batch-4 CPU `dW` reduction split into batch-1 oneDNN calls by default, reducing shared LPDDR/fabric pressure during NPU `dX` overlap;
- full-frame batch-4 decoder/residual programs for the current Strix training path.

The bundled full-frame family is enabled automatically. The current Strix package also enables validated resident-weight / queue-pipelined fast paths for the large decoder dX family and conv11 dX when their bundled artifacts are present. Disable them independently for A/B debugging with:

```bash
XDNA_ENABLE_FAST_FULLFRAME=0 python ...
XDNA_ENABLE_FAST_CONV11=0 python ...
```

Disable the full-frame backend entirely with:

```bash
XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_CONV=0 python ...
```

For batch-4 heterogeneous Conv backward, the CPU weight-gradient reduction is split into batch-1 oneDNN calls by default. This preserves oneDNN's AVX-512 BF16 compute kernel while reducing the working set and shared memory-fabric pressure seen by concurrent NPU dX. Override or disable it with:

```bash
XDNA_CPU_DW_BATCH_CHUNK=0 python ...   # disable
XDNA_CPU_DW_BATCH_CHUNK=2 python ...   # alternate chunk size
```

An optional virtual concat + nearest-upsample representation is available:

```bash
XDNA_LAZY_DECODER_VALUES=1 python ...
```

It avoids materializing large decoder intermediates when the following supported Conv can consume the virtual value. It is intentionally not the default yet because its end-to-end gain is smaller and more variable than the core path.

## Current measured checkpoint

On the development Ryzen AI 9 365 / Strix machine, XRT 2.21.75, BF16, 16 CPU threads, performance power profile, the real full-frame DRIVE `MyNet(3,1,7)` training+validation loop has recently measured approximately:

```text
CPU BF16:                              0.151 - 0.155 it/s
XDNA legacy full-frame path:           ~0.200 - 0.201 it/s
XDNA q4 resident full-frame + conv11:  ~0.210 - 0.212 it/s
Mean speedup vs CPU:                   ~1.38x end-to-end
```

In the latest package-level ABBA bracket, the q4 resident full-frame path alone averaged about 0.2038 it/s; adding the validated conv11 q4 dX path averaged about 0.2111 it/s. The large 256→128 @ 584 decoder dX itself improved from roughly 615 ms to roughly 469 ms, while preserving BF16 numerical agreement with the CPU reference.

A later same-package ABBA test of the batch-split CPU dW scheduler measured 0.20295 it/s enabled versus 0.19823 it/s disabled (+2.4% throughput in that run). On the isolated 256→128 @ 584 backward region, the same change reduced median wall time from about 659 ms to about 621–623 ms while keeping the NPU dX path unchanged and preserving BF16 training loss.

The next generic fallback audit found seven large `aten::add.out` calls in backward costing about 60–66 ms per step through the generic fallback. Registering a mapped-storage `add.out` implementation removed that fallback; a short interleaved NIS check measured roughly 0.2117 it/s with the native overload versus 0.2053 it/s without it, with unchanged loss.

Applying the bounded rolling-DMA schedule to the non-resident forward streams improved fwd292 from about 124 ms to about 97 ms and fwd584 from about 486 ms to about 382 ms without changing the universal XCLBIN. A clean-fork NIS ABBA measured about 0.21572 it/s with the rolled forward streams versus about 0.20995 it/s with the previous streams (+2.7%), with unchanged loss.

The metric is the notebook metric:

```python
len(train_loader) / (train_time + validation_time)
```

with training batch size 4 and validation batch size 2.

These numbers are hardware/workload-specific checkpoints, not a general performance claim.

## Design rules

The backend deliberately avoids several XDNA2 performance anti-patterns discovered during development:

- do not promote an NPU primitive from kernel-only timing;
- do not materialize im2col matrices;
- preserve useful physical layouts across producer/consumer regions;
- overlap CPU and NPU only when lifetime and memory-bandwidth costs are measured;
- do not retain giant monotonic XRT arenas merely to reduce allocation calls;
- prefer bounded resident physical program families over per-op program switching;
- measure the complete training loop after every promoted optimization.

See:

- `docs/xdna-device-architecture.md`
- `docs/fullframe-conv-backward-plan.md`

## Experimental patch-40 Conv output layout

The B40 256→128 Conv family emits a native `[Y,X,B,C]` layout. Its default
zero-copy, strided PyTorch view is efficient for some producers, but **CPU
BatchNorm forward/backward can be substantially slower** on these unusual
strides. This is not an XRT transfer issue; zero-copy does not imply cheap CPU
execution. The backend therefore converts to contiguous NCHW by default;
to keep the raw layout for an NPU-to-NPU consumer:

```bash
XDNA_CONV_OUTPUT_NCHW=0 python training.py
```

The conversion reorders each matching NPU Conv output exactly once inside
XDNA mapped storage; the data remain on the logical `xdna` device.
Consumer/layout-aware graph partitioning should eventually replace this
global switch.

On Ryzen AI 9 365, the **real NIS RGB style-transfer** training loop (MyNet
3→3, 7 residual blocks, 36×36 patches, B40, VGG perceptual loss, Adam)
measured **598.3 → 510.1 ms per training step** for `XDNA_CONV_OUTPUT_NCHW=0 → 1`
in two reversed-order runs at 10 Torch threads and 4 packing threads.
The CPU BF16 comparison was 243.2 ms/step. This experiment used actual
training images but only the first validation image, avoiding the original
script's unnecessary preload of 546 complete test frames. Results are located
in `torch-xdna-train/results/maintained-fork-layout-ab-20261010` on the
development machine. These numbers are workload-specific and *not* a general
NPU speedup claim.

Physical tests showed that the Conv outputs are bitwise identical across
layouts and that NCHW BatchNorm results and input/weight gradients are
considerably closer to the CPU BF16 reference. Run the hardware smoke test
with `python tests/run_conv_output_layout.py`, or with `pytest` installed,
`XDNA_TRAIN_PHYSICAL=1 pytest tests/test_conv_output_layout.py`.

A second real-script ABBA compared conversion at the CPU BatchNorm consumer
instead of the NPU Conv producer. It measured 611.2 ms/step with no
conversion, 526.3 ms/step with conversion inside BN, and **496.3 ms/step
with a single producer-side conversion**. Consumer-side repacking therefore
remains experimental, not part of this backend's default dispatch. Results:
`torch-xdna-train/results/bn-consumer-layout-ab-20261010`.

For *diagnostics only*, `XDNA_ENABLE_PATCH_CONV=0` disables registration of
the small-patch resident Conv family (default `1`). With NPU Conv enabled,
the same real script ran at 517.6 ms/step versus 710.7 ms when its supported
Conv operations were forced to mapped-CPU fallback, confirming that the NPU
kernels are still beneficial. See
`torch-xdna-train/results/maintained-fork-conv-gate-ab-20261010`.

## conv3x3w: general 3x3 convolution family

`xdna_train/_designs/conv3x3w.py` (forward and dX) and `conv3x3w_dw.py` (dW)
run any 3x3 / stride-1 / pad-1 convolution with width <= 36 on the full
array: bf16 activations with bfp16 weights/operands, ~1% relative error
against FP32 (CPU BF16 is ~0.2%). A 300-step NIS training run with every
eligible forward, dX and dW in this arithmetic tracked BF16 training within
noise (`benchmarks/bfp16_training_gate.py`).

- One xclbin per family; instruction streams per (groups, padded width,
  channel chunks[, pass]) are compiled on first use in a niced background
  process (~7 s each) and cached under `~/.cache/torch-xdna2/<family>/<source
  hash>/`. Until a stream exists the convolution takes the previous path.
  `XDNA_IRON_PYTHON` names the mlir-aie (IRON) Python used for builds.
- dW lives in a second xclbin; a hardware-context switch costs ~2.6 ms, so NPU
  dWs are deferred to the gradient's first reader (normally the optimizer)
  and run back to back. dWs below 1 GMAC stay on the CPU, overlapped with
  the NPU dX chain.
- Switches: `XDNA_CONV3X3W=0`, `XDNA_CONV3X3W_DW=0`,
  `XDNA_CONV3X3W_MIN_MACS`, `XDNA_CONV3X3W_DW_MIN_MACS`,
  `XDNA_CONV3X3W_WEIGHT_CACHE=0`, `XDNA_CONV3X3W_SYNC_BUILD=1` (build before
  the first call, for tests).
- Tests: `tests/run_conv3x3w_pack.py`, `tests/run_conv3x3w_torch.py`,
  `tests/run_conv3x3w_hw.py`, `tests/run_conv3x3w_dw_hw.py`.

NIS style transfer (MyNet(3,3,5), B40, 36x36, VGG loss, Adam), Ryzen AI 9 365
on AC, quick-mode ABBA (`benchmarks/nis_style_transfer.py`): CPU BF16 171.4
ms/step, XDNA 147.7 ms/step (1.16x); without NPU dW 167.7 ms/step.

## Compatibility

Currently tested on:

- Linux / `amdxdna`;
- XDNA2 / Ryzen AI Strix;
- XRT 2.21.75;
- PyTorch 2.14.1;
- Python 3.12;
- BF16 training.

The package pins the supported PyTorch range to 2.14.x because the native `PrivateUse1` ABI is version-sensitive. The extension is rebuilt for each Python/PyTorch/source fingerprint.

## Training coverage suite

The installed `torch-xdna2-coverage` command runs small real training motifs rather than inference-only operator probes. It currently covers MLP, residual CNN, U-Net-like decoder, Transformer, Mobile/depthwise, embedding/MLP, and a `nis-mini` workload. `nis-mini` preserves the NIS MyNet channel widths, all seven residual blocks, both cat/nearest-upsample decoder stages, final 1x1 Conv, sigmoid+BCE, and Adam while shrinking only the spatial input from 584x564 to 72x68. Each result reports full-step time, overload-aware generic CPU fallbacks, and XDNA handlers that still execute CPU math over mapped XRT storage.

```bash
torch-xdna2-coverage --motif all --steps 2 --threads 8
torch-xdna2-coverage --motif nis-mini --steps 5 --threads 16
torch-xdna2-coverage --motif transformer --json
```

Use this suite as an admission test for generic backend work: an optimization should improve the relevant motif and should not regress unrelated motifs before it is promoted. On the development Strix machine, the first `nis-mini` checkpoint measured about 96.6 ms/step through XDNA versus about 73.0 ms/step on CPU BF16; roughly 75 ms/step of the XDNA run was generic-shape Conv forward/backward still executing on CPU over mapped XRT storage, making it a fast target for broadening NPU Conv support.

## Development

Build a wheel:

```bash
uv build
```

Install a local checkout:

```bash
uv pip install --no-deps .
```

Run the import-only test:

```bash
pytest
```

Hardware check:

```bash
python -m xdna_train
```

## License

Project code is MIT licensed. Bundled/generated third-party components retain their own licenses; see `THIRD_PARTY_NOTICES.md` and `LICENSES/`.
