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
- CPU `dW` overlapped with dependency-critical NPU `dX` when that is faster;
- full-frame batch-4 decoder/residual programs for the current Strix training path.

The bundled full-frame family is enabled automatically. Disable it for debugging with:

```bash
XDNA_ENABLE_EXPERIMENTAL_FULLFRAME_CONV=0 python ...
```

An optional virtual concat + nearest-upsample representation is available:

```bash
XDNA_LAZY_DECODER_VALUES=1 python ...
```

It avoids materializing large decoder intermediates when the following supported Conv can consume the virtual value. It is intentionally not the default yet because its end-to-end gain is smaller and more variable than the core path.

## Current measured checkpoint

On the development Ryzen AI 9 365 / Strix machine, XRT 2.21.75, BF16, 16 CPU threads, performance power profile, the real full-frame DRIVE `MyNet(3,1,7)` training+validation loop has recently measured approximately:

```text
CPU BF16:               0.149 - 0.154 it/s
XDNA BF16 core path:    ~0.183 it/s
XDNA + virtual decoder: 0.185 - 0.186 it/s in matched ABBA runs
```

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

## Compatibility

Currently tested on:

- Linux / `amdxdna`;
- XDNA2 / Ryzen AI Strix;
- XRT 2.21.75;
- PyTorch 2.14.1;
- Python 3.12;
- BF16 training.

The package pins the supported PyTorch range to 2.14.x because the native `PrivateUse1` ABI is version-sensitive. The extension is rebuilt for each Python/PyTorch/source fingerprint.

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

No open-source license has been selected yet.
