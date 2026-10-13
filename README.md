# torch-xdna2

Train BF16 PyTorch models on AMD Ryzen AI XDNA2 (Strix) using `torch.device("xdna")`.
The backend offloads selected convolutions and matrix multiplies to the NPU.
Other supported operations run on the CPU using shared, host-visible memory.

**Alpha:** operator coverage and performance depend on your model. CPU fallback
does not guarantee that every PyTorch operation works.

## Requirements

- Linux with a working `amdxdna` driver and an XDNA2 / Strix NPU.
- AMD XRT runtime and development headers, plus a C++20 compiler with OpenMP.
- Python 3.12 to 3.14 and PyTorch 2.14.x.

Tested on a Ryzen AI 9 365 with XRT 2.21.75, PyTorch 2.14.1 and Python 3.12.
The Python package does not install the driver or XRT.

## Install

In a virtual environment with your preferred PyTorch 2.14.x build:

```bash
uv pip install "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

For pip, replace `uv pip install` with `python -m pip install`.

The first import checks system dependencies, builds the native extension, and
prepares a private [IRON / Peano compiler](docs/usage.md#convolution-compiler)
for general 3x3 convolution. Its pinned wheels download about 410 MB once, plus
Python 3.12 if needed. Your training environment is unchanged; no compiler
activation or path configuration is needed.

Install the driver/XRT and build tools using the [Arch, Ubuntu or Fedora guide](docs/system-dependencies.md).
Missing prerequisites produce actionable warnings on import. `torch-xdna2-doctor`
checks them without loading the backend or downloading anything. To disable
automatic downloads, set `XDNA_AUTO_SETUP=0` before import.

Check the installation:

```bash
python -m xdna_train
```

Look for `xdna_available=True`. See [setup and troubleshooting](docs/usage.md)
for offline use, XRT paths and build failures.

## Train

A complete training step with synthetic inputs:

```python
import torch
import xdna_train

xdna_train.register_xdna_device()
device = torch.device("xdna")
model = torch.nn.Conv2d(128, 128, 3, padding=1).bfloat16().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(16, 128, 36, 36, dtype=torch.bfloat16, device=device)
target = torch.zeros_like(x)

optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
loss.backward()
optimizer.step()
print("loss:", loss.item())
```

For an existing model, move its parameters and inputs to `xdna` and use BF16 for
floating-point inputs. Keep class labels and token indices integer. The normal
forward, loss, backward and optimizer calls remain the same.

## Performance and limits

The [recorded 12-workload benchmark](docs/performance.md) on Ryzen AI 9 365
measured **1.21x CPU BF16 throughput on average** (geometric mean), with individual
results from 0.97x to 2.03x. It used AC power and 10 CPU threads (the PyTorch default on this machine). Gains include CPU
execution improvements; they do not all come from NPU computation.

NPU acceleration covers selected BF16 3x3 convolutions and fixed matrix shapes.
Small operations, attention, recurrent layers and most other matrix shapes use
the CPU. Measure your own workload after compilation and warmup.

- FP32 uses CPU paths. FP16 training and inference speed are not validated.
- General NPU convolutions use block floating-point arithmetic (BFP16), so results
  differ from CPU BF16. Check loss and gradients for your model.
- Concurrent NPU processes can exhaust hardware contexts and fail to initialize.
- Import installs allocator and optimizer hooks and changes backward threading
  on the importing thread. See [runtime behavior](docs/usage.md#runtime-behavior).

When benchmarking, include `optimizer.step()` or explicitly read the gradients.
`torch.xdna.synchronize()` alone does **not** execute deferred weight gradients.

## Documentation and help

- [Usage and troubleshooting](docs/usage.md): setup, CPU fallback, tuning and runtime behavior.
- [Performance](docs/performance.md): all measurements, tested workloads and reproduction commands.
- [Development](docs/development.md): tests, local builds and backend architecture.

For bugs, [open an issue](https://github.com/mistrjirka/pytorch-XDNA2/issues) with a
minimal example, tensor shapes and dtypes, hardware/software versions, and the
output of `python -m xdna_train`. Include power settings and full-step timings for
performance reports.

## License

[MIT](LICENSE). Bundled components retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES/).
