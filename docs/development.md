# Development

[Back to the README](../README.md)

## Build a local checkout

Use the [runtime prerequisites](../README.md#requirements) and a Python environment
with the supported PyTorch build and Ninja installed.

```bash
uv build
uv pip install --no-deps .
python -m xdna_train
```

`uv build` requires the uv tool. The native extension is compiled against the
active PyTorch ABI when loaded. `--no-deps` preserves the installed PyTorch build
but skips dependency installation and resolution.

## Setup tests without an NPU

The setup helpers live in `xdna_tools`, separate from the backend's auto-registering
import. These standard-library tests mock missing packages, paths and downloads:

```bash
python -m unittest discover -s tests -p 'test_setup_diagnostics.py'
python -m unittest discover -s tests -p 'test_toolchain_setup.py'
python -m unittest discover -s tests -p 'test_setup_integration.py'
```

An installed wheel must contain the compiler lock file and the doctor/setup entry
points. Test the isolated managed compiler on hardware before changing its pins.

## Correctness tests

From the repository root on an XDNA2 machine:

```bash
PYTHONPATH=. python tests/quick_check.py
```

This trains a small convolution network on CPU and XDNA from the same seed, checks
loss agreement and actual NPU calls, then runs the standalone regression scripts.
They cover views, tensor/storage semantics, matrix multiplication, deferred writes,
weight-cache invalidation, recurrent-layer gradients, packing and runtime behavior.
The suite requires IRON or the relevant cached convolution programs.

Run NPU tests one at a time. Concurrent processes can exhaust hardware contexts.
For a focused convolution check:

```bash
PYTHONPATH=. python tests/run_conv3x3w_torch.py
```

Other [test scripts](../tests/) cover hardware kernels, weight-gradient dispatch,
output layouts and cost policy. Some force specific paths and compile new programs.

With pytest installed:

```bash
PYTHONPATH=. python -m pytest
XDNA_TRAIN_PHYSICAL=1 PYTHONPATH=. python -m pytest
```

The first command leaves hardware-marked tests skipped. Importing the package can
still attempt native compilation and device registration; an import test is not
proof that hardware initialization succeeded.

## Training coverage

The installed coverage command trains smaller MLP, residual CNN, U-Net-like,
Transformer, depthwise, embedding and NIS-style motifs:

```bash
torch-xdna2-coverage --motif all --steps 2 --threads 8
torch-xdna2-coverage --motif nis-mini --steps 5 --threads 16
torch-xdna2-coverage --motif transformer --json
```

`nis-mini` keeps the NIS graph while reducing its image size to 72x68. Many
convolutions then run on the CPU, so it checks compatibility at that size rather
than reproducing full-frame acceleration. The command reports whole-step timing,
CPU fallback operations and mapped-CPU handlers.

For backend changes, compare the affected workload and unrelated motifs. Check
numerical agreement and dispatch counts as well as wall time. See
[performance methodology](performance.md#reproducing-the-measurements).

## Architecture and reference

- [Runtime behavior](usage.md#runtime-behavior): import effects and user-visible controls.
- [Device architecture](xdna-device-architecture.md): tensor storage, coherency and dispatch.
- [Environment variables](MODES_AUDIT.md): defaults, overrides and removed experiments.
- [Optimization log](OPTIMIZATION_LOG.md): historical measurements and decisions.
- [Full-frame backward design](fullframe-conv-backward-plan.md): the specialized decoder path.

The main implementation lives in [device.py](../xdna_train/device.py),
[torch_xdna_device.cpp](../xdna_train/_native/torch_xdna_device.cpp) and
[conv_pack.cpp](../xdna_train/_native/conv_pack.cpp). Kernel generators are in
[_designs](../xdna_train/_designs/). The older
[agent handoff](AGENT_HANDOFF_2026-10-10.md) records local research paths and
historical state; check current source before relying on its settings.

For a contribution, include a small regression test for changed behavior and
reproducible full-step measurements for performance changes. Report failures with
a minimal example, input shapes/dtypes, environment settings and the output of
`python -m xdna_train`.
