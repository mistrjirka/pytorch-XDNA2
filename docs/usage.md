# Usage and troubleshooting

Start with the [installation and training example](../README.md).

## Installation options

Install a PyTorch 2.14.x build suited to your system first. Use Python 3.12 to 3.14;
Python 3.12 and PyTorch 2.14.1 are the tested combination. The package also depends
on Ninja for compiling its native extension.

The system needs the `amdxdna` driver, XRT headers and runtime, and a C++20 compiler
with OpenMP. On the development Arch Linux machine the XRT packages are `xrt` and
`xrt-plugin-amdxdna`. These are system prerequisites, not Python dependencies.
See the upstream [driver](https://github.com/amd/xdna-driver) and
[XRT](https://github.com/Xilinx/XRT) documentation for system setup.

### pip or a uv project

```bash
python -m pip install "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

For a uv-managed project:

```bash
uv add "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

To reinstall only this backend while keeping an existing PyTorch build:

```bash
uv pip install --reinstall --no-deps "git+https://github.com/mistrjirka/pytorch-XDNA2.git"
```

`--no-deps` skips dependency installation and resolution. Ensure PyTorch 2.14.x
and Ninja are already installed; the backend has no separate runtime version check.

### XRT outside the default paths

The compiler must find `xrt/xrt_device.h`, and the linker must find
`libxrt_coreutil.so`. Set these before importing the package:

```bash
export XRT_INCLUDE_DIR=/path/to/xrt/include
export XRT_LIB_DIR=/path/to/xrt/lib
python -m xdna_train
```

### Convolution compiler

General 3x3 convolution programs are built with [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie)
in a separate Python environment:

```bash
export XDNA_IRON_PYTHON=/path/to/mlir-aie-env/bin/python
```

The backend also looks in `~/mlir-aie-env/bin/python` and `~/ironenv/bin/python`.
An empty `XDNA_IRON_PYTHON` disables on-demand compilation. Already cached programs
can still be used. Bundled matrix and full-frame programs do not need an IRON build.

A new convolution shape can require another program. While it builds, that
convolution uses a CPU path. The default cache is `~/.cache/torch-xdna2/`;
`TORCH_XDNA2_CACHE` changes its root. Historical builds took about 7 seconds per
instruction stream on the development machine; compilation time varies.

## Check a failed installation

Run this in the same environment as the application:

```bash
TORCH_XDNA2_BUILD_VERBOSE=1 python -m xdna_train
```

The command registers the device and prints the PyTorch version, availability,
native-library path and allocator statistics. It exercises device setup, not every
NPU kernel. `torch-xdna2-build` prints the native-library path; use
`python -m xdna_train` for device-status output.

If import warned that registration failed, diagnose that warning before creating
XDNA tensors. Check the driver and XRT installation, compiler output, supported
PyTorch version, and whether another process is using the NPU. Changing to another
Python environment can also select a different installed backend.

## Check which paths a workload uses

A tensor reporting `device='xdna'` may still be processed by the CPU. CPU fallback
keeps the logical device while running supported CPU kernels over mapped memory.

Enable profiling before starting the application:

```bash
XDNA_PROFILE_MAPPED_CPU=1 XDNA_PROFILE_FALLBACK=1 python train.py
```

After a complete training step, inspect the counters in that process:

```python
import xdna_train

print(xdna_train.mapped_cpu_stats())
print(xdna_train.fallback_stats())
print(xdna_train.autotune_result())
```

Entries such as `npu_conv3x3w.fwd.run`, `.dx.run` and `.dw.run` show actual
convolution dispatches. Other entries report work done on the CPU. Profiling can
add overhead, so disable it for final timing runs.

## Tuning

The step autotuner measures the first optimizer using XDNA parameters and chooses
among these modes:

| Mode | Eligible computation |
|---|---|
| `0` | CPU paths over XDNA storage |
| `1` | NPU paths except NPU weight gradients |
| `2` | All supported NPU paths, subject to shape and cost checks |

Set `XDNA_AUTOTUNE_VERBOSE=1` to print its choice. `autotune_result()` returns
`None` before a decision or when tuning is disabled. The current tuner discards
measurements affected by pending builds and retries up to five times. It does not
continually retune when a model or batch shape changes.

For a controlled comparison, choose a mode before starting the application:

```bash
XDNA_NPU_MODE=1 python train.py
```

`xdna_train.set_npu_mode(1)` changes the mode in an existing process and stops its
autotuner. `XDNA_AUTOTUNE=0` leaves all NPU paths eligible without step tuning.
Mode 0 still has XDNA storage and dispatch overhead; compare against a model on
`torch.device("cpu")` for a native CPU baseline.

General 3x3 weight gradients below 3 billion multiply-accumulates stay on the CPU
by default. `XDNA_CONV3X3W_DW_MIN_MACS` controls that threshold. It is a cost
heuristic, not a guarantee that the faster device is selected for every shape.

See [the environment-variable reference](MODES_AUDIT.md) for feature switches,
host-thread counts, buffer-cache limits and removed settings. Set variables before
import unless a public runtime setter is documented. Removed variables produce a
warning and have no effect.

## Runtime behavior

Importing `xdna_train` builds/loads the native bridge and registers `xdna` through
PyTorch's `PrivateUse1` backend slot. It cannot share that slot with another backend
in the same process. Registration can fail after some initialization has occurred;
do not assume a failed import attempt leaves all PyTorch settings untouched.

Successful registration also changes these behaviors:

- Module moves to XDNA temporarily enable parameter swapping, preserving the
  `Parameter` objects an optimizer may already reference. This is scoped around
  `Module.to`, `.xdna()` and `.to_empty()` calls targeting XDNA.
- Eligible XDNA Adam, AdamW and SGD parameter groups use fused kernels when neither
  `fused` nor `foreach` was explicitly chosen. Optimizer hooks also maintain
  parameter versions and run the autotuner. `XDNA_FUSED_OPTIMIZER=0` disables the
  automatic fused choice.
- A process-wide CPU allocator wrapper lets CPU fallback results use NPU-visible
  storage. Other allocations pass through. `XDNA_ADOPT_HOST_OUTPUTS=0` disables it.
- Backward runs on the calling thread for the thread that registered the device.
  This also affects unrelated backward calls on that thread. Set
  `XDNA_AUTOGRAD_THREAD=1` before import to retain PyTorch's default threading.

## Deferred gradients and timing

NPU convolution weight gradients can be returned before their computation runs.
The first reader, normally the optimizer, executes the pending work. A gradient
freed without being read can be discarded without running it.

`torch.xdna.synchronize()` does not force these deferred gradients. Include the
optimizer step when timing training. For a backward-only benchmark, explicitly
read every gradient being measured, for example with `.cpu()`, and account for
that readback cost. See [benchmark methodology](performance.md#reproducing-the-measurements).

## Further reading

[Backend architecture](xdna-device-architecture.md) describes mapped tensor
storage, CPU fallback and NPU dispatch. [Development](development.md) lists
correctness tests and coverage tools.
