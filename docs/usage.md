# Usage and troubleshooting

Start with the [installation and training example](../README.md).

## Installation options

Install a PyTorch 2.14.x build suited to your system first. Use Python 3.12 to 3.14;
Python 3.12 and PyTorch 2.14.1 are the tested combination. The package also depends
on Ninja for compiling its native extension.

See [Arch, Ubuntu and Fedora instructions](system-dependencies.md).

The system needs the `amdxdna` driver, XRT headers and runtime, and a C++20 compiler
with OpenMP. On the development Arch Linux machine the packages are `xrt-base`, `xrt-npu`,
`amdxdna-driver` and `xrt-plugin-amdxdna` (see the [Arch recipe](system-dependencies.md#arch-linux)).
These are system prerequisites, not Python dependencies.
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

An explicit `XRT_INCLUDE_DIR` or `XRT_LIB_DIR` is used as given. If it is wrong,
the search does not fall back to other locations. When these variables are unset,
the search also checks the directories named by `XRT_ROOT` and `XILINX_XRT`
(`include/` for headers; `lib/` and `lib64/` for the library, see
`xdna_tools/diagnostics.py` lines 76-86 and 106-110). It then checks `/usr/include`,
`/usr/lib`, `/usr/lib64`, `/usr/lib/x86_64-linux-gnu` and `/opt/xilinx/xrt`.

The C++ compiler is taken from `CXX` when it is set. `CXX` may contain extra words,
such as `ccache g++`. The doctor checks the programs named there and reports an
error if one is missing (`xdna_tools/diagnostics.py` lines 152-170). The native
build records `CXX` in its cache key (`xdna_train/_build.py` line 39). When `CXX` is
unset, the doctor looks for `c++` and then `g++`.

### Convolution compiler

General 3x3 convolution requires IRON and Peano. When no existing compiler is
configured, the backend automatically prepares a private Python 3.12 environment
on its first successful initialization. This works independently of an application
using Python 3.12, 3.13 or 3.14. The initial download is about 410 MB plus Python if
needed. Setup prints its log path and waits for the compiler before returning.
It never runs sudo or installs packages into the training environment.

The package pins MLIR-AIE 1.3.4 and Peano 21.0.0.2026062301+cb664e8c, together with
all their Python dependencies. Downloads must match the included SHA256 hashes;
no LLVM source build is needed. The managed environment lives in
`~/.cache/torch-xdna2/toolchains/<lock-hash>/env/`. `XDG_CACHE_HOME` changes the
cache parent; `TORCH_XDNA2_CACHE` overrides the entire backend cache root.

```bash
torch-xdna2-setup          # Optional: prepare the compiler before the application
torch-xdna2-setup --check  # Read cache status only; no network or backend import
torch-xdna2-setup --retry  # Explicit retry after fixing a failed setup
```

Setup is serialized across processes. An interrupted environment is never marked
ready. A failed automatic attempt emits a warning, retains its `setup.log` and
falls back to CPU convolution. Repeated automatic attempts have a 10-minute
cooldown; explicit `--retry` bypasses it. Restart the training process after
repairing compiler setup so the families are configured again.

For offline or administrator-managed environments, set `XDNA_AUTO_SETUP=0` or
`UV_OFFLINE=1` before import. Neither removes an existing cached compiler. To use
your own compiler instead:

```bash
export XDNA_IRON_PYTHON=/path/to/mlir-aie-env/bin/python
```

An explicit value takes precedence over automatic setup, and an empty value
disables compilation and setup. Without an explicit override, the backend uses the
first of these that exists:

1. the managed compiler prepared by `torch-xdna2-setup`
2. `mlir-aie-env` in the parent directory of the repository checkout
3. `~/mlir-aie-env/bin/python`
4. `~/ironenv/bin/python`

If none exists, automatic setup runs unless `XDNA_AUTO_SETUP=0` or offline mode is set.

A new convolution shape can still require another instruction stream. Those
streams compile on demand and are cached separately; while they build, the
operation uses CPU fallback. Managed streams are keyed by compiler lock and source
hash, so changing the compiler cannot reuse an incompatible stream. Build errors
print an actionable warning and identify the family's `build.log`.

Bundled matrix and full-frame programs need no IRON compilation. The driver,
XRT and host C++ compiler remain [system prerequisites](system-dependencies.md).

## Check a failed installation

Start with `torch-xdna2-doctor` in the application environment. It checks local
dependencies without importing PyTorch or opening the NPU. Then run:

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

## Fused regions

Optional host-side fusions that remove CPU passes and intermediate tensors. Convolutions still dispatch
through the normal backend (NPU or CPU), the fused part is native CPU code. They compute in BF16 with fewer
rounding steps than the eager graph, so results differ slightly (loss curves over 400 real steps matched within 0.1%
per 40-step window). `XDNA_REGIONS=0` disables them.

- `regions.fuse_perceptual_loss(loss_module, features_attr="perceptual_vgg")`: for a loss whose
  `compute_perceptual_loss` runs `getattr(loss_module, features_attr).model.features` (a `Sequential` of
  `Conv2d`, `ReLU`, `MaxPool2d(2)`), compares the layers listed in `.feature_layers` and uses frozen weights.
  Every call re-checks the layers (exact types, conv attributes, hooks, parametrizations, parameter identity,
  `requires_grad`); anything that no longer matches runs the module's own method. Measured: NIS step 107 to 93 ms.
- `regions.fuse_norm_activation(bn, act)`: declare that `act` (`nn.ReLU` or `nn.LeakyReLU`) is applied directly
  to the output of `bn` (`nn.BatchNorm2d`). Used only above 4 MB per tensor (`XDNA_REGION_NORM_ACT_MIN_BYTES`).
  More accurate than eager BF16 on large layers; the full-step gain is small (about 1-3 ms, within noise).

A fused model can be `copy.deepcopy`'d (copy a paired norm and activation together) and its `state_dict` is
unchanged; `torch.save(model)` of the whole object is not supported. The regions are once-differentiable.

## Deferred gradients and timing

NPU convolution weight gradients can be returned before their computation runs.
The first reader, normally the optimizer, executes the pending work. A gradient
freed without being read can be discarded without running it.

`torch.xdna.synchronize()` does not force these deferred gradients. Include the
optimizer step when timing training. For a backward-only benchmark, explicitly
read every gradient being measured, for example with `.cpu()`, and account for
that readback cost. See [benchmark methodology](performance.md#reproducing-the-measurements).

`XDNA_ASYNC_DW` (default `1`) starts the deferred weight-gradient passes earlier. Once the backend has seen two training steps with the same number K of NPU input-gradient (dX) calls in backward, a helper thread starts the deferred dW passes right after the K-th dX call of the next steps, so they overlap the CPU work that remains in backward. The first two steps, and models whose NPU dX count per step is not stable, keep the lazy behaviour. To check whether it is active, call `xdna_train.async_dw_stats()`; it returns `enabled`, `kicks` (helper wake-ups), `started_by_helper` (jobs the helper started) and `dx_per_step` (the learned K, 0 until two steps agree). Set `XDNA_ASYNC_DW=0` before import to turn it off. A weight-gradient job that the helper thread already started is awaited when its gradient is read.

## Further reading

[Backend architecture](xdna-device-architecture.md) describes mapped tensor
storage, CPU fallback and NPU dispatch. [Development](development.md) lists
correctness tests and coverage tools.
