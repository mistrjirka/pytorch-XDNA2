# System dependencies

After the system prerequisites below are installed, use the [normal uv install](../README.md#install)
and run your program. On first use the backend prepares its own Python 3.12,
MLIR-AIE and Peano environment. It does not install system packages or use sudo.

Only the Arch development machine has been tested with this backend. The Ubuntu
and Fedora recipes below follow upstream instructions checked on 2026-10-13;
they are not claims of end-to-end testing on those distributions.

## Check before changing anything

```bash
torch-xdna2-doctor
# Machine-readable report; no downloads or NPU initialization:
torch-xdna2-doctor --json
```

The checker lists missing headers, libraries, tools and NPU access, with suggested
fixes. `python -m xdna_train` then builds/loads the backend and checks device setup.
A successful doctor report does not validate every kernel or detect all firmware
and XRT version mismatches.

A wheel install has no standard post-install execution hook. These warnings run
when the backend is imported or when you run the doctor, not during `uv pip install`.

## What the operating system must provide

Enable the NPU/IPU in BIOS and install a mutually compatible `amdxdna` driver,
firmware, XRT runtime, **XRT NPU plugin**, development headers and utilities.
An FPGA-only XRT package is insufficient. The [AMD driver repository](https://github.com/amd/xdna-driver)
recommends building XRT from its matched `xrt` submodule when using source builds.
Kernel and firmware requirements are those of the `amdxdna` driver; this package enforces no minimum (the doctor suggests Linux 6.14 or newer, see `xdna_tools/diagnostics.py` line 208), and the Ubuntu and Fedora commands in this file were not re-verified by the maintainer on 2026-10-13.

The native extension also needs a C++20 compiler with OpenMP, Ninja and headers for
the **Python interpreter running your program**. uv-managed Python includes its
headers; the distribution's `python3-dev` or `python3-devel` package is for its
system Python. The managed convolution compiler uses glibc 2.35 or newer.

Use signed modules or your distribution's module-signing/MOK procedure when
Secure Boot is enabled. Do not disable Secure Boot merely to follow this guide.
If a driver module or firmware changes, reboot before testing.

## Arch Linux

Install basic build dependencies:

```bash
sudo pacman -S --needed base-devel python ninja git git-lfs
```

The development machine uses packages named `xrt-base`, `xrt-npu`, `amdxdna-driver`
and `xrt-plugin-amdxdna`, built as in the recipe below. These names do not imply
availability in Arch's official repositories. For a reproducible
source installation, follow [AMD's Arch build steps](https://github.com/amd/xdna-driver#steps-to-create-release-build-packages-arch-linux).
Clone with submodules and Git LFS, inspect the dependency script, and build the
matched XRT and driver sources. Upstream provides these packaging recipes:

| Directory in `xdna-driver` | Build recipes |
|---|---|
| `xrt/build/arch/` | `PKGBUILD-xrt-base`, `PKGBUILD-xrt-npu` |
| `build/arch/` | `PKGBUILD-amdxdna-driver`, `PKGBUILD-xrt-plugin-amdxdna` |

After building with `makepkg`, install the resulting files, in this order:

```bash
# From the directory containing the generated XRT packages:
sudo pacman -U xrt-base-*.pkg.tar.zst xrt-npu-*.pkg.tar.zst
# From the directory containing the generated driver/plugin packages:
sudo pacman -U amdxdna-driver-*.pkg.tar.zst xrt-plugin-amdxdna-*.pkg.tar.zst
```

The kernel headers used for a DKMS build must match your kernel (`linux-headers`,
`linux-lts-headers`, or the equivalent for your installed kernel).

## Ubuntu 24.04

Basic build dependencies:

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-venv ninja-build software-properties-common
```

The [upstream MLIR-AIE guide](https://github.com/Xilinx/mlir-aie#install-from-upstream-packages-ubuntu-2404-with-linux-617)
documents AMD's PPA for **Ubuntu 24.04 with kernel 6.17 or newer**. Check `uname -r`
and use Ubuntu's supported kernel-update mechanism before proceeding if needed.
Adding this PPA trusts AMD's package repository:

```bash
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
sudo reboot
```

For other Ubuntu versions or unsupported kernels, use the matching
[AMD source-build recipe](https://github.com/amd/xdna-driver#build). Do not mix
PPA libraries with unrelated source-built plugins. The PPA's `pyxrt` bindings are
for Python 3.12; the managed compiler uses that version independently of your
application Python.

## Fedora

Basic build dependencies:

```bash
sudo dnf install gcc-c++ python3-devel libgomp ninja-build git git-lfs
```

AMD's current [Fedora build recipe](https://github.com/amd/xdna-driver#steps-to-create-release-build-packages-fedora)
lists Fedora 44 or newer. Use its dependency script and the matched `xrt` submodule
to build RPMs. A generic `xrt` RPM without the NPU plugin is not sufficient.
From the respective generated-package directories, install XRT before the plugin:

```bash
# In xdna-driver/xrt/build:
sudo dnf install ./Release/xrt-*.rpm
# In xdna-driver/build, after building the plugin:
sudo dnf install ./Release/xrt_plugin-*.rpm
```

Install the kernel development files appropriate to the running kernel when
building a module. The [non-Ubuntu upstream guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLinNonUbuntu.md)
also describes using an existing in-tree `amdxdna` driver with a userspace SHIM.
That driver must support the ioctls expected by the selected XRT/plugin; kernel
module presence alone is not a compatibility check.

## Device permissions and locked memory

```bash
ls -l /dev/accel/
xrt-smi examine
ulimit -l
```

If XRT is installed under `/opt/xilinx/xrt`, source its `setup.sh` to put `xrt-smi`
on your shell's PATH. The backend itself automatically searches the common
`/usr`, `/usr/lib64`, multiarch and `/opt/xilinx/xrt` include/library locations.
Explicit `XRT_INCLUDE_DIR` and `XRT_LIB_DIR` overrides remain available.

If the NPU node belongs to group `render`, add your user and log out/back in:

```bash
sudo usermod -aG render "$USER"
```

Use the group actually owning the node and the package's udev rules. Do not make
the device world-writable. Missing `/dev/accel/accel*` usually requires checking
BIOS settings, the driver and firmware before changing permissions.

Large NPU buffers can exceed `RLIMIT_MEMLOCK`. If the doctor warns or allocation
fails, an administrator can grant a suitable locked-memory limit to your user in
`/etc/security/limits.d/99-amdxdna.conf`, then log out and back in. Use a finite
limit appropriate to your workload; for example, 8 GiB is `8388608` KiB. Apply the
limit only to the intended user, not every account. Both soft and hard limits
may need changing. See [AMD's memory-limit notes](https://github.com/amd/xdna-driver#q-how-to-allocate-huge-size-bo).

Fedora desktop sessions may also inherit limits from `systemd --user`; changing
PAM limits alone may not suffice. Check the upstream Fedora note and adjust the
specific user service through your administrator, then restart that user's login
session. The doctor never changes groups, limits, modules or package repositories.

## Compiler downloads and offline machines

The compiler setup uses version-pinned, SHA256-checked binary wheels in a private
cache. The initial download is roughly 410 MB, plus Python when not available.
The training environment is not modified. No activation or compiler path is
needed. To prepare it explicitly, inspect its cache, or retry a failed download:

```bash
torch-xdna2-setup
torch-xdna2-setup --check
torch-xdna2-setup --retry
```

Set `XDNA_AUTO_SETUP=0` before importing to disable automatic installation.
`UV_OFFLINE=1` also prevents first-time downloads. Cached toolchains still work;
missing convolution programs use CPU fallback. An explicit `XDNA_IRON_PYTHON`,
including an empty value, takes precedence. See [usage](usage.md#convolution-compiler)
for cache locations, existing compiler environments and build logs.
