# XDNA1 bf16 GEMM + training test kit

Tests a plain bf16 matrix multiply on a Ryzen 7 8845HS / Phoenix-class NPU (npu1, 4 columns), in two parts:
1. **GEMM cases** (kernel microbenchmark): is it correct, how fast, how much CPU time and (on battery) energy, compared with
   `torch.mm` in bf16 on the CPU at the same shapes.
2. **End-to-end training** (the result that counts): a small MLP (3 layers, 1024 or 2048 wide) is trained with the NPU doing
   every matmul, forward and backward (`y = x@W`, `dX = dY@W^T`, `dW = X^T@dY`), against the identical training on the CPU.
   Loss curves, step time, CPU time per step, energy and a per-step breakdown are recorded for both.

The xclbins are prebuilt. Nothing is installed system-wide and nothing outside `results/` (and uv's cache) is written.

## (a) Prerequisites (Manjaro / Arch)

```
sudo pacman -S --needed xrt xrt-plugin-amdxdna linux-firmware python uv git
```
* `xrt` and `xrt-plugin-amdxdna` are in Arch `extra`. Whether Manjaro stable carries them (same version for both)
  must be checked first: `pacman -Si xrt xrt-plugin-amdxdna`. If they are missing, stop and tell us.
* Firmware comes from `linux-firmware` (`ls /lib/firmware/amdnpu/1502_00/`). Kernel >= 6.14 with the in-tree `amdxdna`
  module: `uname -r`, `modinfo amdxdna | head -3`. After `pacman -S` reboot once so the module and firmware are loaded.
* Your user must be able to open `/dev/accel/accel0` (normally group `render`): `ls -l /dev/accel/accel0`,
  otherwise `sudo usermod -aG render $USER` and log in again.
* `pyxrt` is installed only for the system python (`/usr/lib/python3.*/site-packages/pyxrt*.so`), so the venv must be created on
  `/usr/bin/python3` with `--system-site-packages`. `uv` is in Arch `extra`; the CPU-only torch wheel comes from the PyTorch index.

Setup (once):
```
git clone -b xdna1-test-kit --depth 1 https://github.com/mistrjirka/pytorch-XDNA2.git && cd pytorch-XDNA2/xdna1_kit
sha256sum -c --quiet SHA256SUMS
uv venv --python /usr/bin/python3 --system-site-packages kitvenv
uv pip install --python kitvenv numpy torch --index-url https://download.pytorch.org/whl/cpu
kitvenv/bin/python xdna1_kit.py --dry-run      # checks NPU, access, pyxrt, torch, files; loads nothing; must end with "--dry-run: ... check out"
```
(The one `uv pip install` takes numpy and the CPU-only torch from the PyTorch index, ~200 MB; a "Failed to hardlink" warning is harmless.
If it cannot find torch for your python, tell us which python version `/usr/bin/python3 --version` shows; the GEMM part still runs
with `--no-e2e --npu-only`.)
Close other NPU users (FastFlowLM, other XRT programs) and heavy applications. Leave the power profile as it is; the kit records it.

## (b) One run on AC (charger plugged in, about 10 min: GEMM cases, then the training test)
```
kitvenv/bin/python xdna1_kit.py --power ac
```

## (c) One run on battery (charger unplugged, battery above ~50 %, about 10 min)
Unplug, wait 10 seconds, then:
```
kitvenv/bin/python xdna1_kit.py --power battery
```
The kit refuses to start (exit 7, nothing run) if the machine is not in the requested power state. On battery the
kit integrates the battery gauge, which gives whole-system energy per call without root. (Optional, for package energy
on AC: `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj` before the AC run; it reverts at reboot.)

## (d) What to send back
The two files each run writes to `results/` (the `work_*` folders are not needed):
```
tar czf kit-results.tar.gz results/kit_results_*.json results/kit_summary_*.txt
```
Send `kit-results.tar.gz`. They contain kernel, firmware and XRT versions, `xrt-smi examine`, CPU, power state,
per-case correctness and timing, CPU baselines, the full training loss curves and per-step breakdowns, and for any failure
the full traceback and kernel-log lines.
If a case shows ERROR, also send the output of `sudo dmesg | tail -200`.

## If it stops early
| exit | meaning | fix |
|---|---|---|
| 3 | no NPU / `amdxdna` not loaded | `lsmod \| grep amdxdna`, `ls /dev/accel`; IOMMU must be enabled in BIOS |
| 4 | NPU is not npu1 | nothing loaded; `--cpu-only` still gives the CPU baseline |
| 5 | no access to `/dev/accel/accel0` | `sudo usermod -aG render $USER`, log in again |
| 6 | `pyxrt` not importable | the venv was not made with `uv venv --python /usr/bin/python3 --system-site-packages kitvenv`, or `xrt` is not installed for `/usr/bin/python3` |
| 7 | `--power` state wrong | plug / unplug the charger and run again |

Buffer allocation errors (case shows ERROR with "alloc"/"memory"): the `memlock` limit is too small. Run `ulimit -l unlimited`
in the same shell and retry; if that is refused, add `* - memlock unlimited` to `/etc/security/limits.d/99-kit.conf`, log in again.
`kitvenv/bin/python xdna1_kit.py --quick --power ac` is a shorter run (5 GEMM shapes, 60 training steps, ~3 min).
A failing GEMM case, a failing training run, or a crashed worker never aborts the others. Options: `--no-e2e` (GEMM only), `--e2e-only` (training only).

## What the training test does (the part we judge by)
* Model: 3 linear layers `1024 -> 1024 -> 1024 -> 1024` (GELU between them), batch 1024, MSE loss on a synthetic regression
  target (fixed random teacher, learnable), Adam, 300 steps; a second, heavier config `2048` wide, batch 2048, 60 steps.
  Because batch = width, **all 8 matmuls of a step (3 forward, 2 dX, 3 dW) have one shape** (1024^3, resp. 2048^3).
* On the NPU: forward `y = x@W`, `dX = dY@W^T`, `dW = X^T@dY` all run on the NPU as bf16 inputs with fp32 accumulation
  (fp32 output; the bf16-output xclbin is run too). Bias, GELU, loss, autograd bookkeeping and the Adam optimizer stay on the CPU in fp32.
  The transposed operands are handled two ways, both timed:
  * `npu-f32out`, `npu-bf16out`: one xclbin; a background thread transposes `W^T` and `X^T` on the host while the NPU runs.
  * `npu-f32out-npuT`: no host transposes; separate dX (B column-major) and dW (A column-major) xclbins read the operands as
    stored. This costs three NPU context switches per step, so this run also measures the switch cost on your NPU.
  The dX/dW xclbins of the 1024 config are also GEMM cases above (ids ending `_dx`, `_dw`).
* CPU baselines, same code, same init, same data: `cpu-bf16` (torch.mm on bf16 operands with fp32 master weights and Adam,
  i.e. what bf16 autocast does; the fair speed comparison) and `cpu-fp32` (loss reference), each at default OpenMP and under
  `OMP_WAIT_POLICY=PASSIVE`; the NPU run is repeated under PASSIVE too (its host ops are multithreaded as well).
* Reported per run: loss curve, final and held-out loss, step-0 gradient error against fp32 autograd, median/p90 step time,
  process CPU time per step, energy per step (battery or RAPL), and the breakdown matmul (NPU wait) / host copies+transposes+sync /
  optimizer / other CPU ops. Pass criteria printed in the summary: final loss within 5 % of CPU fp32, gradient error under 3 %
  (bf16-output 6 %).

Terms: `resident` = submit + wait only; `e2e` = copy in, run, copy out; `relL2` = relative L2 error against the fp32
product of the same bf16 inputs (f32-out: about 1e-6 or better; bf16-out: under 1 %, thresholds are in `manifest.json`).
