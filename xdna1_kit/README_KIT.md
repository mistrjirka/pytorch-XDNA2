# XDNA1 bf16 GEMM test kit

Tests a plain bf16 matrix multiply on a Ryzen 7 8845HS / Phoenix-class NPU (npu1, 4 columns): is it correct, how fast,
how much CPU time and (on battery) energy, compared with `torch.mm` in bf16 on the CPU at the same shapes.
The xclbins are prebuilt. Nothing is installed system-wide and nothing outside `results/` is written.

## (a) Prerequisites (Manjaro / Arch)

```
sudo pacman -S --needed xrt xrt-plugin-amdxdna linux-firmware python
```
* `xrt` and `xrt-plugin-amdxdna` are in Arch `extra`. Whether Manjaro stable carries them (same version for both)
  must be checked first: `pacman -Si xrt xrt-plugin-amdxdna`. If they are missing, stop and tell us.
* Firmware comes from `linux-firmware` (`ls /lib/firmware/amdnpu/1502_00/`). Kernel >= 6.14 with the in-tree `amdxdna`
  module: `uname -r`, `modinfo amdxdna | head -3`. After `pacman -S` reboot once so the module and firmware are loaded.
* Your user must be able to open `/dev/accel/accel0` (normally group `render`): `ls -l /dev/accel/accel0`,
  otherwise `sudo usermod -aG render $USER` and log in again.
* `pyxrt` is installed only for the system python, so make the venv with `--system-site-packages`.

Setup (once):
```
git clone -b xdna1-test-kit --depth 1 https://github.com/mistrjirka/pytorch-XDNA2.git && cd pytorch-XDNA2/xdna1_kit
sha256sum -c --quiet SHA256SUMS
python3 -m venv --system-site-packages kitvenv
kitvenv/bin/pip install numpy
kitvenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
kitvenv/bin/python xdna1_kit.py --dry-run      # checks NPU, access, pyxrt, files; loads nothing; must end with "--dry-run: ... check out"
```
Close other NPU users (FastFlowLM, other XRT programs) and heavy applications. Leave the power profile as it is; the kit records it.

## (b) One run on AC (charger plugged in, ~6 min)
```
kitvenv/bin/python xdna1_kit.py --power ac
```

## (c) One run on battery (charger unplugged, battery above ~50 %, ~6 min)
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
per-case correctness and timing, CPU baselines, and for any failure the full traceback and kernel-log lines.
If a case shows ERROR, also send the output of `sudo dmesg | tail -200`.

## If it stops early
| exit | meaning | fix |
|---|---|---|
| 3 | no NPU / `amdxdna` not loaded | `lsmod \| grep amdxdna`, `ls /dev/accel`; IOMMU must be enabled in BIOS |
| 4 | NPU is not npu1 | nothing loaded; `--cpu-only` still gives the CPU baseline |
| 5 | no access to `/dev/accel/accel0` | `sudo usermod -aG render $USER`, log in again |
| 6 | `pyxrt` not importable | venv without `--system-site-packages`, or `xrt` not installed for this python |
| 7 | `--power` state wrong | plug / unplug the charger and run again |

Buffer allocation errors (case shows ERROR with "alloc"/"memory"): the `memlock` limit is too small. Run `ulimit -l unlimited`
in the same shell and retry; if that is refused, add `* - memlock unlimited` to `/etc/security/limits.d/99-kit.conf`, log in again.
`kitvenv/bin/python xdna1_kit.py --quick --power ac` is a shorter run (5 shapes). A case failing never aborts the others.

Terms: `resident` = submit + wait only; `e2e` = copy in, run, copy out; `relL2` = relative L2 error against the fp32
product of the same bf16 inputs (f32-out: about 1e-6 or better; bf16-out: under 1 %, thresholds are in `manifest.json`).
