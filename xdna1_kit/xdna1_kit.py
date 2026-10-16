#!/usr/bin/env python3
"""XDNA1 (Phoenix / Hawk Point, mlir-aie device npu1) bf16 GEMM test kit.

ONE command:   python3 xdna1_kit.py            (add --quick for a fast pass)

What it does
  1. Reports device / driver / firmware / XRT / CPU / power setup.
  2. Refuses cleanly (no xclbin is loaded) when no NPU is present or when the NPU generation does not match
     the generation the xclbins in manifest.json were built for.
  3. NPU worker: for each case in manifest.json loads the prebuilt xclbin + instruction stream, checks the
     result against an fp32 reference of the same bf16 inputs, and times it:
        resident  = submit + wait only, operands already on the device (what run_gemm.py reports)
        e2e       = host prep (copy A,B into the buffers) + sync + submit + wait + sync back + read C
        profile   = median per-call split of the e2e path, plus the dispatch floor (tiny shape)
  4. CPU worker(s): torch bf16 mm at the same shapes, default OpenMP and OMP_WAIT_POLICY=PASSIVE variants.
  5. For every timed loop: wall time, process CPU time (getrusage user+sys) and system-wide busy CPU time
     (/proc/stat) per call, and package energy where available:
        RAPL  /sys/class/powercap/intel-rapl:0/energy_uj   (usually root-only: run with sudo, or see README)
        battery gauge /sys/class/power_supply/BAT*/power_now  (works unprivileged, only while discharging)
  6. Writes results/kit_results_<host>[_tag]_<time>.json and a text summary. Send both back.
     Failures (xclbin rejected, firmware error, crashes) are recorded with tracebacks and kernel-log excerpts and never abort the run.

The NPU worker needs: python with pyxrt + numpy (Arch/Manjaro: `pacman -S xrt xrt-plugin-amdxdna`, system python).
The CPU worker needs: python with torch (CPU build is enough). Both run with the python that starts the script
(a `--system-site-packages` venv sees pyxrt and torch); --python-npu / --python-cpu override.
No mlir-aie / IRON / Peano installation is needed: the xclbins are prebuilt.

Safe by construction: nothing is installed, no files outside --outdir are written, no sudo is called.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = 2

# PCI device ids of the amdxdna-supported NPUs -> mlir-aie device generation.
PCI_NPU1 = {0x1502}                                   # Phoenix / Hawk Point (AIE2, 4 columns)
PCI_NPU2 = {0x17F0, 0x17F2, 0x17F3, 0x1B0B, 0x1B0C}    # Strix / Krackan / Strix Halo / later (AIE2P family)


# ----------------------------------------------------------------------------------------------------------
# helpers shared by orchestrator and workers
# ----------------------------------------------------------------------------------------------------------
def bf16_bits(x):
    """float32 ndarray -> uint16 bf16 bit patterns, round-to-nearest-even (no ml_dtypes/torch needed)."""
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def bf16_val(b):
    import numpy as np
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def make_inputs(M, K, N, seed=1234):
    """Deterministic bf16 operands (identical in the CPU and NPU workers)."""
    import numpy as np
    rng = np.random.default_rng(seed + M * 7 + K * 13 + N * 17)
    A = bf16_bits(rng.standard_normal((M, K), dtype=np.float32) * 0.1)
    B = bf16_bits(rng.standard_normal((K, N), dtype=np.float32) * 0.1)
    return A, B


def reference(A_bits, B_bits):
    return bf16_val(A_bits) @ bf16_val(B_bits)   # fp32 product of the bf16 operands


def rel_l2(x, ref):
    import numpy as np
    d = np.linalg.norm((x - ref).astype(np.float64))
    n = np.linalg.norm(ref.astype(np.float64))
    return float(d / n) if n else float("inf")


def percentile(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def stats_ms(vals):
    return {"min": min(vals), "median": statistics.median(vals), "p90": percentile(vals, 90),
            "mean": statistics.fmean(vals), "n": len(vals)}


def cpu_busy_seconds():
    """System-wide busy CPU seconds (all cores, user+system+irq+softirq+steal) from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        tck = os.sysconf("SC_CLK_TCK")
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return (sum(v[:8]) - idle) / tck
    except Exception:
        return float("nan")


class EnergyMeter:
    """Package energy via RAPL if readable, else battery power_now sampling (only while discharging)."""

    def __init__(self):
        self.kind = "none"
        self.detail = "no readable energy source"
        self._rapl = None
        self._bat = None
        for p in ("/sys/class/powercap/intel-rapl:0/energy_uj",):
            try:
                with open(p) as f:
                    int(f.read())
                self._rapl = p
                self._rapl_max = int(open(os.path.dirname(p) + "/max_energy_range_uj").read())
                self.kind, self.detail = "rapl", p
                return
            except Exception:
                pass
        for d in sorted(glob.glob("/sys/class/power_supply/BAT*")):
            try:
                status = open(d + "/status").read().strip()
                if os.path.exists(d + "/power_now"):
                    self._bat = (d + "/power_now", None)
                elif os.path.exists(d + "/current_now") and os.path.exists(d + "/voltage_now"):
                    self._bat = (d + "/current_now", d + "/voltage_now")
                else:
                    continue
                self.kind = "battery" if status == "Discharging" else "battery-not-discharging"
                self.detail = f"{d} status={status}"
                return
            except Exception:
                pass

    def _bat_watts(self):
        try:
            a = int(open(self._bat[0]).read())
            if self._bat[1] is None:
                return a / 1e6
            return a * int(open(self._bat[1]).read()) / 1e12
        except Exception:
            return None

    def start(self):
        self._t0 = time.perf_counter()
        if self.kind == "rapl":
            self._e0 = int(open(self._rapl).read())
        elif self.kind == "battery":
            self._samples = []
            self._stop = False

            def run():
                while not self._stop:
                    w = self._bat_watts()
                    if w is not None:
                        self._samples.append(w)
                    time.sleep(0.2)
            self._th = threading.Thread(target=run, daemon=True)
            self._th.start()

    def stop(self):
        """-> (joules or None, mean watts or None)."""
        dt = time.perf_counter() - self._t0
        if self.kind == "rapl":
            e1 = int(open(self._rapl).read())
            d = e1 - self._e0
            if d < 0:
                d += self._rapl_max
            return d / 1e6, d / 1e6 / dt
        if self.kind == "battery":
            self._stop = True
            self._th.join(1.0)
            if self._samples:
                w = statistics.fmean(self._samples)
                return w * dt, w
        return None, None


def timed_loop(fn, min_seconds, min_iters, max_iters=100000):
    """Run fn() repeatedly. Returns per-call wall list (s) and totals (wall, process cpu, system busy cpu, joules, watts)."""
    em = EnergyMeter()
    walls = []
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    b0 = cpu_busy_seconds()
    em.start()
    t_start = time.perf_counter()
    while True:
        t = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t)
        el = time.perf_counter() - t_start
        if len(walls) >= max_iters or (len(walls) >= min_iters and el >= min_seconds):
            break
    total = time.perf_counter() - t_start
    joules, watts = em.stop()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    b1 = cpu_busy_seconds()
    n = len(walls)
    proc_cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    out = {
        "iters": n, "total_wall_s": total,
        "wall_ms": stats_ms([w * 1e3 for w in walls]),
        "proc_cpu_ms_per_call": proc_cpu / n * 1e3,
        "proc_cpu_over_wall": proc_cpu / total,        # 1.0 == one core fully busy
        "sys_busy_cpu_ms_per_call": (b1 - b0) / n * 1e3,
        "energy_source": em.kind,
        "mean_power_W": watts,
        "energy_mJ_per_call": (joules / n * 1e3) if joules is not None else None,
    }
    return out


def idle_baseline(seconds):
    em = EnergyMeter()
    b0 = cpu_busy_seconds()
    em.start()
    time.sleep(seconds)
    j, w = em.stop()
    b1 = cpu_busy_seconds()
    return {"seconds": seconds, "energy_source": em.kind, "idle_power_W": w, "sys_busy_cpu_s": b1 - b0}


# ----------------------------------------------------------------------------------------------------------
# device / environment detection (stdlib only)
# ----------------------------------------------------------------------------------------------------------
def read(p, default=""):
    try:
        return Path(p).read_text().strip()
    except Exception:
        return default


def detect_npu():
    """Find amdxdna-bound PCI devices. Returns list of dicts."""
    out = []
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        if read(dev + "/vendor") != "0x1022":
            continue
        drv = os.path.basename(os.path.realpath(dev + "/driver")) if os.path.exists(dev + "/driver") else ""
        did = int(read(dev + "/device", "0x0"), 16)
        if drv != "amdxdna" and did not in PCI_NPU1 | PCI_NPU2:
            continue
        gen = "npu1" if did in PCI_NPU1 else "npu2" if did in PCI_NPU2 else "unknown"
        out.append({"bdf": os.path.basename(dev), "pci_id": f"1022:{did:04x}", "revision": read(dev + "/revision"),
                    "driver": drv or "(none bound)", "mlir_aie_device": gen})
    return out


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def find_tool(name):
    """Tool on PATH, else in the usual XRT install dirs (/opt/xilinx/xrt/bin on tarball installs, /usr/bin on Arch)."""
    for cand in (shutil.which(name), f"/opt/xilinx/xrt/bin/{name}", f"/usr/bin/{name}"):
        if cand and os.path.exists(cand):
            return cand
    return None


XRT_SMI = find_tool("xrt-smi")


T_START = time.time()   # process start; error records ask for kernel messages since then


def kernel_log_tail(n_all=30, n_npu=60, since=None):
    """Last kernel messages. dmesg is root-only on many distros (kernel.dmesg_restrict=1), so fall back to
    `journalctl -k` (works for wheel/systemd-journal members). Never raises."""
    out = {"source": None, "npu_related": "", "tail": "", "note": ""}
    jc = ["journalctl", "-k", "-b", "--no-pager", "-n", "3000"] + (["--since", f"@{int(since)}"] if since else [])
    for name, cmd in (("dmesg", ["dmesg"]), ("journalctl -k", jc)):
        if not shutil.which(cmd[0]):
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except Exception as e:
            out["note"] += f"{name}: {type(e).__name__}: {e}; "
            continue
        if r.returncode != 0 or (not r.stdout.strip() and not since):
            out["note"] += f"{name}: rc={r.returncode} {r.stderr.strip()[:200]}; "
            continue
        lines = r.stdout.splitlines()
        key = ("amdxdna", "xdna", "accel", " npu", "xrt")
        out["source"] = name
        out["npu_related"] = "\n".join([l for l in lines if any(k in l.lower() for k in key)][-n_npu:])
        out["iommu_related"] = "\n".join([l for l in lines if "ivrs" not in l.lower() and ("iommu" in l.lower() or "amd-vi" in l.lower())][-10:])
        out["tail"] = "\n".join(lines[-n_all:])
        return out
    out["note"] += "kernel log not readable (try: sudo dmesg | tail -100 > dmesg.txt and send it too)"
    return out


def power_snapshot():
    """AC / battery state right now (so a plug/unplug during the run is visible in the results)."""
    snap = {"time": time.strftime("%H:%M:%S"), "ac_online": [], "battery": []}
    for d in sorted(glob.glob("/sys/class/power_supply/*")):
        typ = read(d + "/type")
        if typ == "Mains" or typ == "USB":
            snap["ac_online"].append({"name": os.path.basename(d), "online": read(d + "/online")})
        elif typ == "Battery":
            snap["battery"].append({k: read(f"{d}/{k}") for k in
                                    ("status", "capacity", "power_now", "energy_now", "energy_full", "charge_now", "voltage_now", "current_now")
                                    if os.path.exists(f"{d}/{k}")} | {"name": os.path.basename(d)})
    return snap


def on_battery(snap):
    return any(b.get("status") == "Discharging" for b in snap["battery"])


def environment_report(python_npu, python_cpu):
    env = {
        "kit_schema": SCHEMA,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kernel": platform.release(),
        "distro": read("/etc/os-release").split("PRETTY_NAME=")[-1].split("\n")[0].strip('"') if "PRETTY_NAME" in read("/etc/os-release") else "?",
        "cpu_model": next((l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo").splitlines() if l.startswith("model name")), "?"),
        "cpu_threads": os.cpu_count(),
        "cpu_flags": [f for f in ("avx2", "avx512f", "avx512_bf16", "avx_vnni", "amx_bf16") if f in read("/proc/cpuinfo")],
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "epp": read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "power_profile": sh(["powerprofilesctl", "get"]) if shutil.which("powerprofilesctl") else "n/a",
        "ac_online": [read(p) for p in glob.glob("/sys/class/power_supply/A*/online")],
        "battery_status": [read(p) for p in glob.glob("/sys/class/power_supply/BAT*/status")],
        "power_snapshot": power_snapshot(),
        "kernel_full": " ".join(platform.uname()),
        "kernel_cmdline": read("/proc/cmdline"),
        "user": {"uid": os.getuid(), "groups": [g for g in sh(["id", "-Gn"]).split()]},
        "npu_devices": detect_npu(),
        "accel_nodes": sorted(glob.glob("/dev/accel/accel*")),
        "accel_node_access": [os.access(p, os.R_OK | os.W_OK) for p in sorted(glob.glob("/dev/accel/accel*"))],
        "memlock_soft": resource.getrlimit(resource.RLIMIT_MEMLOCK)[0],
        "memlock_hard": resource.getrlimit(resource.RLIMIT_MEMLOCK)[1],
        "amdxdna_module": sh(["modinfo", "-F", "version", "amdxdna"]) + " | " + sh(["modinfo", "-F", "filename", "amdxdna"]),
        "amdxdna_loaded": "amdxdna" in read("/proc/modules"),
        "accel_sysfs": {os.path.basename(p) + "/" + k: read(f"{p}/device/{k}")
                        for p in sorted(glob.glob("/sys/class/accel/accel*")) for k in ("fw_version", "vbnv", "power_state")},
        "accel_node_stat": [sh(["ls", "-l", p]) for p in sorted(glob.glob("/dev/accel/accel*"))],
        "xrt_smi_path": XRT_SMI,
        "xrt_smi_version": sh([XRT_SMI, "--version"]) if XRT_SMI else "",
        "xrt_smi_examine": sh([XRT_SMI, "examine"], 60) if XRT_SMI else "xrt-smi not found (looked on PATH, /opt/xilinx/xrt/bin, /usr/bin)",
        "xrt_smi_examine_platform": sh([XRT_SMI, "examine", "-r", "platform"], 60) if XRT_SMI else "",
        "packages": sh(["pacman", "-Q", "xrt", "xrt-plugin-amdxdna", "linux-firmware", "linux-firmware-amd", "python"]) if shutil.which("pacman") else "pacman n/a",
        "firmware_files": sorted(os.path.relpath(p, "/lib/firmware") + (" -> " + os.readlink(p) if os.path.islink(p) else "")
                                 for p in glob.glob("/lib/firmware/amdnpu/1502_00/*")),
        "kernel_log": kernel_log_tail(),   # whole boot: shows when amdxdna loaded the firmware
        "energy_meter": (lambda m: {"kind": m.kind, "detail": m.detail})(EnergyMeter()),
        "rapl_readable": os.access("/sys/class/powercap/intel-rapl:0/energy_uj", os.R_OK),
        "python_npu": python_npu, "python_cpu": python_cpu, "python_running": sys.version,
        "pyxrt": sh([python_npu, "-c", "import pyxrt,sys;print(pyxrt.__file__)"]),
        "numpy": sh([python_npu, "-c", "import numpy;print(numpy.__version__, numpy.__file__)"]),
        "torch": sh([python_cpu, "-c", "import torch;print(torch.__version__, 'threads', torch.get_num_threads())"]),
        "env_omp": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "GOMP_", "MKL_", "KMP_"))},
    }
    return env


# ----------------------------------------------------------------------------------------------------------
# NPU worker (python with pyxrt + numpy)
# ----------------------------------------------------------------------------------------------------------
def npu_worker(args):
    import numpy as np
    import pyxrt

    manifest = json.loads(Path(args.manifest).read_text())
    cases = select_cases(manifest, args)
    results = {"role": "npu", "device_name": None, "cases": [], "finished": False}
    out_path = Path(args.json_out)

    def flush(current=None):
        """Rewrite the results file after every case: a crash/hang of the worker must not lose finished cases."""
        results["current_case"] = current
        out_path.write_text(json.dumps(results, indent=1))

    flush()
    try:
        dev = pyxrt.device(0)
        results["device_name"] = str(dev.get_info(pyxrt.xrt_info_device.name))
    except Exception:
        results["error"] = traceback.format_exc()
        results["kernel_log"] = kernel_log_tail(since=T_START - 2)
        flush()
        return 0
    sync_to = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    sync_from = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    base = Path(args.manifest).resolve().parent
    results["idle"] = idle_baseline(args.idle_seconds)
    flush()

    for case in cases:
        M, K, N = case["M"], case["K"], case["N"]
        out_bytes = 4 if case["dtype_out"] == "f32" else 2
        rec = {"id": case["id"], "M": M, "K": K, "N": N, "tile": case["tile"], "dtype_out": case["dtype_out"]}
        flush(case["id"])
        print(f"[npu] {case['id']} ...", flush=True)
        try:
            t = time.perf_counter()
            xb = pyxrt.xclbin(str(base / case["xclbin"]))
            dev.register_xclbin(xb)
            t_reg = time.perf_counter() - t
            t = time.perf_counter()
            ctx = pyxrt.hw_context(dev, xb.get_uuid())
            t_ctx = time.perf_counter() - t
            name = [k.get_name() for k in xb.get_kernels()][0]
            k = pyxrt.kernel(ctx, name)
            rec["load_ms"] = {"register_xclbin": t_reg * 1e3, "hw_context": t_ctx * 1e3}
            instr = np.fromfile(str(base / case["insts"]), dtype=np.uint32)
            bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
            bo_a = pyxrt.bo(dev, M * K * 2, pyxrt.bo.host_only, k.group_id(3))
            bo_b = pyxrt.bo(dev, K * N * 2, pyxrt.bo.host_only, k.group_id(4))
            bo_c = pyxrt.bo(dev, M * N * out_bytes, pyxrt.bo.host_only, k.group_id(5))
            bo_t = pyxrt.bo(dev, 1, pyxrt.bo.host_only, k.group_id(6))
            bo_r = pyxrt.bo(dev, 4, pyxrt.bo.host_only, k.group_id(7))
            bo_i.write(instr.tobytes(), 0)
            bo_i.sync(sync_to)

            A, B = make_inputs(M, K, N)
            ref = reference(A, B)
            a_bytes, b_bytes = A.tobytes(), B.tobytes()
            # --- correctness -------------------------------------------------------------------------
            bo_a.write(a_bytes, 0); bo_a.sync(sync_to)
            bo_b.write(b_bytes, 0); bo_b.sync(sync_to)
            bo_c.write(bytes(M * N * out_bytes), 0); bo_c.sync(sync_to)   # poison/zero the output first
            r = k(3, bo_i, len(instr), bo_a, bo_b, bo_c, bo_t, bo_r)
            state = r.wait(60000)   # ms; a firmware hang/rejection shows up as a non-COMPLETED state, not as a stuck worker
            rec["first_run_state"] = str(state)
            if "COMPLETED" not in str(state):
                raise RuntimeError(f"first dispatch did not complete: ert_cmd_state={state} (see kernel_log)")
            bo_c.sync(sync_from)
            raw = bo_c.read(M * N * out_bytes, 0)
            C = (np.frombuffer(raw, dtype=np.float32) if out_bytes == 4 else bf16_val(np.frombuffer(raw, dtype=np.uint16))).reshape(M, N)
            finite = bool(np.isfinite(C).all())
            err = rel_l2(C, ref) if finite else float("inf")
            rec["correctness"] = {
                "cmd_state": str(state), "all_finite": finite, "rel_l2_vs_fp32_of_bf16_inputs": err,
                "max_abs_err": float(np.abs(C - ref).max()) if finite else None,
                "ref_rms": float(np.sqrt((ref.astype(np.float64) ** 2).mean())),
                "threshold": case["max_rel_l2"], "pass": bool(finite and err <= case["max_rel_l2"]),
                "all_zero_output": bool((C == 0).all()),
            }
            if not rec["correctness"]["pass"] and not args.time_even_if_wrong:
                rec["timing_skipped"] = "correctness failed; use --time-even-if-wrong to time anyway"
                results["cases"].append(rec)
                flush()
                del bo_i, bo_a, bo_b, bo_c, bo_t, bo_r, k, ctx
                continue
            # --- resident dispatch: operands already on device -----------------------------------------
            def resident():
                rr = k(3, bo_i, len(instr), bo_a, bo_b, bo_c, bo_t, bo_r)
                rr.wait()
            for _ in range(args.warmup):
                resident()
            rec["resident"] = timed_loop(resident, args.min_seconds, args.min_iters)
            # --- end-to-end with per-stage profile -----------------------------------------------------
            stages = {"host_prep_ms": [], "sync_in_ms": [], "submit_ms": [], "wait_ms": [], "sync_out_ms": [], "read_ms": []}
            pc = time.perf_counter

            def e2e():
                t0 = pc(); bo_a.write(a_bytes, 0); bo_b.write(b_bytes, 0)
                t1 = pc(); bo_a.sync(sync_to); bo_b.sync(sync_to)
                t2 = pc(); rr = k(3, bo_i, len(instr), bo_a, bo_b, bo_c, bo_t, bo_r)
                t3 = pc(); rr.wait()
                t4 = pc(); bo_c.sync(sync_from)
                t5 = pc(); raw_ = bo_c.read(M * N * out_bytes, 0); _ = np.frombuffer(raw_, dtype=np.uint8)
                t6 = pc()
                for key, (x, y) in zip(stages, ((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t5, t6))):
                    stages[key].append((y - x) * 1e3)
            for _ in range(args.warmup):
                e2e()
            for v in stages.values():
                v.clear()
            rec["e2e"] = timed_loop(e2e, args.min_seconds, args.min_iters)
            rec["e2e_stage_median_ms"] = {key: statistics.median(v) for key, v in stages.items()}
            rec["e2e_stage_p90_ms"] = {key: percentile(v, 90) for key, v in stages.items()}
            flops = 2.0 * M * K * N
            rec["tflops_resident_median"] = flops / (rec["resident"]["wall_ms"]["median"] * 1e-3) / 1e12
            rec["tflops_resident_best"] = flops / (rec["resident"]["wall_ms"]["min"] * 1e-3) / 1e12
            rec["tflops_e2e_median"] = flops / (rec["e2e"]["wall_ms"]["median"] * 1e-3) / 1e12
        except BaseException as e:   # one bad case must not hide the others (xclbin rejected, firmware error, BO alloc, ...)
            if isinstance(e, KeyboardInterrupt):
                results["interrupted"] = True
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            rec["kernel_log"] = kernel_log_tail(since=T_START - 2)
            rec["env_hint"] = ("memlock soft limit is %s bytes (BO allocation failed? see README: memlock)" % resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
                               if "alloc" in str(e).lower() or "memory" in str(e).lower() else "")
            print(f"[npu] {case['id']} ERROR {rec['error']}", flush=True)
            if isinstance(e, KeyboardInterrupt):
                results["cases"].append(rec)
                break
        results["cases"].append(rec)
        flush()
    results["finished"] = True
    flush()


# ----------------------------------------------------------------------------------------------------------
# CPU worker (python with torch)
# ----------------------------------------------------------------------------------------------------------
def cpu_worker(args):
    import numpy as np
    import torch

    manifest = json.loads(Path(args.manifest).read_text())
    cases = select_cases(manifest, args)
    shapes = []
    for c in cases:
        s = (c["M"], c["K"], c["N"])
        if s not in shapes:
            shapes.append(s)
    res = {"role": "cpu", "variant": args.variant, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "(unset)"), "shapes": []}
    res["idle"] = idle_baseline(args.idle_seconds)
    res["errors"] = []
    for (M, K, N) in shapes:
        print(f"[cpu:{args.variant}] {M}x{K}x{N} ...", flush=True)
        try:
            A, B = make_inputs(M, K, N)
            a = torch.from_numpy(A.view(np.int16)).view(torch.bfloat16)
            b = torch.from_numpy(B.view(np.int16)).view(torch.bfloat16)
            ref = reference(A, B)
            out = torch.mm(a, b)
            err = rel_l2(out.float().numpy(), ref)
            for _ in range(args.warmup):
                torch.mm(a, b)
            rec = {"M": M, "K": K, "N": N, "rel_l2_vs_fp32_of_bf16_inputs": err}
            rec["mm"] = timed_loop(lambda: torch.mm(a, b), args.min_seconds, args.min_iters)
            rec["tflops_median"] = 2.0 * M * K * N / (rec["mm"]["wall_ms"]["median"] * 1e-3) / 1e12
            res["shapes"].append(rec)
        except Exception as e:
            res["errors"].append({"shape": [M, K, N], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        Path(args.json_out).write_text(json.dumps(res, indent=1))
    Path(args.json_out).write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------------------------------------
def select_cases(manifest, args):
    cases = manifest["cases"]
    if getattr(args, "quick", False):
        cases = [c for c in cases if c.get("quick")]
    if getattr(args, "only", None):
        cases = [c for c in cases if args.only in c["id"]]
    return cases


def verify_manifest_files(manifest, base):
    problems = []
    for c in manifest["cases"]:
        for key in ("xclbin", "insts"):
            p = base / c[key]
            if not p.is_file():
                problems.append(f"missing {p}")
            elif c.get(key + "_sha256") and hashlib.sha256(p.read_bytes()).hexdigest() != c[key + "_sha256"]:
                problems.append(f"checksum mismatch {p}")
    return problems


def run_worker(python, role, args, json_out, extra_env=None, extra_args=(), timeout=3600):
    """Run one worker subprocess. Returns (returncode, stderr_tail). A crash or hang of the worker never kills the kit."""
    cmd = [python, str(Path(__file__).resolve()), "--role", role, "--manifest", args.manifest, "--json-out", str(json_out),
           "--min-seconds", str(args.min_seconds), "--min-iters", str(args.min_iters), "--warmup", str(args.warmup),
           "--idle-seconds", str(args.idle_seconds)]
    if args.quick:
        cmd.append("--quick")
    if args.only:
        cmd += ["--only", args.only]
    if args.time_even_if_wrong:
        cmd.append("--time-even-if-wrong")
    cmd += list(extra_args)
    env = dict(os.environ)
    env.update(extra_env or {})
    try:
        r = subprocess.run(cmd, env=env, stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        err = (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        print(f"{role} worker TIMED OUT after {timeout}s and was killed", flush=True)
        return -9, "TIMEOUT\n" + err[-6000:]
    except Exception as e:
        return -1, f"could not start worker: {type(e).__name__}: {e}"
    if r.stderr:
        sys.stderr.write(r.stderr)
    return r.returncode, r.stderr[-6000:]


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def summarize(env, npu, cpus, manifest, run_info=None):
    L = []
    L.append(f"XDNA1 bf16 GEMM kit  {env['time']}  {env['cpu_model']}  kernel {env['kernel']}")
    L.append(f"NPU: {env['npu_devices']}   energy source: {env['energy_meter']['kind']} ({env['energy_meter']['detail']})")
    L.append(f"power profile: {env['power_profile']}  governor: {env['cpu_governor']}  AC online: {env['ac_online']}  battery: {env['battery_status']}")
    L.append(f"kernel: {env['kernel']}  xrt-smi: {(env.get('xrt_smi_version') or '').splitlines()[0:1]}  NPU fw: {env.get('accel_sysfs')}  memlock soft: {'unlimited' if env.get('memlock_soft') == resource.RLIM_INFINITY else env.get('memlock_soft')}")
    if run_info:
        ps, pe = run_info["power_start"], run_info["power_end"]
        L.append(f"power at start: {ps['battery']}  at end: {pe['battery']}  plugged/unplugged during run: {run_info['plug_state_changed_during_run']}  "
                 f"worker rc: {run_info['worker_returncodes']}  total wall: {run_info['total_wall_s']:.0f}s")
    for w in env.get("warnings", []):
        L.append("WARNING: " + w)
    if npu:
        L.append(f"idle: {fmt((npu.get('idle') or {}).get('idle_power_W'))} W ({(npu.get('idle') or {}).get('energy_source')})")
        L.append("")
        L.append("NPU cases (A,B resident = dispatch only; e2e = copy in + dispatch + copy out)")
        L.append(f"{'case':44s} {'ok':>3s} {'relL2':>8s} {'res ms':>8s} {'e2e ms':>8s} {'TFLOPS':>7s} {'cpu ms/call':>11s} {'cpu/wall':>8s} {'mJ/call':>8s}")
        for c in npu["cases"]:
            if "error" in c:
                L.append(f"{c['id']:44s} ERROR {c['error']}")
                continue
            cr = c["correctness"]
            if "resident" not in c:
                L.append(f"{c['id']:44s} {'NO':>3s} {cr['rel_l2_vs_fp32_of_bf16_inputs']:8.4f}  {c.get('timing_skipped','')}")
                continue
            L.append(f"{c['id']:44s} {'yes' if cr['pass'] else 'NO':>3s} {cr['rel_l2_vs_fp32_of_bf16_inputs']:8.4f} "
                     f"{c['resident']['wall_ms']['median']:8.3f} {c['e2e']['wall_ms']['median']:8.3f} {c['tflops_resident_median']:7.2f} "
                     f"{c['resident']['proc_cpu_ms_per_call']:11.3f} {c['resident']['proc_cpu_over_wall']:8.2f} {fmt(c['resident']['energy_mJ_per_call']):>8s}")
        npass = sum(1 for c in npu["cases"] if c.get("correctness", {}).get("pass"))
        L.append(f"correctness: {npass}/{len(npu['cases'])} NPU cases pass; errors: {sum(1 for c in npu['cases'] if 'error' in c)}")
        L.append("")
        L.append("NPU e2e stage medians (ms)")
        for c in npu["cases"]:
            if "e2e_stage_median_ms" in c:
                s = c["e2e_stage_median_ms"]
                L.append(f"  {c['id']:44s} " + " ".join(f"{k.replace('_ms','')}={v:.3f}" for k, v in s.items()))
    if npu and (npu.get("error") or any("error" in c for c in npu["cases"])):
        L.append("")
        L.append("NPU ERRORS (full tracebacks and kernel log excerpts are in the results json)")
        if npu.get("error"):
            L.append(npu["error"].strip().splitlines()[-1] if npu["error"].strip() else "")
        for c in npu["cases"]:
            if "error" in c:
                L.append(f"  {c['id']}: {c['error']}")
                kl = c.get("kernel_log") or {}
                for ln in (kl.get("npu_related") or "").splitlines()[-6:]:
                    L.append("      kernel: " + ln)
    for cpu in cpus:
        if cpu.get("errors"):
            L.append("")
            L.append(f"CPU ERRORS variant {cpu['variant']}: " + "; ".join(f"{e['shape']}: {e['error']}" for e in cpu["errors"]))
    for cpu in cpus:
        L.append("")
        L.append(f"CPU torch bf16 mm, variant {cpu['variant']} (threads {cpu['threads']}, OMP_WAIT_POLICY {cpu['omp_wait_policy']})")
        L.append(f"{'MxKxN':>20s} {'ms med':>8s} {'ms min':>8s} {'TFLOPS':>7s} {'cpu ms/call':>11s} {'cpu/wall':>8s} {'mJ/call':>8s}")
        for s in cpu["shapes"]:
            m = s["mm"]
            L.append(f"{s['M']:>6d}x{s['K']:>5d}x{s['N']:>5d} {m['wall_ms']['median']:8.3f} {m['wall_ms']['min']:8.3f} {s['tflops_median']:7.2f} "
                     f"{m['proc_cpu_ms_per_call']:11.3f} {m['proc_cpu_over_wall']:8.2f} {fmt(m['energy_mJ_per_call']):>8s}")
    if npu and cpus:
        L.append("")
        L.append("NPU(e2e) vs CPU, same shape. Ratio > 1 means the NPU path is better (faster / less CPU time / less energy).")
        L.append(f"{'case':44s} {'cpu variant':>14s} {'speed':>6s} {'cpu-time':>8s} {'energy':>7s}")
        for c in npu["cases"]:
            if "e2e" not in c:
                continue
            for cpu in cpus:
                s = next((x for x in cpu["shapes"] if (x["M"], x["K"], x["N"]) == (c["M"], c["K"], c["N"])), None)
                if not s:
                    continue
                m, e = s["mm"], c["e2e"]
                speed = m["wall_ms"]["median"] / e["wall_ms"]["median"]
                cput = m["proc_cpu_ms_per_call"] / e["proc_cpu_ms_per_call"] if e["proc_cpu_ms_per_call"] else float("inf")
                en = (m["energy_mJ_per_call"] / e["energy_mJ_per_call"]) if m["energy_mJ_per_call"] and e["energy_mJ_per_call"] else None
                L.append(f"{c['id']:44s} {cpu['variant']:>14s} {speed:6.2f} {cput:8.2f} {fmt(en):>7s}")
    return "\n".join(L)


def refuse(outdir, host, stamp, env, code, reason, tag):
    """Early exit that still leaves a small results file behind (nothing was loaded on any device)."""
    print("\n" + reason)
    out = {"env": env, "status": "refused", "exit_code": code, "reason": reason, "tag": tag}
    (outdir / f"kit_results_{host}{tag}_{stamp}.json").write_text(json.dumps(out, indent=1))
    print(f"(wrote {outdir}/kit_results_{host}{tag}_{stamp}.json -- send it back if you were told to)")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(HERE / "manifest.json" if (HERE / "manifest.json").exists() else HERE / "npu1" / "manifest.json"))
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--quick", action="store_true", help="only the cases marked quick in the manifest, shorter loops")
    ap.add_argument("--only", help="substring filter on case ids")
    ap.add_argument("--min-seconds", type=float, default=None, help="minimum timed seconds per loop (default 4, quick 1.5)")
    ap.add_argument("--min-iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--idle-seconds", type=float, default=3.0)
    ap.add_argument("--python-npu", default=sys.executable, help="python with pyxrt + numpy (default: the python running this script)")
    ap.add_argument("--python-cpu", default=sys.executable, help="python with torch + numpy (default: the python running this script)")
    ap.add_argument("--cpu-variants", default="default,passive,passive4", help="comma list of: default, passive, passive4")
    ap.add_argument("--cpu-only", action="store_true", help="skip the NPU part (works on any machine)")
    ap.add_argument("--npu-only", action="store_true")
    ap.add_argument("--env-only", action="store_true", help="print the environment report and exit")
    ap.add_argument("--dry-run", action="store_true", help="check device, manifest and files; load nothing on the NPU, time nothing")
    ap.add_argument("--power", choices=["ac", "battery"],
                    help="declare how this run is meant to be powered; exits with code 7 (nothing run) if the machine is not in that state. "
                         "`battery` is what gives energy numbers without root. Also tags the result files.")
    ap.add_argument("--tag", default="", help="label added to the result file names (default: the --power value)")
    ap.add_argument("--worker-timeout", type=int, default=3600, help="kill a worker after this many seconds (default 3600)")
    ap.add_argument("--allow-device-mismatch", action="store_true", help="DEBUG/developer only: run although the NPU generation differs from the manifest target")
    ap.add_argument("--time-even-if-wrong", action="store_true")
    ap.add_argument("--pmode", choices=["default", "powersaver", "balanced", "performance", "turbo"],
                    help="set the NPU power mode with `xrt-smi configure --pmode` for the NPU part and restore `default` afterwards "
                         "(an Ryzen 7 8700G thesis measured NPU clocks 0.79/1.02/1.79 GHz for powersaver/balanced/turbo)")
    ap.add_argument("--role", choices=["npu", "cpu"], help=argparse.SUPPRESS)
    ap.add_argument("--variant", default="default", help=argparse.SUPPRESS)
    ap.add_argument("--json-out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.min_seconds is None:
        args.min_seconds = 1.5 if args.quick else 4.0

    if args.role == "npu":
        return npu_worker(args)
    if args.role == "cpu":
        return cpu_worker(args)

    tag = "_" + (args.tag or args.power) if (args.tag or args.power) else ""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    host = platform.node() or "host"
    t_run0 = time.time()
    env = environment_report(args.python_npu, args.python_cpu)
    print(json.dumps({k: v for k, v in env.items() if k not in ("xrt_smi_examine", "xrt_smi_examine_platform", "kernel_log")}, indent=1))
    print("--- xrt-smi examine ---\n" + env["xrt_smi_examine"] + "\n" + env["xrt_smi_examine_platform"])
    warnings = []
    if env["memlock_soft"] != resource.RLIM_INFINITY and env["memlock_soft"] < (128 << 20):
        warnings.append(f"memlock soft limit is {env['memlock_soft'] >> 20} MiB (hard: "
                        f"{'unlimited' if env['memlock_hard'] == resource.RLIM_INFINITY else str(env['memlock_hard'] >> 20) + ' MiB'}): if buffer allocation fails see README 'memlock'")
    if env["energy_meter"]["kind"] not in ("rapl", "battery"):
        warnings.append("no energy source: run with --power battery (unplugged) or make RAPL readable (see README); times and CPU time are still measured")
    for w in warnings:
        print("WARNING:", w)
    env["warnings"] = warnings
    if args.env_only:
        (outdir / f"kit_env_{host}{tag}_{stamp}.json").write_text(json.dumps(env, indent=1))
        return 0
    if args.power:
        snap = power_snapshot()
        if (args.power == "battery") != on_battery(snap):
            return refuse(outdir, host, stamp, env, 7,
                          f"POWER STATE MISMATCH: --power {args.power} requested but the machine is "
                          f"{'on battery (discharging)' if on_battery(snap) else 'not discharging (AC connected or battery full/charging)'}. "
                          f"{'Unplug the charger' if args.power == 'battery' else 'Plug in the charger'}, wait ~10 s, and run again. Nothing was run.", tag)

    manifest = json.loads(Path(args.manifest).read_text())
    base = Path(args.manifest).resolve().parent
    problems = verify_manifest_files(manifest, base)
    if problems:
        return refuse(outdir, host, stamp, env, 2, "MANIFEST PROBLEMS (re-extract the tarball):\n  " + "\n  ".join(problems), tag)
    target = manifest["target"]
    print(f"\nmanifest: {len(manifest['cases'])} cases, built for {target} ({manifest.get('built_with','?')})")

    want_npu = not args.cpu_only
    if want_npu:
        devs = env["npu_devices"]
        if not devs:
            return refuse(outdir, host, stamp, env, 3,
                          "NO NPU FOUND: no amdxdna-bound PCI device (is the amdxdna driver loaded? `lsmod | grep amdxdna`, "
                          "`ls /dev/accel`). Nothing was loaded on any device. Re-run with --cpu-only for the CPU baseline.", tag)
        gens = {d["mlir_aie_device"] for d in devs}
        if target not in gens and not args.allow_device_mismatch:
            return refuse(outdir, host, stamp, env, 4,
                          f"DEVICE MISMATCH: this machine's NPU is {devs[0]['pci_id']} = {sorted(gens)}, but the xclbins in the "
                          f"manifest were built for {target}. They would not load, so nothing was loaded. "
                          f"Re-run with --cpu-only for the CPU baseline.", tag)
        if not env["accel_nodes"] or not all(env["accel_node_access"]):
            return refuse(outdir, host, stamp, env, 5,
                          f"NO ACCESS to {env['accel_nodes'] or '/dev/accel/accel*'} for this user: {env['accel_node_stat']}, your groups: {env['user']['groups']}. "
                          "Fix: `sudo usermod -aG render $USER` (the node is normally group render), log out and in again, or check the udev rule. "
                          "Nothing was loaded.", tag)
        if ".so" not in env["pyxrt"] or "Traceback" in env["pyxrt"]:
            return refuse(outdir, host, stamp, env, 6,
                          f"pyxrt is not importable with {args.python_npu}: {env['pyxrt']}\n"
                          "pyxrt comes with the Arch/Manjaro `xrt` package and only exists for the SYSTEM python "
                          "(/usr/lib/python3.*/site-packages/pyxrt*.so). Create the venv with `python3 -m venv --system-site-packages kitvenv` "
                          "(see README) or pass --python-npu /usr/bin/python3.", tag)
    if args.dry_run:
        print("\n--dry-run: environment, device generation and manifest files check out; nothing was loaded or timed.")
        return 0

    work = outdir / f"work_{stamp}"
    work.mkdir(exist_ok=True)
    npu = None
    cpus = []
    run_info = {"worker_returncodes": {}, "worker_stderr_tail": {}, "power_start": power_snapshot()}
    if want_npu:
        print("\n=== NPU worker ===", flush=True)
        if args.pmode:
            env["pmode_set"] = sh([XRT_SMI or "xrt-smi", "configure", "--pmode", args.pmode])
            print("pmode ->", args.pmode, env["pmode_set"][-200:])
        try:
            rc, err = run_worker(args.python_npu, "npu", args, work / "npu.json", timeout=args.worker_timeout)
        finally:
            if args.pmode:
                print("pmode restore:", sh([XRT_SMI or "xrt-smi", "configure", "--pmode", "default"])[-200:])
        run_info["worker_returncodes"]["npu"] = rc
        run_info["worker_stderr_tail"]["npu"] = err
        if (work / "npu.json").exists():
            try:
                npu = json.loads((work / "npu.json").read_text())   # partial results are kept when the worker crashed
            except Exception:
                npu = None
        if rc != 0:
            print(f"NPU worker exited with {rc} (partial results are kept)")
            run_info["kernel_log_after_npu_worker_failure"] = kernel_log_tail(since=t_run0 - 2)
        if npu is None:
            npu = {"role": "npu", "cases": [], "error": f"NPU worker produced no results (rc={rc})"}
    if not args.npu_only:
        variants = {"default": {}, "passive": {"OMP_WAIT_POLICY": "PASSIVE"},
                    "passive4": {"OMP_WAIT_POLICY": "PASSIVE", "OMP_NUM_THREADS": "4"}}
        for v in [x for x in args.cpu_variants.split(",") if x]:
            print(f"\n=== CPU worker: {v} ===", flush=True)
            outj = work / f"cpu_{v}.json"
            rc, err = run_worker(args.python_cpu, "cpu", args, outj, variants[v], ["--variant", v], timeout=args.worker_timeout)
            run_info["worker_returncodes"]["cpu_" + v] = rc
            run_info["worker_stderr_tail"]["cpu_" + v] = err
            if outj.exists():
                try:
                    cpus.append(json.loads(outj.read_text()))
                except Exception:
                    pass
            if rc != 0:
                print(f"CPU worker {v} exited with {rc} (is torch importable with {args.python_cpu}?)")
    run_info["power_end"] = power_snapshot()
    run_info["plug_state_changed_during_run"] = on_battery(run_info["power_start"]) != on_battery(run_info["power_end"])
    run_info["total_wall_s"] = time.time() - t_run0
    summary = summarize(env, npu, cpus, manifest, run_info)
    print("\n" + summary)
    out = {"env": env, "manifest_target": target, "manifest_built_with": manifest.get("built_with"),
           "manifest_cases": [{k: c[k] for k in ("id", "tile", "max_rel_l2", "xclbin_sha256", "insts_sha256") if k in c} for c in manifest["cases"]],
           "run": run_info, "args": vars(args), "tag": tag, "npu": npu, "cpu": cpus}
    (outdir / f"kit_results_{host}{tag}_{stamp}.json").write_text(json.dumps(out, indent=1))
    (outdir / f"kit_summary_{host}{tag}_{stamp}.txt").write_text(summary + "\n")
    print(f"\nWrote {outdir}/kit_results_{host}{tag}_{stamp}.json and kit_summary_{host}{tag}_{stamp}.txt  -- please send both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
