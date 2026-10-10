#!/usr/bin/env python3
"""Benchmark the real NIS style_transfer.py: CPU BF16 vs XDNA BF16.

The script source is executed in memory with minimal substitutions (BF16
dtype, iteration count, test-frame directory, output directory, ffmpeg
skipped); model, data, loss, optimizer, DataLoader, validation and checkpoints
are the script's own.  The NIS files are never modified.

Configs run serially in alternating (ABBA) order.  A thermal guard waits for
the CPU to cool before each run and kills a run that overheats, because
sustained CPU+NPU load can trip this laptop's thermal shutdown.

    python benchmarks/nis_style_transfer.py --iters 200 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NIS = Path("/home/jirka/programovani/magistr/nis/lab02_cnn")
PY = "/home/jirka/programovani/magistr/nis/.venv/bin/python"
REPO = Path(__file__).resolve().parents[1]

CONFIGS = {
    "cpu-bf16": {"NIS_DEVICE": "cpu"},
    "xdna-bf16-installed": {"NIS_DEVICE": "xdna"},
    "xdna-bf16-repo": {"NIS_DEVICE": "xdna", "PYTHONPATH": str(REPO)},
}

RUNNER = r'''
import os, sys, time, json, statistics
NIS = sys.argv[1]
os.chdir(NIS); sys.path.insert(0, NIS)
src = open(os.path.join(NIS, "style_transfer.py")).read()
def sub(old, new):
    global src
    assert src.count(old) == 1, "style_transfer.py changed near: " + old[:60]
    src = src.replace(old, new)
sub("    training_dtype = torch.float32",
    "    training_dtype = torch.bfloat16\n"
    "    print('xdna_train from', xdna_train.__file__, flush=True)")
sub('    out_pth = "./output"', '    out_pth = os.environ["BENCH_OUT"]')
sub('        num_patches=200000,', '        num_patches=int(os.environ["BENCH_ITERS"]) * 40,')
sub('        input_src=f"data/{vid_id}/test/input",\n        target_src=f"data/{vid_id}/test/input",',
    '        input_src=os.environ["BENCH_TEST_DIR"],\n        target_src=os.environ["BENCH_TEST_DIR"],')
sub('    start_time = time.time()', '    start_time = time.time()\n    __T0_WALL = time.time()\n    __TRAIN_T = [time.perf_counter()]')
sub('            print(f"Iter: {idx:04d}', '            __TRAIN_T.append(time.perf_counter())\n            print(f"Iter: {idx:04d}')
sub('    print("Stylizing full video frames...")',
    '    print("Stylizing full video frames...")\n    __GEN_T = [time.perf_counter()]')
sub('            for i in range(gen_out.shape[0]):',
    '            __GEN_T.append(time.perf_counter())\n            for i in range(gen_out.shape[0]):')
sub('        subprocess.run(cmd, check=True)', '        pass  # ffmpeg skipped in benchmark')
src += """
import json, statistics
_d = [b - a for a, b in zip(__TRAIN_T, __TRAIN_T[1:])]
_skip = int(os.environ.get("BENCH_SKIP", "5"))
_steady = [d for i, d in enumerate(_d) if i >= _skip and i % on_step != 0]
_valid = [d for i, d in enumerate(_d) if i > 0 and i % on_step == 0]
_g = [b - a for a, b in zip(__GEN_T, __GEN_T[1:])][3:]
print("RESULT=" + json.dumps({
    "iters": len(_d),
    "train_step_median_ms": 1e3 * statistics.median(_steady),
    "train_step_mean_ms": 1e3 * statistics.mean(_steady),
    "train_mean_incl_validation_ms": 1e3 * statistics.mean(_d[_skip:]),
    "validation_iter_mean_ms": 1e3 * statistics.mean(_valid) if _valid else None,
    "final_loss": float(loss.item()),
    "stylize_frame_median_ms": 1e3 * statistics.median(_g),
    "stylize_frames_timed": len(_g),
    "train_step_ms": [round(1e3 * d, 2) for d in _d],
    "train_t0_wall": __T0_WALL,
}))
if os.environ.get("XDNA_PROFILE_MAPPED_CPU") == "1":
    from xdna_train import device as _xd
    _n = max(1, len(_d))
    print("PHASES (ms per step, incl. warmup):")
    for _k, _v in sorted(_xd.mapped_cpu_stats().items(), key=lambda kv: -kv[1]["total_ms"])[:30]:
        print(f"  {_k:44s} {_v['calls'] / _n:6.1f}/step {_v['total_ms'] / _n:8.2f} ms")
"""
exec(compile(src, "style_transfer.py[bench]", "exec"), {"__name__": "__main__"})
'''


def cpu_temp() -> float | None:
    """Hottest AMD CPU sensor (k10temp Tctl), in degrees C."""
    best = None
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() != "k10temp":
                continue
            for t in hw.glob("temp*_input"):
                v = int(t.read_text()) / 1000
                best = v if best is None else max(best, v)
        except OSError:
            pass
    return best


def package_power_w() -> float | None:
    """SoC package power (PPT) from the APU's amdgpu sensor; RAPL needs root."""
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() == "amdgpu":
                return int((hw / "power1_input").read_text()) / 1e6
        except OSError:
            pass
    return None


def avg_cpu_mhz() -> float | None:
    """Mean effective core clock (APERF/MPERF based) across CPUs."""
    vals = []
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_avg_freq"):
        try:
            vals.append(int(f.read_text()) / 1000)
        except OSError:
            pass
    return sum(vals) / len(vals) if vals else None


def preheat(seconds: float) -> None:
    """Load all cores so every config is measured in the hot, throttled state
    real training runs in (a cold CPU is 6-9% faster for the first minute)."""
    if seconds <= 0:
        return
    burn = ("import torch, time\n"
            "a = torch.randn(2048, 2048, dtype=torch.bfloat16)\n"
            f"t = time.time()\nwhile time.time() - t < {seconds}: a @ a\n")
    subprocess.run([PY, "-c", burn], cwd="/tmp", check=True)


def wait_cool(limit: float, timeout: float = 600) -> float | None:
    t0 = time.time()
    while (t := cpu_temp()) is not None and t > limit and time.time() - t0 < timeout:
        time.sleep(5)
    return cpu_temp()


def run_one(name: str, args, test_dir: str, out_root: str) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CONFIGS[name], BENCH_ITERS=str(args.iters), BENCH_TEST_DIR=test_dir,
               BENCH_SKIP=str(args.skip),
               BENCH_OUT=tempfile.mkdtemp(dir=out_root, prefix=name + "-"))
    start_temp = wait_cool(args.cool_to) if args.preheat <= 0 else (preheat(args.preheat) or cpu_temp())
    p = subprocess.Popen([PY, "-c", RUNNER, str(NIS)], env=env, cwd="/tmp",
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = start_temp or 0.0
    telemetry = []
    while p.poll() is None:
        t = cpu_temp()
        telemetry.append((round(time.time(), 2), t, package_power_w(), avg_cpu_mhz()))
        if t is not None:
            peak = max(peak, t)
            if t >= args.abort_at:
                p.kill()
                return {"config": name, "error": f"aborted: CPU at {t:.0f} C"}
        time.sleep(0.5)
    out, err = p.communicate()
    result = next((l[7:] for l in out.splitlines() if l.startswith("RESULT=")), None)
    if result is None:
        return {"config": name, "error": (err.strip().splitlines() or ["no output"])[-1]}
    res = json.loads(result)
    if "PHASES (" in out:
        print(f"[{name}] " + out[out.index("PHASES ("):], flush=True)
    res.update(config=name, start_temp_c=start_temp, peak_temp_c=peak,
               telemetry=telemetry,
               package=next((l for l in out.splitlines() if l.startswith("xdna_train from")), ""))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=60, help="test frames to stylize")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cool-to", type=float, default=70.0)
    ap.add_argument("--abort-at", type=float, default=200.0)
    ap.add_argument("--skip", type=int, default=5, help="warmup iterations excluded")
    ap.add_argument("--preheat", type=float, default=0.0,
                    help="seconds of all-core load before each run (replaces cooldown)")
    ap.add_argument("--quick", action="store_true",
                    help="fast A/B: preheat 45 s, 60 iterations (10 warmup), 1 round, 5 frames")
    ap.add_argument("--define", action="append", default=[],
                    help="extra config NAME=BASE:K=V,K=V (e.g. nonpu=xdna-bf16-repo:XDNA_ENABLE_PATCH_CONV=0)")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "nis_style_transfer.json"))
    args = ap.parse_args()
    if args.quick:
        args.preheat, args.iters, args.skip, args.rounds, args.frames = 45.0, 60, 10, 1, 5
    for spec in args.define:
        name, rest = spec.split("=", 1)
        base, _, kvs = rest.partition(":")
        CONFIGS[name] = dict(CONFIGS[base], **dict(kv.split("=", 1) for kv in kvs.split(",") if kv))

    work = Path(tempfile.mkdtemp(prefix="nis-bench-"))
    test_dir = work / "test"
    test_dir.mkdir()
    src = NIS / "data" / "VID20261009180405" / "test" / "input"
    for f in sorted(os.listdir(src))[: args.frames]:
        (test_dir / f).symlink_to(src / f)

    names = args.configs.split(",")
    results = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for r in range(args.rounds):
        for name in names if r % 2 == 0 else names[::-1]:
            res = run_one(name, args, str(test_dir), str(work))
            res["round"] = r
            results.append(res)
            print(json.dumps(res), flush=True)
            Path(args.out).write_text(json.dumps(results, indent=1))
            if "error" in res and res["error"].startswith("aborted"):
                print("thermal abort; stopping benchmark", flush=True)
                return

    print("\nconfig                 train ms/step (median)  incl. validation  stylize ms/frame  vs cpu-bf16")
    cpu = [r["train_step_median_ms"] for r in results if r.get("config") == "cpu-bf16" and "error" not in r]
    base = statistics.mean(cpu) if cpu else None
    for name in names:
        rs = [r for r in results if r.get("config") == name and "error" not in r]
        if not rs:
            continue
        m = statistics.mean
        print(f"{name:22s} {m(r['train_step_median_ms'] for r in rs):12.1f}"
              f" {m(r['train_mean_incl_validation_ms'] for r in rs):20.1f}"
              f" {m(r['stylize_frame_median_ms'] for r in rs):16.1f}"
              + (f"  {base / m(r['train_step_median_ms'] for r in rs):9.2f}x" if base else ""))


if __name__ == "__main__":
    main()
