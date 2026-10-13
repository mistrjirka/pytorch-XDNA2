#!/usr/bin/env python3
"""The step autotuner must not finalize while NPU kernel builds are pending.

Pure Python (simulated clock and build status, no NPU work).  Builds finish
after step 100, well past the warm-up wait.  Until then every mode runs the CPU
path and costs the same; afterwards mode 2 is the fastest.  A tuner that
decides on the build-time samples picks CPU only for good.
"""
import sys
import time
import types

import xdna_train
from xdna_train import device

state = {"step": 0, "mode": 2, "clock": 0.0}
BUILT_AT = 100
COST = {2: 10.0, 1: 12.0, 0: 20.0}      # ms per step once the kernels exist


def cost():
    return COST[state["mode"]] if state["step"] >= BUILT_AT else 20.0


fake_c = types.SimpleNamespace(
    iron_builds_pending=lambda: 1 if state["step"] < BUILT_AT else 0,
    set_npu_mode=lambda m: state.update(mode=m),
)
real_c, real_clock = device._C, time.perf_counter
device._C = fake_c
time.perf_counter = lambda: state["clock"]
try:
    tuner = device._StepAutotuner(warmup=3, steps=2, rounds=1, tolerance=0.03)
    param = types.SimpleNamespace(device=types.SimpleNamespace(type="xdna"))
    opt = types.SimpleNamespace(param_groups=[{"params": [param]}])
    for _ in range(400):
        state["step"] += 1
        state["clock"] += cost() / 1e3
        tuner(opt, (), {})
finally:
    device._C, time.perf_counter = real_c, real_clock

ok = tuner.result is not None and tuner.result["mode"] == 2
print(f"autotuner mode after builds finished: {tuner.result and tuner.result['mode']} "
      f"(restarts {getattr(tuner, "restarts", "n/a")}): {'OK' if ok else 'FAIL (expected 2)'}")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
