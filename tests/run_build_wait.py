#!/usr/bin/env python3
"""The optimizer-step build waiter and its interplay with the step autotuner.

Pure Python (fake native module, no NPU work): at a step boundary with builds
pending, training waits until they are done and the autotuner drops the
measurement of that step; with nothing pending (warm cache) the hook is a no-op;
a timeout ends the wait without hanging.
"""
import sys
import types

import xdna_train
from xdna_train import device

state = {"pending": 3, "waits": 0, "announced": 0, "mode": 2}


def wait(timeout):
    state["waits"] += 1
    if state["waits"] >= 4:
        state["pending"] = 0
    return state["pending"]


fake_c = types.SimpleNamespace(
    iron_builds_pending=lambda: state["pending"],
    iron_builds_announce=lambda: state.update(announced=state["announced"] + 1),
    iron_wait_builds=wait,
    set_npu_mode=lambda m: state.update(mode=m),
)
param = types.SimpleNamespace(device=types.SimpleNamespace(type="xdna"))
opt = types.SimpleNamespace(param_groups=[{"params": [param]}])
cpu_param = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
cpu_opt = types.SimpleNamespace(param_groups=[{"params": [cpu_param]}])

real_c, real_tuner = device._C, device._AUTOTUNER
device._C = fake_c
try:
    device._AUTOTUNER = device._StepAutotuner(warmup=3, steps=2, rounds=1, tolerance=0.03)
    device._AUTOTUNER.phase, device._AUTOTUNER.skip = 2, 0       # mid-measurement
    waiter = device._BuildWaiter(max_wait=60)

    waiter(cpu_opt, (), {})                                      # not an XDNA optimizer
    ok = state["waits"] == 0
    waiter(opt, (), {})
    ok &= state["pending"] == 0 and state["waits"] == 4 and state["announced"] == 1
    ok &= device._AUTOTUNER.phase == 0 and device._AUTOTUNER.skip == 1   # restarted
    n = state["waits"]
    waiter(opt, (), {})                                          # warm: nothing pending
    ok &= state["waits"] == n and state["announced"] == 1

    state.update(pending=2, waits=-10 ** 9)                      # never finishes: timeout
    ok &= device.wait_for_builds(0.0) is False
finally:
    device._C, device._AUTOTUNER = real_c, real_tuner

print("build waiter:", "OK" if ok else "FAIL")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
