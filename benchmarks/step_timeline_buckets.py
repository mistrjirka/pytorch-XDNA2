#!/usr/bin/env python3
"""Exclusive host time per kind of work and phase, from a step_timeline.py trace.

    python benchmarks/step_timeline_buckets.py out.trace.json

Phases are the `step:*` user-annotation spans written by step_timeline.py.
Exclusive time is computed per (pid, tid) stack, so events of different threads
never deduct from each other.  Events on threads other than the main thread are
listed separately (CPU busy time of that thread, not NPU occupancy).

What the buckets are NOT: "native conv dispatch wrapper" is the exclusive time of
aten::*convolution*_overrideable, which runs for NPU calls AND for the CPU
fallback branch of the same op; it includes host pack/unpack/wait for NPU calls
but is not a hardware-proven NPU sum.  Use XDNA_CONV3X3W_TRACE=1 for per-call
pack/wait/unpack and the meta file's mapped_cpu_per_step for the native counters.
Host busy time, blocked host time and NPU occupancy are three different things;
this tool reports the first (and lumps waits into the wrapper bucket).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def category(n: str) -> str:
    if "overrideable" in n:
        return "native conv dispatch wrapper (NPU pack/wait/unpack + CPU-fallback overhead)"
    if "mkldnn_convolution" in n or n == "aten::convolution_backward":
        return "CPU conv (oneDNN)"
    if "batch_norm" in n:
        return "BatchNorm"
    if "leaky" in n or "relu" in n:
        return "activation"
    if "cat" in n or "upsample" in n or "slice" in n or "narrow" in n:
        return "cat / resize"
    if "max_pool" in n:
        return "max-pool"
    if "adam" in n.lower() or "Optimizer" in n or "lerp" in n or "addc" in n:
        return "optimizer"
    if "copy" in n or "_to_copy" in n or "clone" in n or "contiguous" in n:
        return "copies"
    return "other (elementwise, autograd, loss math)"


def exclusive(events):
    """Exclusive duration (us) per event, stacks kept per (pid, tid)."""
    by = defaultdict(list)
    for e in events:
        by[(e.get("pid"), e.get("tid"))].append(e)
    for evs in by.values():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack = []
        for e in evs:
            while stack and e["ts"] >= stack[-1]["end"]:
                stack.pop()
            e["end"] = e["ts"] + e["dur"]
            e["child"] = 0.0
            if stack:
                stack[-1]["child"] += e["dur"]
            stack.append(e)
    return by


def main() -> None:
    d = json.load(open(sys.argv[1]))
    allx = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    ops = [e for e in allx if e.get("cat") == "cpu_op"]
    spans = sorted((e for e in allx if e.get("cat") == "user_annotation"
                    and e["name"].startswith("step:")), key=lambda e: e["ts"])
    if not spans:
        sys.exit("no step:* spans: capture the trace with benchmarks/step_timeline.py")
    main_tid = (spans[0].get("pid"), spans[0].get("tid"))
    by = exclusive(ops)
    t0 = spans[0]["ts"]
    tab = {s["name"]: defaultdict(float) for s in spans}
    for e in by.get(main_tid, []):
        for s in spans:
            if s["ts"] <= e["ts"] < s["ts"] + s["dur"]:
                tab[s["name"]][category(e["name"])] += (e["dur"] - e["child"]) / 1e3
                break
    names = [s["name"] for s in spans]
    cats = sorted({c for t in tab.values() for c in t}, key=lambda c: -sum(t.get(c, 0) for t in tab.values()))
    print("phase spans (ms): " + ", ".join(f"{s['name'][5:]} {s['dur']/1e3:.1f}" for s in spans))
    print(f"{'main-thread exclusive host time, ms':80s}" + "".join(f"{n[5:]:>15s}" for n in names) + f"{'total':>9s}")
    for c in cats:
        v = [tab[n].get(c, 0) for n in names]
        print(f"{c:80s}" + "".join(f"{x:15.1f}" for x in v) + f"{sum(v):9.1f}")
    v = [sum(tab[n].values()) for n in names]
    print(f"{'TOTAL covered by cpu_op events':80s}" + "".join(f"{x:15.1f}" for x in v) + f"{sum(v):9.1f}")
    for tid, evs in by.items():
        if tid == main_tid:
            continue
        top = sum((e["dur"] - e["child"]) for e in evs) / 1e3
        print(f"other thread {tid}: {len(evs)} cpu_op events, {top:.1f} ms exclusive host time")
    if len(by) == 1:
        print("note: only the main thread recorded cpu_op events; helper-thread (async dW) "
              "and NPU activity are not in this trace")


if __name__ == "__main__":
    main()
