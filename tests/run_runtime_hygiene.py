#!/usr/bin/env python3
"""Runtime hygiene: fused optimizer default per optimizer, CPU fallback runs
each op once (RNG and partial writes), out= resize still works, and removed
env vars are reported instead of silently ignored.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import json
import os
import subprocess
import sys

import torch
import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


# A. Every new Adam on XDNA params gets the fused default, even when earlier
# optimizers were freed (CPython reuses id()).
fused_ok = 0
N_OPT = 200
for _ in range(N_OPT):
    p = torch.nn.Parameter(torch.zeros(4, device="xdna"))
    opt = torch.optim.Adam([p], lr=1e-3)
    p.grad = torch.ones(4, device="xdna")
    opt.step()
    fused_ok += bool(opt.param_groups[0].get("fused"))
    del opt, p
check(f"fused default on every optimizer ({fused_ok}/{N_OPT})", fused_ok == N_OPT)

# B. An op that fails on XDNA must consume RNG and write exactly as on CPU.
seq = {}
for dev in ("cpu", "xdna"):
    torch.manual_seed(0)
    x = torch.zeros(4, device=dev)
    p = torch.tensor([0.5, 0.5, 2.0, 0.5], device=dev)
    try:
        x.bernoulli_(p)
        raised = False
    except RuntimeError:
        raised = True
    seq[dev] = (raised, torch.rand(1).item())
check("bernoulli_ with invalid p raises on both", seq["cpu"][0] and seq["xdna"][0])
check(f"RNG state after failed bernoulli_ matches CPU "
      f"(cpu {seq['cpu'][1]:.6f}, xdna {seq['xdna'][1]:.6f})",
      seq["cpu"][1] == seq["xdna"][1])

# B2. out= that must be resized (non-empty, too small) still works like CPU.
res = {}
for dev in ("cpu", "xdna"):
    a = torch.ones(2, 3, device=dev)
    out = torch.zeros(2, device=dev)
    r = torch.cumsum(a, 0, out=out)
    res[dev] = (tuple(r.shape), tuple(out.shape), r.cpu().tolist())
check("cumsum out= resize matches CPU", res["cpu"] == res["xdna"])

# D. Removed env vars are reported once at registration, not silently ignored.
REMOVED = ["XDNA_LAZY_DECODER_VALUES", "XDNA_FAST_DX584_ARTIFACT_DIR"]
probe = (
    "import json, warnings\n"
    "with warnings.catch_warnings(record=True) as w:\n"
    "    warnings.simplefilter('always')\n"
    "    import xdna_train\n"
    "    xdna_train.register_xdna_device()\n"
    "print(json.dumps([str(x.message) for x in w if 'no longer' in str(x.message)]))\n"
)


def removed_warnings(extra_env):
    env = {k: v for k, v in os.environ.items() if k not in REMOVED}
    env.update(extra_env)
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                         text=True, timeout=600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("[")]
    if not lines:
        print(f"  subprocess rc={out.returncode}, stderr tail: {out.stderr[-300:]!r}")
        return None
    return json.loads(lines[-1])


msgs = removed_warnings({"XDNA_LAZY_DECODER_VALUES": "1",
                         "XDNA_FAST_DX584_ARTIFACT_DIR": "/nonexistent"})
check("removed env var warning names both vars",
      msgs is not None and len(msgs) == 1
      and "XDNA_LAZY_DECODER_VALUES" in msgs[0]
      and "XDNA_FAST_DX584_ARTIFACT_DIR" in msgs[0]
      and "docs/MODES_AUDIT.md" in msgs[0])
msgs = removed_warnings({})
check("no removed env var warning when none set", msgs == [])

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
