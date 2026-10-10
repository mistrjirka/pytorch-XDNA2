#!/usr/bin/env python3
"""How universal is the conv3x3w NPU family across batch sizes and image sizes?

For each (Cin, Cout, H=W, B) one 3x3 / stride 1 / pad 1 bf16 Conv2d is run forward + backward
(dX and dW) on CPU bf16 and on device 'xdna', each (xdna: twice) in a FRESH subprocess so a hang
or crash only loses that shape (timeout 300 s per subprocess).  Per shape it records
  * whether conv3x3w engaged: `TRACE <phase>` lines (XDNA_CONV3X3W_TRACE=1) per timed iteration,
    fwd / dx / `TRACE dw`; a second xdna run with XDNA_CONV3X3W=0 gives the non-conv3x3w time;
  * first-call time (XDNA_CONV3X3W_SYNC_BUILD=1 -> includes the serial IRON stream builds), and
    the instruction streams that appeared in ~/.cache/torch-xdna2/conv3x3w{,_dw}/*/ meanwhile;
  * steady-state median ms of fwd+bwd (incl. reading dX and dW back to host) on CPU and xdna;
  * relative error (||a-ref||/||ref||) of y, dX, dW vs an fp32 CPU reference on the same
    bf16 inputs; > 5 % is flagged;
  * padding waste: padded-grid MACs / real MACs (cost-model formula of
    XdnaConv3x3wRuntime::cost_model_prefers_npu, and with the 2048-row group rounding).
Shapes with real MACs < 20e6 are skipped (below the family's default threshold).

usage (repo root, NIS venv python):
  PYTHONPATH=. XDNA_IRON_PYTHON=<mlir-aie python> python benchmarks/conv_shape_sweep.py \
      [--grid priority|full] [--shapes 128,128,18,32 ...] [--resume] [--iters 12]
Needs AC power (battery throttles the CPU and makes the comparison meaningless).
"""
import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "conv_shape_sweep.json")
CACHE = os.path.expanduser("~/.cache/torch-xdna2")
CHANNELS = [(64, 64), (128, 128), (256, 128)]
SPATIAL = [8, 9, 16, 18, 31, 32, 33, 36, 37]
BATCH = [1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 40, 64, 100]
MIN_MACS = 20e6
MAX_W, GROUP_OUT_ROWS = 36, 2048
TIMEOUT = 300


def real_macs(B, cin, cout, h):
    return B * h * h * cin * cout * 9


def padding_waste(B, cin, cout, h):
    """padded / real MACs for the forward pass (dX = same with cin/cout swapped)."""
    wp = h + 2
    chunks = math.ceil(cin / 64)
    padded = B * (h + 2) * wp * chunks * 64 * math.ceil(cout / 128) * 128 * 9
    groups = math.ceil(B * (h + 2) * wp / GROUP_OUT_ROWS)
    padded_grp = groups * GROUP_OUT_ROWS * chunks * 64 * math.ceil(cout / 128) * 128 * 9
    r = real_macs(B, cin, cout, h)
    return padded / r, padded_grp / r, groups, wp, chunks


# --------------------------------------------------------------------------- worker
def worker(mode, B, cin, cout, h, iters):
    """mode 'npu': CPU timing + xdna timing + errors; 'nonpu': xdna timing only."""
    import torch
    import torch.nn.functional as F

    res = {}
    torch.manual_seed(0)
    x = torch.randn(B, cin, h, h).bfloat16()
    w = (torch.randn(cout, cin, 3, 3) * (2.0 / (9 * cin)) ** 0.5).bfloat16()
    dy = torch.randn(B, cout, h, h).bfloat16()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def run_steps(dev):
        xx = x.detach().clone().to(dev).requires_grad_()
        ww = w.detach().clone().to(dev).requires_grad_()
        dd = dy.to(dev)

        def step():
            xx.grad = None
            ww.grad = None
            y = F.conv2d(xx, ww, None, padding=1)
            y.backward(dd)
            return y.detach().cpu(), xx.grad.cpu(), ww.grad.cpu()
        return step

    def timed(step, warm, n):
        t0 = time.perf_counter()
        out = step()
        res_first = (time.perf_counter() - t0) * 1e3
        for _ in range(warm - 1):
            out = step()
        print("MARK timed", file=sys.stderr, flush=True)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            step()
            ts.append((time.perf_counter() - t0) * 1e3)
        print("MARK done", file=sys.stderr, flush=True)
        return res_first, ts, out

    if mode == "npu":
        cstep = run_steps(torch.device("cpu"))
        _, cts, _ = timed(cstep, 2, iters)
        res["cpu_ms"] = statistics.median(cts)
        res["cpu_ms_min"] = min(cts)
        xr = x.float().requires_grad_()
        wr = w.float().requires_grad_()
        yr = F.conv2d(xr, wr, None, padding=1)
        yr.backward(dy.float())
        ref = (yr.detach(), xr.grad, wr.grad)
        cpu_out = cstep()
        res["cpu_err"] = [rel(a, b) for a, b in zip(cpu_out, ref)]

    import xdna_train
    xdna_train.register_xdna_device()
    xstep = run_steps(torch.device("xdna"))
    first, ts, out = timed(xstep, 3, iters)
    res["xdna_first_ms"] = first
    res["xdna_ms"] = statistics.median(ts)
    res["xdna_ms_min"] = min(ts)
    res["xdna_ms_max"] = max(ts)
    if mode == "npu":
        res["xdna_err"] = [rel(a, b) for a, b in zip(out, ref)]
        res["finite"] = all(bool(torch.isfinite(t.float()).all()) for t in out)
    print("RESULT " + json.dumps(res), flush=True)


# --------------------------------------------------------------------------- parent
def streams():
    return set(glob.glob(f"{CACHE}/conv3x3w/*/g*.bin")) | set(glob.glob(f"{CACHE}/conv3x3w_dw/*/g*.bin"))


def run_child(mode, B, cin, cout, h, iters):
    env = dict(os.environ)
    env["XDNA_CONV3X3W_TRACE"] = "1"
    env["XDNA_CONV3X3W_SYNC_BUILD"] = "1"
    if mode == "nonpu":
        env["XDNA_CONV3X3W"] = "0"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode, str(B), str(cin), str(cout), str(h),
           "--iters", str(iters)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired as e:
        return {"status": "timeout", "wall_s": time.time() - t0,
                "stderr_tail": (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))[-400:]}
    r = {"wall_s": time.time() - t0}
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not line:
        r.update(status="crash", rc=p.returncode, stderr_tail=p.stderr[-600:])
        return r
    r.update(json.loads(line[0][7:]))
    # count TRACE lines in the timed section only
    err = p.stderr.splitlines()
    cnt = {"fwd": 0, "dx": 0, "dw": 0}
    on = False
    for l in err:
        if l.startswith("MARK timed"):
            on = True
            cnt = {"fwd": 0, "dx": 0, "dw": 0}   # last 'timed' section = the xdna one
        elif l.startswith("MARK done"):
            on = False
        elif on and l.startswith("TRACE "):
            tok = l.split()[1]
            tok = {"npu_conv3x3w.fwd.run": "fwd", "npu_conv3x3w.dx.run": "dx"}.get(tok, tok)
            if tok in cnt:
                cnt[tok] += 1
    r["trace_per_iter"] = {k: v / iters for k, v in cnt.items()}
    r["trace_other"] = sorted({l.split()[1] for l in err if l.startswith("TRACE ")}
                              - set(cnt) - {"npu_conv3x3w.fwd.run", "npu_conv3x3w.dx.run"})
    r["status"] = "ok"
    return r


def grid(kind):
    pri = []
    for h in SPATIAL:           # spatial axis at batch 32, all channel configs
        for c in CHANNELS:
            pri.append((c[0], c[1], h, 32))
    for c, h in (((128, 128), 18), ((64, 64), 32)):   # full batch axis for two shapes
        for b in BATCH:
            pri.append((c[0], c[1], h, b))
    out = list(dict.fromkeys(pri))
    if kind == "full":
        for c in CHANNELS:
            for h in SPATIAL:
                for b in BATCH:
                    out.append((c[0], c[1], h, b))
        out = list(dict.fromkeys(out))
    return out


def summarize(rec):
    """derive flags from a measured record."""
    n, x = rec.get("npu_run", {}), rec.get("nonpu_run", {})
    if n.get("status") != "ok":
        return
    tp = n["trace_per_iter"]
    rec["engaged"] = {"fwd": tp["fwd"] >= 0.99, "dx": tp["dx"] >= 0.99, "dw": tp["dw"] > 0}
    rec["fallback"] = not (rec["engaged"]["fwd"] or rec["engaged"]["dx"])
    rec["cpu_ms"], rec["xdna_ms"] = n["cpu_ms"], n["xdna_ms"]
    rec["speedup_vs_cpu"] = n["cpu_ms"] / n["xdna_ms"]
    if x.get("status") == "ok":
        rec["xdna_nonconv3x3w_ms"] = x["xdna_ms"]
        rec["speedup_vs_nonconv3x3w"] = x["xdna_ms"] / n["xdna_ms"]
    rec["err_pct"] = [100 * e for e in n["xdna_err"]]          # y, dX, dW
    rec["err_flag_gt5pct"] = any(e > 0.05 for e in n["xdna_err"]) or not n["finite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=5, metavar=("MODE", "B", "CIN", "COUT", "H"))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--grid", choices=["priority", "full"], default="priority")
    ap.add_argument("--shapes", nargs="*", help="cin,cout,h,batch  (overrides --grid)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.worker:
        m, B, cin, cout, h = a.worker
        return worker(m, int(B), int(cin), int(cout), int(h), a.iters)

    with open("/sys/class/power_supply/AC/online") as f:
        if f.read().strip() != "1":
            sys.exit("not on AC power: refusing to benchmark")
    shapes = ([tuple(int(v) for v in s.split(",")) for s in a.shapes] if a.shapes else grid(a.grid))
    results = []
    if a.resume and os.path.exists(a.out):
        results = json.load(open(a.out))["results"]
    done = {(r["cin"], r["cout"], r["h"], r["batch"]) for r in results}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    start_streams = streams()
    consecutive_to = 0
    for cin, cout, h, B in shapes:
        if (cin, cout, h, B) in done:
            continue
        rec = {"cin": cin, "cout": cout, "h": h, "batch": B, "real_gmacs": real_macs(B, cin, cout, h) / 1e9}
        if real_macs(B, cin, cout, h) < MIN_MACS:
            rec["status"] = "skipped_lt_20M_macs"
            results.append(rec)
            continue
        for tag, (ci, co) in (("fwd", (cin, cout)), ("dx", (cout, cin))):
            pf, pg, groups, wp, chunks = padding_waste(B, ci, co, h)
            rec[f"waste_{tag}"] = pf
            rec[f"waste_{tag}_with_group_rounding"] = pg
            if tag == "fwd":
                rec["key_fwd"] = f"g{groups}_w{wp}_c{chunks}"
        rec["width_over_limit"] = h > MAX_W
        before = streams()
        t0 = time.time()
        rec["npu_run"] = run_child("npu", B, cin, cout, h, a.iters)
        new = sorted(os.path.basename(p)[:-4] for p in streams() - before)
        rec["new_streams"] = new
        rec["new_fwd_dx_streams"] = [k for k in new if k.count("_") == 2]
        rec["new_dw_streams"] = [k for k in new if k.count("_") > 2]
        rec["build_wall_s"] = rec["npu_run"].get("xdna_first_ms", 0) / 1e3 if new else 0.0
        if rec["npu_run"]["status"] == "ok":
            rec["nonpu_run"] = run_child("nonpu", B, cin, cout, h, a.iters)
        rec["status"] = rec["npu_run"]["status"]
        summarize(rec)
        rec["total_s"] = time.time() - t0
        results.append(rec)
        consecutive_to = consecutive_to + 1 if "timeout" in (rec["npu_run"]["status"],
                                                             rec.get("nonpu_run", {}).get("status")) else 0
        print(f"[{len(results)}/{len(shapes)}] {cin}->{cout} {h}x{h} B{B}: {rec['status']} "
              f"cpu {rec.get('cpu_ms', float('nan')):.2f} xdna {rec.get('xdna_ms', float('nan')):.2f} ms "
              f"engaged {rec.get('engaged')} err% {[round(e, 2) for e in rec.get('err_pct', [])]} "
              f"new streams {len(new)} ({rec['total_s']:.0f}s)", flush=True)
        json.dump({"streams_at_start": len(start_streams), "streams_now": len(streams()), "results": results},
                  open(a.out, "w"), indent=1)
        if consecutive_to >= 3:
            print("3 consecutive timeouts: NPU assumed wedged, stopping", flush=True)
            break
    json.dump({"streams_at_start": len(start_streams), "streams_now": len(streams()), "results": results},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
