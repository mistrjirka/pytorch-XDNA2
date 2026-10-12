#!/usr/bin/env python3
"""Writes that race a pending host result or a stale NPU weight cache.

1. copy_ / slice assignment into a conv weight gradient whose NPU dW is still
   deferred must not be overwritten when the dW runs later.
2. In-place writes through `.data` (and the fused optimizer kernel called
   directly) must invalidate the encoded conv3x3w weight cache.
3. Freeing a conv weight gradient whose deferred dW failed (or never ran) must
   not abort the process; a never-read gradient must not dispatch its dW.

Run with the NIS venv python and PYTHONPATH=<repo>.  Case 3 runs in
subprocesses (`--child <scenario>`), so an abort shows up as a failed case.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")
os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")
os.environ.setdefault("XDNA_CONV3X3W_MIN_MACS", "0")
os.environ.setdefault("XDNA_CONV3X3W_DW_MIN_MACS", "0")

failures = []


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}", flush=True)
    if not cond:
        failures.append(name)


def dw_stream_dir():
    """The default dW stream cache (device.py: name + source hash)."""
    designs = os.path.join(REPO, "xdna_train", "_designs")
    h = hashlib.sha256()
    for name in ("conv3x3w_dw.py", "conv3x3w_dw_kernel.cc"):
        with open(os.path.join(designs, name), "rb") as f:
            h.update(f.read())
    root = os.environ.get("TORCH_XDNA2_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "torch-xdna2")
    return os.path.join(root, "conv3x3w_dw", h.hexdigest()[:16])


def child(scenario):
    """Runs one failure/lifetime scenario in this process (see case 3)."""
    import glob
    import torch
    import torch.nn.functional as F
    import xdna_train

    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)

    def dw_runs():
        return xdna_train.mapped_cpu_stats().get("npu_conv3x3w.dw.run", {}).get("calls", 0)

    x = torch.ones(16, 192, 36, 36).bfloat16().to("xdna")
    w = torch.ones(128, 192, 3, 3).bfloat16().to("xdna").requires_grad_()
    go = torch.ones(16, 128, 36, 36).bfloat16().to("xdna")
    g = torch.autograd.grad(F.conv2d(x, w, padding=1), w, go)[0]
    runs0 = dw_runs()
    if scenario != "ok_del_unread":
        # Make the deferred job fail in read_file(), before any NPU dispatch.
        for f in glob.glob(os.path.join(os.environ["XDNA_CONV3X3W_DW_DIR"], "g*.bin")):
            os.remove(f)
    if scenario == "fail_read_then_del":
        for i in range(2):
            try:
                g.cpu()
                print("read did not raise", flush=True)
            except Exception as e:
                print(f"read #{i + 1} raised {type(e).__name__}", flush=True)
    del g
    print(f"DW_RUN_DELTA={dw_runs() - runs0}", flush=True)
    print("SURVIVED", flush=True)


def run_child(scenario, stream_dir):
    env = dict(os.environ)
    env["XDNA_CONV3X3W_DW_DIR"] = stream_dir
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", scenario],
                       capture_output=True, text=True, env=env)
    return r


def part_grad_writes():
    """1: writes into a gradient whose dW is still deferred."""
    import torch
    import torch.nn.functional as F
    import xdna_train

    def dw_runs():
        return xdna_train.mapped_cpu_stats().get("npu_conv3x3w.dw.run", {}).get("calls", 0)

    def mk():
        x = torch.ones(16, 192, 36, 36).bfloat16().to("xdna")
        w = torch.ones(128, 192, 3, 3).bfloat16().to("xdna").requires_grad_()
        go = torch.ones(16, 128, 36, 36).bfloat16().to("xdna")
        return torch.autograd.grad(F.conv2d(x, w, padding=1), w, go)[0]

    cases = [
        ("grad copy_ xdna", lambda g: g.copy_(torch.full_like(g, .25))),
        ("grad copy_ cpu", lambda g: g.copy_(torch.full(g.shape, .25).bfloat16())),
        ("grad slice assign", lambda g: g.__setitem__(slice(None), torch.full_like(g, .25))),
        ("grad view copy_", lambda g: g.view(-1).copy_(
            torch.full((g.numel(),), .25).bfloat16().to("xdna"))),
    ]
    for name, op in cases:
        g = mk()
        before = dw_runs()
        op(g)
        vals = g.cpu().float()
        ran = dw_runs() > before
        check(f"{name}: keeps written 0.25 and dW ran",
              ran and bool((vals == 0.25).all()))


def part_data_writes():
    """2: `.data` writes and the direct fused optimizer must refresh the cache."""
    import torch
    import torch.nn.functional as F
    import xdna_train
    from xdna_train import device as xd

    x = torch.randn(16, 128, 36, 36).bfloat16().to("xdna")
    other = torch.randn(128, 128, 3, 3).bfloat16().to("xdna")

    def fwd_runs():
        return xd.mapped_cpu_stats().get("npu_conv3x3w.fwd.run", {}).get("calls", 0)

    def fused_adam(w):
        grad = torch.full_like(w, 0.5)
        st = [torch.zeros_like(w), torch.zeros_like(w)]
        torch._fused_adam_([w], [grad], [st[0]], [st[1]], [], [torch.tensor(1.)],
                           lr=0.05, beta1=0.9, beta2=0.999, weight_decay=0.,
                           eps=1e-8, amsgrad=False, maximize=False)

    cases = [
        ("w.data.add_", lambda w: w.data.add_(0.05)),
        ("w.data.mul_", lambda w: w.data.mul_(1.5)),
        ("w.data.copy_", lambda w: w.data.copy_(other)),
        ("w.data.zero_", lambda w: w.data.zero_()),
        ("_fused_adam_", fused_adam),
    ]
    torch.manual_seed(0)
    for name, op in cases:
        conv = torch.nn.Conv2d(128, 128, 3, padding=1, bias=False).bfloat16().to("xdna")
        with torch.no_grad():
            conv(x)  # encodes the weight for the cache
            op(conv.weight)
            before = fwd_runs()
            y = conv(x)
        ran = fwd_runs() > before
        ref = F.conv2d(x.cpu().float(), conv.weight.detach().cpu().float(), padding=1)
        err = ((y.cpu().float() - ref).norm() / (ref.norm() + 1e-6)).item()
        check(f"{name}: NPU fwd ran and matches new weights (rel {err:.4f})",
              ran and err < 0.03)


def part_lifetime():
    """3: freeing a deferred gradient must not abort the process."""
    stream_dir = dw_stream_dir()
    if not os.path.isdir(stream_dir):
        check(f"dW stream cache present ({stream_dir})", False)
        return
    scratch = tempfile.mkdtemp(prefix="pending_writes_")
    try:
        copies = []
        for i, scenario in enumerate(("fail_read_then_del", "fail_del_unread", "ok_del_unread")):
            d = os.path.join(scratch, f"dw{i}")
            shutil.copytree(stream_dir, d)
            copies.append((scenario, d))
        for scenario, d in copies:
            r = run_child(scenario, d)
            out, err = r.stdout, r.stderr
            aborted = "terminate" in err or r.returncode != 0
            if scenario == "fail_read_then_del":
                injected = "read #1 raised" in out
                check(f"{scenario}: failure injected", injected)
                check(f"{scenario}: deleter does not abort",
                      not aborted and "SURVIVED" in out)
            elif scenario == "fail_del_unread":
                check(f"{scenario}: deleter does not abort",
                      not aborted and "SURVIVED" in out)
            else:
                delta = [ln for ln in out.splitlines() if ln.startswith("DW_RUN_DELTA=")]
                check(f"{scenario}: unread gradient freed without NPU dW (delta {delta})",
                      not aborted and delta == ["DW_RUN_DELTA=0"])
            if aborted:
                print(err[-800:], flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_part(part):
    """Runs one in-process part in a child so this process never holds the NPU:
    quick_check already holds it, and a third process exhausts hardware contexts."""
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", part],
                       capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
        failures.append(f"part {part} (exit {r.returncode})")


def part_main(part):
    import xdna_train  # noqa: F401  (registers the device; compiles on first import)

    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)
    {"grad": part_grad_writes, "data": part_data_writes}[part]()
    sys.exit(0 if not failures else 1)


def main():
    run_part("grad")   # also builds the dW stream the lifetime scenarios copy
    run_part("data")
    part_lifetime()
    print("ALL OK" if not failures else f"FAILED: {failures}", flush=True)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--part":
        part_main(sys.argv[2])
    else:
        main()
