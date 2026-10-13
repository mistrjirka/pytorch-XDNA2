#!/usr/bin/env python3
"""Deferred conv3x3w weight gradients started early on a helper thread.

After the last NPU input-gradient call of a step the NPU is idle while the CPU
finishes backward, so the backend starts the deferred NPU dW passes then
(learned from the previous steps; XDNA_ASYNC_DW=0 keeps the lazy behaviour).
The result must not depend on that:

1. identical losses and final weights with XDNA_ASYNC_DW=0 and 1;
2. the helper thread really started jobs (kicks > 0) and the policy learned the
   step length (2 NPU dX calls);
3. freeing the gradients right after backward, while the helper may still be
   running, neither crashes nor hangs;
4. a step structure that changes after the policy learned (a third NPU conv)
   still gives identical results, although the early start then mispredicts.

The parent process never touches the NPU (quick_check already holds it and a
third process would exhaust the hardware contexts); every run is a child.
Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import os
import subprocess
import sys

HERE = os.path.abspath(__file__)


def child(async_on):
    os.environ["XDNA_ASYNC_DW"] = "1" if async_on else "0"
    os.environ.setdefault("XDNA_AUTO_SETUP", "0")
    os.environ.setdefault("XDNA_CONV3X3W_SYNC_BUILD", "1")
    os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")
    os.environ.setdefault("XDNA_CONV3X3W_FORCE", "1")
    os.environ.setdefault("XDNA_CONV3X3W_MIN_MACS", "0")
    os.environ["XDNA_CONV3X3W_DW_MIN_MACS"] = "0"
    os.environ["XDNA_NPU_MODE"] = "2"

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import xdna_train

    xdna_train.register_xdna_device()

    class Net(nn.Module):
        """CPU stem, then NPU 3x3 convs, so backward ends with a CPU tail."""

        def __init__(self, npu_convs):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 128, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
            self.mid = nn.ModuleList(
                [nn.Conv2d(128, 128, 3, padding=1, bias=False) for _ in range(npu_convs)])
            self.head = nn.Conv2d(128, 3, 1)

        def forward(self, x):
            y = self.stem(x)
            for conv in self.mid:
                y = F.leaky_relu(conv(y), 0.2)
            return self.head(y)

    def run(npu_convs, steps, seed, tag):
        torch.manual_seed(seed)
        model = Net(npu_convs).bfloat16().to("xdna")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(seed + 1)
        x = torch.rand(16, 3, 36, 36, generator=g).bfloat16().to("xdna")
        y = (x.flip(1) * 0.5 + 0.25).bfloat16().to("xdna")
        for i in range(steps):
            loss = F.mse_loss(model(x).float(), y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"LOSS {tag} {i} {loss.item():.9g}", flush=True)
        total = sum(p.detach().float().double().sum().item() for p in model.parameters())
        mag = sum(p.detach().float().double().abs().sum().item() for p in model.parameters())
        print(f"CKSUM {tag} {total!r} {mag!r}", flush=True)
        return model, opt, x, y

    run(2, 6, 0, "A")                       # 1: equal results, 2: kicks
    print("STATS_A", dict(xdna_train.async_dw_stats()), flush=True)

    # 3: free the gradients while the helper may still be running dW.
    model, opt, x, y = run(2, 4, 1, "B")    # learns the step length again
    for i in range(6):
        loss = F.mse_loss(model(x).float(), y.float())
        loss.backward()
        opt.zero_grad(set_to_none=True)     # grads dropped unread right after backward
        del loss
    loss = F.mse_loss(model(x).float(), y.float())
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"LOSS C 0 {loss.item():.9g}", flush=True)
    print("SURVIVED_FREE", flush=True)

    # 4: the step now has 3 NPU convs: the learned count (2) mispredicts once.
    run(3, 4, 2, "D")
    print("STATS_END", dict(xdna_train.async_dw_stats()), flush=True)
    print("SURVIVED", flush=True)


def spawn(async_on):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, HERE, "--child", "1" if async_on else "0"],
                       capture_output=True, text=True, env=env, timeout=900)
    return r


def lines(out, prefix):
    return [ln for ln in out.splitlines() if ln.startswith(prefix)]


def main():
    failures = []

    def check(name, cond, extra=""):
        print(f"{name}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
        if not cond:
            failures.append(name)

    runs = {}
    for label, flag in (("sync", False), ("async", True)):
        r = spawn(flag)
        runs[label] = r
        ok = r.returncode == 0 and "SURVIVED" in r.stdout and "SURVIVED_FREE" in r.stdout
        check(f"{label}: child ran to the end", ok,
              "" if ok else f"exit {r.returncode}\n{r.stderr[-1500:]}")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)

    so, ao = runs["sync"].stdout, runs["async"].stdout
    check("losses identical (sync vs async)", lines(so, "LOSS") == lines(ao, "LOSS"),
          "" if lines(so, "LOSS") == lines(ao, "LOSS") else
          f"\n sync  {lines(so, 'LOSS')[:8]}\n async {lines(ao, 'LOSS')[:8]}")
    check("final weights identical (sync vs async)", lines(so, "CKSUM") == lines(ao, "CKSUM"),
          "" if lines(so, "CKSUM") == lines(ao, "CKSUM") else
          f"\n sync  {lines(so, 'CKSUM')}\n async {lines(ao, 'CKSUM')}")

    def stats(out, prefix):
        ln = lines(out, prefix)
        return eval(ln[-1].split(" ", 1)[1]) if ln else {}

    sa = stats(ao, "STATS_A")
    check("async: step length learned", sa.get("dx_per_step") == 2, str(sa))
    check("async: helper thread started jobs", sa.get("kicks", 0) >= 1 and sa.get("started_by_helper", 0) >= 1, str(sa))
    ss = stats(so, "STATS_END")
    check("sync: no early starts", ss.get("kicks") == 0 and ss.get("started_by_helper") == 0, str(ss))
    se = stats(ao, "STATS_END")
    check("async: still started jobs after the structure changed", se.get("started_by_helper", 0) >= sa.get("started_by_helper", 0), str(se))

    print("ALL OK" if not failures else f"FAILED: {failures}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2] == "1")
    else:
        main()
