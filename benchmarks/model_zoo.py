#!/usr/bin/env python3
"""Diverse training benchmark: CPU BF16 vs XDNA BF16 on realistic model sizes.

Each model trains for a few steps on fixed synthetic data on both devices, in
separate processes (so one device's thread/allocator state cannot affect the
other).  Reports median train-step time, speedup and whether the XDNA loss
tracks CPU.  The point is generalization: a backend change should be judged on
this whole matrix, not on one network.

    PYTHONPATH=/path/to/torch-xdna2 python benchmarks/model_zoo.py
    python benchmarks/model_zoo.py --models resnet18,lstm --steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Models.  Each factory returns (model, make_batch(gen) -> (x, y), loss_fn).
# --------------------------------------------------------------------------

def _resnet18():
    import torchvision
    m = torchvision.models.resnet18(num_classes=100)
    return m, lambda g: (torch.randn(32, 3, 64, 64, generator=g),
                         torch.randint(0, 100, (32,), generator=g)), F.cross_entropy


def _mobilenet_v3():
    import torchvision
    m = torchvision.models.mobilenet_v3_small(num_classes=100)
    return m, lambda g: (torch.randn(32, 3, 96, 96, generator=g),
                         torch.randint(0, 100, (32,), generator=g)), F.cross_entropy


class _UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.e1, self.e2, self.e3 = block(3, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.d2, self.d1 = block(6 * c, 2 * c), block(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        e = self.d1(torch.cat([F.interpolate(d, scale_factor=2), a], 1))
        return self.out(e)


def _unet():
    return _UNet(), lambda g: (torch.randn(8, 3, 96, 96, generator=g),
                               (torch.rand(8, 1, 96, 96, generator=g) > 0.5).float()), \
        F.binary_cross_entropy_with_logits


class _Transformer(nn.Module):
    def __init__(self, vocab=4096, d=256, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 128, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.enc(self.emb(x) + self.pos)).flatten(0, 1)


def _transformer():
    return _Transformer(), lambda g: (torch.randint(0, 4096, (16, 128), generator=g),
                                      torch.randint(0, 4096, (16 * 128,), generator=g)), F.cross_entropy


def _mlp():
    m = nn.Sequential(nn.Linear(784, 1024), nn.GELU(), nn.LayerNorm(1024),
                      nn.Linear(1024, 1024), nn.GELU(), nn.LayerNorm(1024),
                      nn.Linear(1024, 10))
    return m, lambda g: (torch.randn(256, 784, generator=g),
                         torch.randint(0, 10, (256,), generator=g)), F.cross_entropy


class _LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 256, num_layers=2, batch_first=True)
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])


def _lstm():
    return _LSTM(), lambda g: (torch.randn(32, 64, 64, generator=g),
                               torch.randint(0, 10, (32,), generator=g)), F.cross_entropy


def _conv1d_audio():
    layers, c = [], 1
    for o, s in [(32, 4), (64, 2), (64, 2), (128, 2), (128, 2)]:
        layers += [nn.Conv1d(c, o, 9, stride=s, padding=4), nn.BatchNorm1d(o), nn.SiLU()]
        c = o
    m = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, 35))
    return m, lambda g: (torch.randn(16, 1, 16000, generator=g),
                         torch.randint(0, 35, (16,), generator=g)), F.cross_entropy


def _conv_autoencoder():
    m = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
    return m, lambda g: (torch.rand(16, 3, 128, 128, generator=g),) * 2, F.mse_loss


class _EmbedBag(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50000, 128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 20))

    def forward(self, x):
        return self.mlp(self.emb(x).mean(1))


def _embedding():
    return _EmbedBag(), lambda g: (torch.randint(0, 50000, (128, 64), generator=g),
                                   torch.randint(0, 20, (128,), generator=g)), F.cross_entropy


def _nis_mynet():
    sys.path.insert(0, "/home/jirka/programovani/magistr/nis/lab02_cnn")
    from models import MyNet
    return MyNet(3, 3, 5), lambda g: (torch.randn(40, 3, 36, 36, generator=g),
                                      torch.rand(40, 3, 36, 36, generator=g)), F.l1_loss


MODELS = {
    "resnet18": _resnet18,
    "mobilenet_v3": _mobilenet_v3,
    "unet": _unet,
    "transformer": _transformer,
    "mlp": _mlp,
    "lstm": _lstm,
    "conv1d_audio": _conv1d_audio,
    "conv_autoencoder": _conv_autoencoder,
    "embedding": _embedding,
    "nis_mynet": _nis_mynet,
}


# --------------------------------------------------------------------------
# Worker: one model on one device, in its own process.
# --------------------------------------------------------------------------

def _worker(name: str, device: str, steps: int, warmup: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    if device == "xdna":
        import xdna_train
        xdna_train.register_xdna_device()
    sync = (lambda: torch.xdna.synchronize()) if device == "xdna" else (lambda: None)
    torch.manual_seed(0)
    model, make_batch, loss_fn = MODELS[name]()
    model = model.to(device=device, dtype=torch.bfloat16).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(1)
    batches = [make_batch(gen) for _ in range(2)]

    def to_dev(t):
        t = t.to(device)
        return t.to(torch.bfloat16) if t.is_floating_point() else t

    batches = [tuple(to_dev(t) for t in b) for b in batches]
    times, losses = [], []
    for i in range(warmup + steps):
        x, y = batches[i % len(batches)]
        sync()
        t0 = time.perf_counter()
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.item())
        t1 = time.perf_counter()
        losses.append(value)
        if i >= warmup:
            times.append((t1 - t0) * 1e3)
    return {"median_ms": statistics.median(times), "losses": losses}


def _run_worker(name, device, args) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", name, device,
           "--steps", str(args.steps), "--warmup", str(args.warmup), "--threads", str(args.threads)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    for line in p.stdout.splitlines():
        if line.startswith("ZOO_RESULT="):
            return json.loads(line[len("ZOO_RESULT="):])
    err = (p.stderr.strip().splitlines() or ["no output"])[-1]
    return {"error": err[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", default="")
    ap.add_argument("--_worker", nargs=2)
    args = ap.parse_args()

    if args._worker:
        res = _worker(*args._worker, args.steps, args.warmup, args.threads)
        print("ZOO_RESULT=" + json.dumps(res), flush=True)
        return

    rows = []
    print(f"{'model':18s} {'cpu ms':>9s} {'xdna ms':>9s} {'speedup':>8s}  {'loss cpu→xdna (first / last)':34s} status")
    for name in args.models.split(","):
        cpu = _run_worker(name, "cpu", args)
        npu = _run_worker(name, "xdna", args)
        row = {"model": name, "cpu": cpu, "xdna": npu}
        if "error" in cpu or "error" in npu:
            status = "ERROR: " + (npu.get("error") or cpu.get("error"))
            print(f"{name:18s} {cpu.get('median_ms', float('nan')):9.1f} "
                  f"{npu.get('median_ms', float('nan')):9.1f} {'-':>8s}  {'':34s} {status}")
        else:
            speed = cpu["median_ms"] / npu["median_ms"]
            l0c, l0x = cpu["losses"][0], npu["losses"][0]
            lnc, lnx = cpu["losses"][-1], npu["losses"][-1]
            # BF16 training diverges slowly between backends; the first loss
            # must match closely, the last only needs to be finite and close.
            first_ok = abs(l0c - l0x) <= 0.02 * max(1.0, abs(l0c))
            last_ok = abs(lnc - lnx) <= 0.10 * max(1.0, abs(lnc)) and lnx == lnx
            status = "ok" if first_ok and last_ok else "LOSS MISMATCH"
            row.update(speedup=speed, status=status)
            print(f"{name:18s} {cpu['median_ms']:9.1f} {npu['median_ms']:9.1f} {speed:7.2f}x  "
                  f"{l0c:.4g}→{l0x:.4g} / {lnc:.4g}→{lnx:.4g}".ljust(34 + 40) + f" {status}")
        rows.append(row)
    ok = [r["speedup"] for r in rows if "speedup" in r]
    if ok:
        geo = statistics.geometric_mean(ok)
        print(f"geomean speedup over {len(ok)} models: {geo:.2f}x")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
