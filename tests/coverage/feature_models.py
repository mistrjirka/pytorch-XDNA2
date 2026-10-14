"""End-to-end models that use the newer backend features and have to LEARN something.

Unit tests check operators; these check that a model built on each feature trains on
XDNA to the same quality as the reference.  Every case trains on synthetic but
learnable data, then checks (a) the loss falls a lot and the held-out metric beats
chance by a clear margin, and (b) XDNA ends within a stated gap of the reference.

  autocast_cnn_classifier     torch.autocast("xdna") BF16 compute, FP32 master weights, AdamW
  autocast_style_net          the benchmark generator under autocast (convs may reach the NPU)
  conv3d_avgpool_video        3-D CNN with BF16 AvgPool3d (no BF16 CPU kernel exists)
  sparse_recommender          CPU-resident sparse embeddings + XDNA dense tower, SparseAdam
  fourier_operator_learning   complex trainable weights (Adam updates through view_as_real)

    python tests/coverage/feature_models.py --out /tmp/feature_models.json [--case NAME]

Output uses the coverage-gate schema (case, dtype, status, metrics, npu_runs).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")  # counts NPU runs

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))


def gen(seed):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------- autocast CNN

class QuadrantNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16)
        self.c2, self.b2 = nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 4 * 4, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.b2(self.c2(x))), 4)
        return self.fc(x.flatten(1))


def quadrant_batch(n, g):
    x = torch.rand(n, 1, 16, 16, generator=g) * 0.3
    y = torch.randint(0, 4, (n,), generator=g)
    for i in range(n):
        r, c = divmod(int(y[i]), 2)
        x[i, :, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] += 0.5
    return x, y


def case_autocast_cnn(device):
    torch.manual_seed(0)
    model = QuadrantNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = gen(1)
    first = last = None
    bf16_seen = False
    for step in range(120):
        x, y = quadrant_batch(64, g)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x)
            bf16_seen |= logits.dtype == torch.bfloat16
            loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    model.eval()
    xt, yt = quadrant_batch(512, gen(99))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        acc = (model(xt.to(device)).argmax(1).cpu() == yt).float().mean().item()
    fp32 = all(p.dtype == torch.float32 for p in model.parameters())
    return dict(first_loss=first, final_loss=last, metric=acc, metric_name="accuracy", chance=0.25,
                extra=dict(bf16_compute=bf16_seen, fp32_master_weights=fp32))


# --------------------------------------------------------------- autocast generator

def case_autocast_style_net(device):
    from style_net import StyleNet
    torch.manual_seed(0)
    model = StyleNet(3, 3, 5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = gen(2)
    first = last = None
    for step in range(12):
        x = torch.rand(40, 3, 36, 36, generator=g)
        t = (x * torch.tensor([1.2, 0.9, 1.1]).view(1, 3, 1, 1)).clamp(0, 1)
        x, t = x.to(device), t.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = F.l1_loss(model(x).float(), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    return dict(first_loss=first, final_loss=last, metric=last, metric_name="final_l1", chance=None,
                extra=dict(fp32_master_weights=all(p.dtype == torch.float32 for p in model.parameters())))


# --------------------------------------------------------------- 3-D CNN with AvgPool3d

class VideoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1 = nn.Conv3d(1, 8, 3, padding=1), nn.BatchNorm3d(8)
        self.c2 = nn.Conv3d(8, 16, 3, padding=1)
        self.pool = nn.AvgPool3d(2)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.c1(x))))
        x = self.pool(F.relu(self.c2(x)))
        return self.fc(x.mean(dim=(2, 3, 4)))


def video_batch(n, g):
    """A bright blob drifting left->right (class 0) or right->left (class 1) over 8 frames."""
    x = torch.rand(n, 1, 8, 16, 16, generator=g) * 0.2
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        for t in range(8):
            pos = 2 + 2 * t if y[i] == 0 else 14 - 2 * t
            x[i, 0, t, 6:10, max(pos - 1, 0):pos + 2] += 1.0
    return x, y


def case_video(device, dtype):
    torch.manual_seed(0)
    model = VideoNet().to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = gen(3)
    first = last = None
    for step in range(70):
        x, y = video_batch(16, g)
        x, y = x.to(device=device, dtype=dtype), y.to(device)
        loss = F.cross_entropy(model(x).float(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    model.eval()
    xt, yt = video_batch(128, gen(98))
    with torch.no_grad():
        acc = (model(xt.to(device=device, dtype=dtype)).argmax(1).cpu() == yt).float().mean().item()
    return dict(first_loss=first, final_loss=last, metric=acc, metric_name="accuracy", chance=0.5, extra={})


# --------------------------------------------------------------- sparse recommender

class Recommender(nn.Module):
    def __init__(self, users, items, dim, device):
        super().__init__()
        from xdna_train.cpu_sparse import CpuSparseEmbedding
        self.device_name = device
        self.user = CpuSparseEmbedding(users, dim)
        self.item = CpuSparseEmbedding(items, dim)
        self.tower = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u, i):
        d = None if self.device_name == "cpu" else self.device_name
        return self.tower(torch.cat([self.user(u, device=d), self.item(i, device=d)], 1)).squeeze(1)


def case_recommender(device):
    from xdna_train.cpu_sparse import split_parameters
    users, items, dim = 1000, 300, 12
    g = gen(4)
    ub, ib = torch.randn(users, generator=g), torch.randn(items, generator=g)

    def pairs(n, seed):
        gg = gen(seed)
        u, i = torch.randint(0, users, (n,), generator=gg), torch.randint(0, items, (n,), generator=gg)
        return u, i, ((ub[u] + ib[i]) > 0).float()

    torch.manual_seed(0)
    model = Recommender(users, items, dim, device)
    model.tower.to(device)
    sp, dn = split_parameters(model)
    opts = [torch.optim.SparseAdam(sp, lr=0.02), torch.optim.Adam(dn, lr=0.01)]
    table_before = model.user.weight.detach().clone()
    first = last = None
    gg = gen(5)
    for step in range(500):
        u = torch.randint(0, users, (256,), generator=gg)
        i = torch.randint(0, items, (256,), generator=gg)
        y = ((ub[u] + ib[i]) > 0).float()
        for o in opts:
            o.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(u, i).float(), y.to(model.tower[0].weight.device))
        loss.backward()
        for o in opts:
            o.step()
        first = float(loss) if first is None else first
        last = float(loss)
    ut, it, yt = pairs(4000, 97)
    with torch.no_grad():
        acc = ((model(ut, it).float().cpu() > 0).float() == yt).float().mean().item()
    return dict(first_loss=first, final_loss=last, metric=acc, metric_name="held_out_accuracy", chance=0.5,
                extra=dict(table_on_cpu=model.user.weight.device.type == "cpu",
                           table_changed=not torch.equal(table_before, model.user.weight.detach())))


# --------------------------------------------------------------- Fourier operator

class SpectralFilter(nn.Module):
    def __init__(self, modes=4):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(0.1 * torch.randn(modes, modes, dtype=torch.complex64))

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros_like(xf)
        out[:, :, :self.modes, :self.modes] = xf[:, :, :self.modes, :self.modes] * self.w
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


def case_fourier(device):
    g = gen(6)
    true_w = torch.randn(4, 4, dtype=torch.complex64, generator=g)
    target_layer = SpectralFilter()
    with torch.no_grad():
        target_layer.w.copy_(true_w)
    torch.manual_seed(0)
    model = SpectralFilter().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first = last = None
    gg = gen(7)
    for step in range(150):
        x = torch.randn(8, 2, 16, 16, generator=gg)
        with torch.no_grad():
            t = target_layer(x)
        loss = F.mse_loss(model(x.to(device)), t.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    return dict(first_loss=first, final_loss=last, metric=last, metric_name="final_mse", chance=None, extra={})


# --------------------------------------------------------------- harness

CASES = {
    # name: (family, runner(device) -> metrics, dtype label, higher_is_better, reference)
    "autocast_cnn_classifier": ("Autocast (BF16 compute, FP32 master weights)", case_autocast_cnn, "bf16-autocast", True, "cpu autocast"),
    "autocast_style_net": ("Autocast on the benchmark generator", case_autocast_style_net, "bf16-autocast", False, "cpu autocast"),
    "conv3d_avgpool_video": ("3-D CNN with BF16 AvgPool3d", None, "bf16", True, "cpu fp32"),
    "sparse_recommender": ("CPU sparse embeddings + XDNA dense tower", case_recommender, "fp32", True, "cpu"),
    "fourier_operator_learning": ("Complex trainable weights", case_fourier, "fp32", False, "cpu"),
}


def evaluate(name, x, r):
    """Compare XDNA metrics ``x`` with reference ``r`` and return (status, checks)."""
    checks = {}
    checks["loss_fell"] = x["final_loss"] < 0.6 * x["first_loss"]
    higher = CASES[name][3]
    if x["chance"] is not None:
        checks["beats_chance_by_0.15"] = x["metric"] > x["chance"] + 0.15
    gap = abs(x["metric"] - r["metric"]) / max(abs(r["metric"]), 1e-6)
    if name in ("autocast_cnn_classifier", "conv3d_avgpool_video", "sparse_recommender"):
        checks["metric_within_0.08_of_reference"] = abs(x["metric"] - r["metric"]) <= 0.08
    elif name == "autocast_style_net":
        checks["loss_within_10pct_of_reference"] = abs(x["final_loss"] - r["final_loss"]) <= 0.10 * r["final_loss"]
        checks["loss_fell"] = x["final_loss"] < 0.9 * x["first_loss"]
    else:  # Fourier: both must recover the filter
        checks["filter_learned_mse_below_1e-4"] = x["final_loss"] < 1e-4 and r["final_loss"] < 1e-4
    for k, v in x.get("extra", {}).items():
        checks[k] = bool(v)
    return ("pass" if all(checks.values()) else "numeric_mismatch"), checks, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case")
    a = ap.parse_args()
    torch.set_num_threads(4)
    import xdna_train
    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)
    results = []
    for name, (family, runner, label, _hib, ref_name) in CASES.items():
        if a.case and a.case != name:
            continue
        xdna_train.reset_mapped_cpu_stats()
        rec = dict(case=name, dtype=label, family=family, reference=ref_name)
        t0 = time.time()
        try:
            if name == "conv3d_avgpool_video":
                x = case_video("xdna", torch.bfloat16)
                r = case_video("cpu", torch.float32)  # the CPU has no BF16 AvgPool3d
            else:
                dev = "xdna"
                x = runner(dev)
                r = runner("cpu")
            status, checks, gap = evaluate(name, x, r)
            rec.update(status=status, checks=checks, xdna=x, reference_result=r, metric_gap=gap)
        except Exception as e:  # noqa: BLE001
            rec.update(status="xdna_error", error=str(e)[:800], traceback=traceback.format_exc())
        rec["seconds"] = round(time.time() - t0, 1)
        rec["npu_runs"] = {k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()
                           if k.startswith("npu") and k.endswith(".run")}
        results.append(rec)
        Path(a.out).write_text(json.dumps({"torch": torch.__version__, "scope": "end-to-end feature models",
                                           "results": results}, indent=2, default=str) + "\n")
        xm = rec.get("xdna", {})
        rm = rec.get("reference_result", {})
        print(f"{name}: {rec['status']}  xdna {xm.get('metric_name')}={xm.get('metric')}  "
              f"ref={rm.get('metric')}  loss {xm.get('first_loss')}->{xm.get('final_loss')}  "
              f"npu_runs={rec['npu_runs']}  {rec['seconds']}s", flush=True)
        if rec["status"] != "pass":
            print("   checks/error:", rec.get("checks", rec.get("error")), flush=True)
    sys.exit(0 if all(r["status"] == "pass" for r in results) else 1)


if __name__ == "__main__":
    main()
