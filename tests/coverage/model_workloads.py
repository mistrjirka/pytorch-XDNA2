"""Complete small training workloads for model families that operator smoke tests miss.

Each workload trains a whole (small) model for several optimizer steps on the CPU and
on XDNA from identical initial weights and data, then compares the loss trajectory
and the final parameters.  The families are the ones the coverage roadmap lists as
"poorly tested": selective state-space scan, routed mixture-of-experts, learned
spectral (complex-weight) layers, a detection head with ROI pooling, and
fake-quantization training.  A passing result means the model trains the same way as
on CPU; it does not mean the NPU accelerated it (see ``npu_runs``).

    python tests/coverage/model_workloads.py --out /tmp/workloads.json [--case NAME]

The output uses the same schema as network_matrix.py so tests/coverage/gate.py can
check it against the manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
import traceback
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- models

class SelectiveSSM(nn.Module):
    """Mamba-style selective scan: input-dependent step size, B and C; sequential recurrence."""

    def __init__(self, d=8, n=4):
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.dt = nn.Linear(d, d)
        self.b = nn.Linear(d, n)
        self.c = nn.Linear(d, n)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
        self.d_skip = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)
        self.n = n

    def forward(self, x, h0=None):  # x: [B, T, d]
        u = F.silu(self.inp(x))
        dt = F.softplus(self.dt(x))                      # [B,T,d]
        bmat, cmat = self.b(x), self.c(x)                # [B,T,n]
        a = -torch.exp(self.a_log)                       # [d,n]
        h = x.new_zeros(x.shape[0], x.shape[2], self.n) if h0 is None else h0
        ys = []
        for t in range(x.shape[1]):
            da = torch.exp(dt[:, t, :, None] * a)        # [B,d,n]
            db = dt[:, t, :, None] * bmat[:, t, None, :]
            h = da * h + db * u[:, t, :, None]
            ys.append((h * cmat[:, t, None, :]).sum(-1) + self.d_skip * u[:, t])
        return self.out(torch.stack(ys, 1))


class TopKMoE(nn.Module):
    """Two-expert-per-token routing with explicit dispatch; one expert is starved on purpose."""

    def __init__(self, d=8, experts=4, k=2):
        super().__init__()
        self.router = nn.Linear(d, experts)
        with torch.no_grad():
            self.router.bias.copy_(torch.tensor([3.0, 2.0, -9.0, 0.0][:experts]))  # expert 2 gets no tokens
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                      for _ in range(experts)])
        self.k = k
        self.aux = torch.zeros(())

    def forward(self, x):  # x: [N, d]
        probs = F.softmax(self.router(x), dim=-1)
        w, idx = probs.topk(self.k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        load = probs.mean(0)
        self.aux = (load * load).sum() * len(self.experts)
        for e, expert in enumerate(self.experts):
            token, slot = (idx == e).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            y = expert(x.index_select(0, token))
            out = out.index_add(0, token, y * w[token, slot, None])
        return out


class FourierLayer(nn.Module):
    """Fourier-neural-operator layer: complex trainable weights on the low modes."""

    def __init__(self, ch=4, modes=3):
        super().__init__()
        scale = 1.0 / ch
        self.w = nn.Parameter(scale * torch.randn(ch, ch, modes, modes, dtype=torch.complex64))
        self.skip = nn.Conv2d(ch, ch, 1)
        self.modes = modes

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(b, c, h, w // 2 + 1, dtype=xf.dtype, device=x.device)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w)
        y = torch.fft.irfft2(out, s=(h, w), norm="ortho")
        return F.gelu(y + self.skip(x))


class DetectorHead(nn.Module):
    """Small backbone + ROI Align + classification and box-regression heads."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU())
        self.fc = nn.Linear(8 * 3 * 3, 16)
        self.cls = nn.Linear(16, 3)
        self.box = nn.Linear(16, 4)

    def forward(self, img, rois):
        from torchvision.ops import roi_align
        fm = self.backbone(img)
        pooled = roi_align(fm, rois, output_size=(3, 3), spatial_scale=0.5, sampling_ratio=2, aligned=True)
        h = F.relu(self.fc(pooled.flatten(1)))
        return self.cls(h), self.box(h)


class FakeQuantMLP(nn.Module):
    """Quantization-aware training: fake-quantize activations (STE) with EMA min/max observers."""

    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(8, 16), nn.Linear(16, 4)
        self.register_buffer("lo", torch.tensor(-1.0))
        self.register_buffer("hi", torch.tensor(1.0))

    def fq(self, x):
        if self.training:
            with torch.no_grad():
                self.lo.mul_(0.9).add_(0.1 * x.min())
                self.hi.mul_(0.9).add_(0.1 * x.max())
        scale = ((self.hi - self.lo) / 255.0).clamp_min(1e-6)
        zp = (-128 - torch.round(self.lo / scale)).clamp(-128, 127).to(torch.int32)
        return torch.fake_quantize_per_tensor_affine(x, float(scale), int(zp), -128, 127)

    def forward(self, x):
        return self.l2(self.fq(F.relu(self.l1(self.fq(x)))))


# ---------------------------------------------------------------- workloads

def _w(name, family, dtypes, build, batch, loss, steps=6, lr=1e-2, tol=None):
    return dict(name=name, family=family, dtypes=dtypes, build=build, batch=batch, loss=loss,
                steps=steps, lr=lr, tol=tol)


def workloads():
    def seq_batch(g):
        x = torch.randn(4, 12, 8, generator=g) * 0.5
        return (x,), torch.roll(x, 1, 1) * 0.5

    def moe_batch(g):
        x = torch.randn(32, 8, generator=g)
        return (x,), x.flip(1) * 0.5

    def fno_batch(g):
        x = torch.randn(2, 4, 8, 8, generator=g)
        return (x,), torch.roll(x, 1, 3)

    def det_batch(g):
        img = torch.randn(2, 3, 16, 16, generator=g)
        rois = torch.tensor([[0, 1.0, 1.0, 9.0, 9.0], [0, 4.0, 3.0, 14.0, 12.0], [1, 0.0, 0.0, 8.0, 15.0]])
        return (img, rois), (torch.tensor([0, 2, 1]), torch.randn(3, 4, generator=g))

    def qat_batch(g):
        x = torch.randn(16, 8, generator=g)
        return (x,), x[:, :4] * 2

    def mse(out, tgt):
        return F.mse_loss(out, tgt)

    def moe_loss(out, tgt, model=None):
        return F.mse_loss(out, tgt)

    def det_loss(out, tgt):
        logits, box = out
        return F.cross_entropy(logits, tgt[0]) + F.smooth_l1_loss(box, tgt[1])

    return [
        _w("ssm_selective_scan", "State-space / Mamba-like models", ("fp32", "bf16"),
           SelectiveSSM, seq_batch, mse),
        _w("moe_top2_routing", "Mixture-of-experts / routed networks", ("fp32", "bf16"),
           TopKMoE, moe_batch, moe_loss),
        _w("fourier_operator_complex", "Learned spectral / Fourier neural operators", ("fp32",),
           FourierLayer, fno_batch, mse, lr=5e-3),
        _w("detector_roi_head", "Detection / ROI heads", ("fp32",),
           DetectorHead, det_batch, det_loss),
        _w("qat_fake_quant", "Quantization-aware training (fake quantization)", ("fp32", "bf16"),
           FakeQuantMLP, qat_batch, mse),
    ]


# ---------------------------------------------------------------- runner

def _to(obj, device, dt):
    if isinstance(obj, tuple):
        return tuple(_to(o, device, dt) for o in obj)
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            obj = obj.to(dt)
        return obj.to(device)
    return obj


def train(w, dtype, device, base, data):
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model = copy.deepcopy(base)
    if dt != torch.float32:  # Module.to(float32) would also turn complex parameters into real ones
        model = model.to(dt)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=w["lr"])
    losses = []
    for step in range(w["steps"]):
        args, tgt = data[step]
        out = model(*_to(args, device, dt))
        loss = w["loss"](out, _to(tgt, device, torch.float32 if dtype == "fp32" else dt))
        if hasattr(model, "aux"):
            loss = loss + 0.01 * model.aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    params = {n: p.detach().cpu() for n, p in model.named_parameters()}
    bufs = {n: b.detach().cpu() for n, b in model.named_buffers()}
    return losses, params, bufs


def rel(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def run_one(w, dtype):
    g = torch.Generator().manual_seed(11)
    torch.manual_seed(5)
    base = w["build"]()
    data = [w["batch"](g) for _ in range(w["steps"])]
    res, err = {}, {}
    for device in ("cpu", "xdna"):
        stage = "train"
        try:
            res[device] = train(w, dtype, device, base, data)
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            err[device] = {"stage": stage, "type": type(e).__name__, "message": str(e)[:1300]}
    if "cpu" in err:
        return {"status": "reference_unsupported", "errors": err}
    if "xdna" in err:
        return {"status": "xdna_error", "errors": err}
    (lc, pc, bc), (lx, px, bx) = res["cpu"], res["xdna"]
    loss_tol, par_tol = (0.002, 0.002) if dtype == "fp32" else (0.06, 0.1)
    loss_gap = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(lx, lc))
    par_gap = {n: rel(px[n], pc[n]) for n in pc}
    buf_gap = {n: rel(bx[n], bc[n]) for n in bc}
    learned = lc[-1] < lc[0]
    bad = {}
    if loss_gap > loss_tol:
        bad["loss_trajectory"] = loss_gap
    bad.update({f"param/{n}": v for n, v in par_gap.items() if v > par_tol})
    bad.update({f"buffer/{n}": v for n, v in buf_gap.items() if v > par_tol})
    return {"status": "numeric_mismatch" if bad else "pass",
            "gate": {"loss_rel": loss_tol, "param_rel": par_tol},
            "metrics": {"loss_gap": loss_gap, "worst_param": max(par_gap.values()),
                        "cpu_loss": [lc[0], lc[-1]], "xdna_loss": [lx[0], lx[-1]], "cpu_loss_decreased": learned},
            "failed": bad}


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
    for w in workloads():
        if a.case and w["name"] != a.case:
            continue
        for dtype in w["dtypes"]:
            xdna_train.reset_mapped_cpu_stats()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                try:
                    r = run_one(w, dtype)
                except Exception as e:  # noqa: BLE001
                    r = {"status": "harness_error", "error": str(e), "traceback": traceback.format_exc()}
            r.update(case=w["name"], dtype=dtype, family=w["family"],
                     warnings=list(dict.fromkeys(str(x.message) for x in ws)),
                     npu_runs={k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()
                               if k.startswith("npu") and k.endswith(".run")})
            results.append(r)
            Path(a.out).write_text(json.dumps({"torch": torch.__version__, "backend": xdna_train.__file__,
                                               "scope": "small whole-model training workloads",
                                               "results": results}, indent=2) + "\n")
            print(w["name"], dtype, r["status"], str(r.get("errors", r.get("failed", {})))[:300], flush=True)


if __name__ == "__main__":
    main()
