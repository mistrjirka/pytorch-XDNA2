"""Standard torchvision architectures: one training step on XDNA vs CPU.

The point is breadth: which well-known architectures train on XDNA at all, and which
operator is missing when one does not.  Each model is built with random weights and
small inputs, dropout / stochastic depth are switched off (they are random), and one
forward + backward pass in train mode is compared with CPU on identical weights:
output, input gradient and every parameter gradient.

    python tests/coverage/torchvision_models.py --out /tmp/tv.json [--case NAME] [--dtype fp32|bf16]

Output uses the coverage-gate schema (case, dtype, status, errors, failed, metrics, npu_runs).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")

import torch
from torch import nn


def models():
    import torchvision.models as tvm
    import torchvision.models.segmentation as seg
    import torchvision.models.video as vid

    n = dict(num_classes=10, weights=None)
    img = (2, 3, 64, 64)
    m = {
        "resnet18": (lambda: tvm.resnet18(**n), img),
        "resnet50": (lambda: tvm.resnet50(**n), img),
        "wide_resnet50_2": (lambda: tvm.wide_resnet50_2(**n), img),
        "resnext50_32x4d": (lambda: tvm.resnext50_32x4d(**n), img),
        "vgg11_bn": (lambda: tvm.vgg11_bn(**n), (2, 3, 64, 64)),
        "densenet121": (lambda: tvm.densenet121(**n), img),
        "squeezenet1_1": (lambda: tvm.squeezenet1_1(**n), img),
        "shufflenet_v2_x1_0": (lambda: tvm.shufflenet_v2_x1_0(**n), img),
        "mobilenet_v2": (lambda: tvm.mobilenet_v2(**n), img),
        "mobilenet_v3_small": (lambda: tvm.mobilenet_v3_small(**n), img),
        "mnasnet0_5": (lambda: tvm.mnasnet0_5(**n), img),
        "efficientnet_b0": (lambda: tvm.efficientnet_b0(**n), img),
        "regnet_y_400mf": (lambda: tvm.regnet_y_400mf(**n), img),
        "convnext_tiny": (lambda: tvm.convnext_tiny(**n), img),
        "googlenet": (lambda: tvm.googlenet(aux_logits=False, init_weights=True, **n), (2, 3, 96, 96)),
        "vit_b_16_small": (lambda: tvm.VisionTransformer(image_size=64, patch_size=16, num_layers=2, num_heads=4,
                                                         hidden_dim=64, mlp_dim=128, num_classes=10), img),
        "swin_t": (lambda: tvm.swin_t(**n), (2, 3, 128, 128)),
        "maxvit_t": (lambda: tvm.maxvit_t(**n), (2, 3, 224, 224)),
        "deeplabv3_mobilenet": (lambda: seg.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=5, aux_loss=False), (2, 3, 64, 64)),
        "fcn_resnet50": (lambda: seg.fcn_resnet50(weights=None, weights_backbone=None, num_classes=5, aux_loss=False), (2, 3, 64, 64)),
        "lraspp_mobilenet": (lambda: seg.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=5), (2, 3, 64, 64)),
        "r3d_18_video": (lambda: vid.r3d_18(num_classes=10, weights=None), (2, 3, 8, 32, 32)),
        "mc3_18_video": (lambda: vid.mc3_18(num_classes=10, weights=None), (2, 3, 8, 32, 32)),
        "r2plus1d_18_video": (lambda: vid.r2plus1d_18(num_classes=10, weights=None), (2, 3, 8, 32, 32)),
    }
    return m


def tame(model):
    """Remove randomness: dropout and stochastic depth off."""
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.p = 0.0
        if type(mod).__name__ == "StochasticDepth":
            mod.p = 0.0
    return model


def first_tensor(out):
    if isinstance(out, dict):
        out = out["out"] if "out" in out else next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    return out


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def run_one(name, build, shape, dtype):
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(3)
    base = tame(build())
    x0 = torch.randn(*shape)
    res, err = {}, {}
    passes = [("cpu", "cpu", dt), ("xdna", "xdna", dt)]
    if dt != torch.float32:
        passes.append(("truth", "cpu", torch.float32))  # FP32 CPU: what BF16 results are judged against
    for key, device, pdt in passes:
        stage = "construct"
        try:
            m = copy.deepcopy(base).to(pdt).to(device).train()
            x = x0.detach().clone().to(pdt).to(device).requires_grad_()
            stage = "forward"
            y = first_tensor(m(x))
            g = torch.linspace(-0.7, 0.9, y.numel()).reshape(y.shape).to(y.dtype).to(device)
            stage = "backward"
            y.backward(g)
            stage = "readback"
            res[key] = dict(out=y.detach().cpu(), dx=x.grad.detach().cpu(),
                               grads={n: p.grad.detach().cpu() for n, p in m.named_parameters() if p.grad is not None},
                               missing=[n for n, p in m.named_parameters() if p.grad is None])
        except Exception as e:  # noqa: BLE001
            err[key] = {"stage": stage, "type": type(e).__name__, "message": str(e)[:900]}
    if "cpu" in err:
        return {"status": "reference_unsupported", "errors": err}
    if "xdna" in err:
        return {"status": "xdna_error", "errors": err}
    a, b = res["xdna"], res["cpu"]
    tol_out, tol_grad = (2e-3, 5e-3) if dtype == "fp32" else (0.06, 0.2)

    def noise_aware(x, ref, top):
        return float((x.double() - ref.double()).norm() / max(float(ref.double().norm()), 1e-3 * top))

    names = [n for n in b["grads"] if n in a["grads"]]
    if dtype == "fp32":
        top = max(float(b["grads"][n].double().norm()) for n in names)
        gaps = {n: noise_aware(a["grads"][n], b["grads"][n], top) for n in names}
        m_out, m_dx = rel(a["out"], b["out"]), rel(a["dx"], b["dx"])
        limit = lambda n: tol_grad
    else:
        # BF16: judge both sides against the FP32 CPU truth.  XDNA must be no worse than
        # CPU BF16 by more than a factor 2 (plus a small slack); gradients that are ~0 in
        # exact arithmetic are measured relative to the largest gradient.
        t = res["truth"]
        top = max(float(t["grads"][n].double().norm()) for n in names)
        gaps = {n: noise_aware(a["grads"][n], t["grads"][n], top) for n in names}
        cpu_gaps = {n: noise_aware(b["grads"][n], t["grads"][n], top) for n in names}
        m_out, m_dx = rel(a["out"], t["out"]), rel(a["dx"], t["dx"])
        base_out, base_dx = rel(b["out"], t["out"]), rel(b["dx"], t["dx"])
        tol_out, tol_dx = 2 * base_out + 0.02, 2 * base_dx + 0.05
        limit = lambda n: 2 * cpu_gaps[n] + 0.05
    bad = {}
    if m_out > tol_out:
        bad["output"] = m_out
    if m_dx > (tol_grad if dtype == "fp32" else tol_dx):
        bad["input_grad"] = m_dx
    bad.update({f"grad/{n}": v for n, v in gaps.items() if v > limit(n)})
    if a["missing"] != b["missing"]:
        bad["missing_grads"] = list(set(a["missing"]) ^ set(b["missing"]))[:5]
    return {"status": "numeric_mismatch" if bad else "pass",
            "gate": {"output": tol_out, "grads": tol_grad if dtype == "fp32" else "2x CPU-bf16 error vs fp32 + 0.05"},
            "metrics": {"output": m_out, "input_grad": m_dx, "worst_param_grad": max(gaps.values(), default=0.0),
                        "params": len(b["grads"])},
            "failed": dict(list(bad.items())[:8])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "both"], default="both")
    a = ap.parse_args()
    torch.set_num_threads(8)
    import xdna_train
    xdna_train.register_xdna_device()
    xdna_train.set_npu_mode(2)
    results = []
    for name, (build, shape) in models().items():
        if a.case and a.case != name:
            continue
        for dtype in (["fp32", "bf16"] if a.dtype == "both" else [a.dtype]):
            xdna_train.reset_mapped_cpu_stats()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                try:
                    r = run_one(name, build, shape, dtype)
                except Exception as e:  # noqa: BLE001
                    r = {"status": "harness_error", "error": str(e), "traceback": traceback.format_exc()}
            r.update(case=name, dtype=dtype, family="torchvision architecture",
                     warnings=list(dict.fromkeys(str(w.message) for w in ws))[:5],
                     npu_runs={k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()
                               if k.startswith("npu") and k.endswith(".run")})
            results.append(r)
            Path(a.out).write_text(json.dumps({"torch": torch.__version__, "scope": "one train step, random weights, small inputs",
                                               "results": results}, indent=2, default=str) + "\n")
            detail = r.get("errors") or r.get("failed") or r.get("error") or ""
            print(f"{name} {dtype}: {r['status']}  {str(detail)[:260]}", flush=True)


if __name__ == "__main__":
    main()
