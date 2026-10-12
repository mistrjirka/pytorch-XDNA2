#!/usr/bin/env python3
"""How much of a model's conv work can the conv3x3w NPU family take, and how many
instruction streams (7 s IRON builds) would it need?  No hardware needed: convs are
traced on the meta device and classified with a copy of the planner's rules
(XdnaConv3x3wRuntime::plan / cost_model_prefers_npu in torch_xdna_device.cpp).

usage: conv_coverage.py [--batch 32] [--tile]   (--tile: also model width tiling)
"""
import argparse
import math
from collections import Counter, defaultdict

import torch
import torchvision.models as tvm

MAX_W, MIN_MACS = 36, 20e6
GROUP_ROWS_OUT = 2048


def cost_prefers_npu(B, cin, cout, h, w, real_macs):
    wp = w + 2
    padded = B * (h + 2) * wp * (math.ceil(cin / 64) * 64) * (math.ceil(cout / 128) * 128) * 9
    io = 2.0 * B * (cin + cout) * h * w
    return 0.5 + padded / 5.5e9 + io / 20e6 < real_macs / 0.9e9


def classify(m, x, tile):
    """-> (reason or 'npu', real_macs, stream key)"""
    B, cin, h, w = x.shape
    cout = m.out_channels
    k, s, p, d, g = m.kernel_size, m.stride, m.padding, m.dilation, m.groups
    macs = B * (h // s[0]) * (w // s[1]) * cin // g * cout * k[0] * k[1]
    if g != 1:
        return ("groups/depthwise", macs, None)
    if k != (3, 3):
        return (f"kernel {k[0]}x{k[1]}", macs, None)
    if s != (1, 1) or p != (1, 1) or d != (1, 1):
        return ("stride/pad/dilation", macs, None)
    nb, ww = B, w
    if w > MAX_W:
        if not tile:
            return ("width>36", macs, None)
        # vertical strips of 34 valid output columns (36 input columns incl. halo)
        strips = math.ceil(w / 34)
        nb, ww = B * strips, MAX_W
    real = nb * h * ww * cin * cout * 9 if w <= MAX_W else macs
    if macs < MIN_MACS:
        return ("too small (<20M MAC)", macs, None)
    if not cost_prefers_npu(nb, cin, cout, h, ww, macs):
        return ("cost model (padding/IO)", macs, None)
    groups = math.ceil(nb * (h + 2) * (ww + 2) / GROUP_ROWS_OUT)
    return ("npu", macs, (groups, ww + 2, math.ceil(cin / 64)))


def trace(model, x):
    rows = []
    hooks = [m.register_forward_hook(lambda m, i, o: rows.append((m, i[0].shape)))
             for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return rows


def models():
    meta = dict(device="meta")
    yield "resnet18 224", tvm.resnet18().to(**meta), (3, 224, 224)
    yield "resnet18 32 (cifar)", tvm.resnet18().to(**meta), (3, 32, 32)
    yield "resnet50 224", tvm.resnet50().to(**meta), (3, 224, 224)
    yield "vgg11 224", tvm.vgg11().to(**meta), (3, 224, 224)
    yield "vgg11 64", tvm.vgg11().to(**meta), (3, 64, 64)
    yield "mobilenet_v3_small 224", tvm.mobilenet_v3_small().to(**meta), (3, 224, 224)
    yield "efficientnet_b0 224", tvm.efficientnet_b0().to(**meta), (3, 224, 224)
    yield "densenet121 64", tvm.densenet121().to(**meta), (3, 64, 64)
    yield "regnet_y_400mf 224", tvm.regnet_y_400mf().to(**meta), (3, 224, 224)
    yield "shufflenet_v2 224", tvm.shufflenet_v2_x1_0().to(**meta), (3, 224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tile", action="store_true")
    a = ap.parse_args()
    all_keys = Counter()
    print(f"batch {a.batch}, tiling {'on' if a.tile else 'off'}")
    print(f"{'model':26s} {'conv GMAC':>9s} {'on NPU':>7s}  top reasons for CPU")
    for name, model, shape in models():
        try:
            rows = trace(model, torch.empty(a.batch, *shape, device="meta"))
        except Exception as e:  # noqa: BLE001
            print(f"{name:26s} trace failed: {e}")
            continue
        tot, npu = 0.0, 0.0
        why = defaultdict(float)
        keys = set()
        for m, shp in rows:
            reason, macs, key = classify(m, torch.empty(shp, device="meta"), a.tile)
            tot += macs
            if reason == "npu":
                npu += macs
                keys.add(key)
                all_keys[key] += 1
            else:
                why[reason] += macs
        top = ", ".join(f"{r} {100 * v / tot:.0f}%" for r, v in sorted(why.items(), key=lambda t: -t[1])[:3])
        print(f"{name:26s} {tot / 1e9:9.1f} {100 * npu / tot:6.0f}%  {top}  [{len(keys)} streams]")
    print(f"\ndistinct (groups, wp, chunks) streams over all models: {len(all_keys)}")


if __name__ == "__main__":
    main()
