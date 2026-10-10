#!/usr/bin/env python3
"""Precision gate: does BFP16 NPU conv arithmetic hurt NIS training?

Trains NIS MyNet (+ the script's VGG perceptual loss) on CPU twice from the same
seed and data: plain BF16, and with every eligible 3x3 stride-1 conv replaced
by an exact emulation of the tall-skinny ATB NPU kernel:
  * activations quantized to bfp16ebs8 (blocks of 8 channels, round-to-nearest,
    as the core's conv_even conversion does),
  * weights quantized to bfp16ebs8 on the host (round-to-nearest),
  * the K reduction done in 64-wide chunks with the running C rounded to BF16
    after each chunk (the kernel's bf16 C buffer).
Forward and input gradient (dX) use the emulation; dW stays exact (--dw exact)
or uses single-pass BFP16 with fp32 sums (--dw bfp1).  Prints both loss
curves and their relative gap.

    python benchmarks/bfp16_training_gate.py --steps 300
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

NIS = "/home/jirka/programovani/magistr/nis/lab02_cnn"


def bfp_quant(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Round-to-nearest bfp16ebs8 along `dim` (size divisible by 8)."""
    xf = x.float().movedim(dim, -1)
    shape = xf.shape
    blk = xf.reshape(*shape[:-1], shape[-1] // 8, 8)
    maxabs = blk.abs().amax(-1, keepdim=True)
    exp = torch.floor(torch.log2(maxabs.clamp_min(1e-38)))
    lsb = torch.exp2(exp - 6)
    q = torch.clamp(torch.round(blk / lsb), -128, 127) * lsb
    q = torch.where(maxabs > 0, q, torch.zeros_like(q))
    return q.reshape(shape).movedim(-1, dim)


def bf16(x):
    return x.to(torch.bfloat16).float()


def npu_conv3x3(x, w, chunk=64):
    """Emulated NPU forward: implicit GEMM, K ordered (dy, dx, c), bf16 C per chunk."""
    B, C, H, W = x.shape
    xq = bfp_quant(x, 1)
    wq = bfp_quant(w.permute(0, 2, 3, 1).reshape(w.shape[0], -1), 1)  # [Cout, (dy,dx,c)]
    cols = F.unfold(xq, 3, padding=1)  # [B, C*9, L], K order (c, dy, dx)
    cols = cols.reshape(B, C, 9, H * W).permute(0, 3, 2, 1).reshape(B * H * W, 9 * C)
    acc = torch.zeros(B * H * W, w.shape[0])
    for k0 in range(0, 9 * C, chunk):
        acc = bf16(acc + cols[:, k0:k0 + chunk] @ wq[:, k0:k0 + chunk].t())
    return acc.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


DW_MODE = "exact"


def npu_conv3x3_dw_bfp1(x, gy, wshape):
    """Single-pass BFP16 dW: both operands quantized in blocks of 8 consecutive
    pixels (the reduction dim, as the NPU kernel's [8 ch][8 px] blocks), fp32 sums."""
    B, C, H, W = x.shape
    L = H * W
    pad = (-L) % 8
    cols = F.unfold(x, 3, padding=1)                         # [B, C*9, L], K (c, dy, dx)
    g = gy.reshape(B, gy.shape[1], L)
    if pad:
        cols, g = F.pad(cols, (0, pad)), F.pad(g, (0, pad))
    cq, gq = bfp_quant(cols, 2), bfp_quant(g, 2)
    gw = torch.einsum("bkl,bol->ok", cq, gq)
    return gw.reshape(wshape)


class NpuConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return npu_conv3x3(x.float(), w.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gx = gw = None
        if ctx.needs_input_grad[0]:
            # dX of a stride-1 3x3 conv = conv of dY with flipped, transposed weights.
            wt = w.flip(2, 3).transpose(0, 1)
            gx = npu_conv3x3(gy.float(), wt.float()).to(x.dtype)
        if ctx.needs_input_grad[1]:
            if DW_MODE == "bfp1":
                gw = npu_conv3x3_dw_bfp1(x.float(), gy.float(), w.shape).to(w.dtype)
            else:
                gw = torch.nn.grad.conv2d_weight(x, w.shape, gy, padding=1)
        return gx, gw


def eligible(m):
    return (isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.stride == (1, 1)
            and m.padding == (1, 1) and m.groups == 1 and m.in_channels % 8 == 0
            and m.out_channels % 8 == 0)


def patch(module):
    n = 0
    for m in module.modules():
        if eligible(m):
            def fwd(x, m=m):
                y = NpuConv.apply(x, m.weight)
                return y if m.bias is None else y + m.bias.view(1, -1, 1, 1)
            m.forward = fwd
            n += 1
    return n


def run(emulate: bool, steps: int, seed: int):
    sys.path.insert(0, NIS)
    os.chdir(NIS)
    from models import MyNet
    from train import ImageGenLoss
    from data import DatasetPatches
    torch.manual_seed(seed)
    dt = torch.bfloat16
    model = MyNet(3, 3, 5).to(dt)
    loss_fn = ImageGenLoss(4.0, 6.0).to(dt)
    if emulate:
        print("emulated convs:", patch(model), "+", patch(loss_fn), "(VGG)")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    vid = "VID20261009180405"
    g = torch.Generator().manual_seed(seed)
    data = DatasetPatches(f"data/{vid}/train/input", f"data/{vid}/train/target",
                          patch_size=36, num_patches=steps * 40)
    loader = torch.utils.data.DataLoader(data, batch_size=40, shuffle=True, generator=g)
    losses = []
    for i, (x, y) in enumerate(loader):
        loss = loss_fn(model(x.to(dt)), y.to(dt))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i % 25 == 0:
            print(f"  {'npu-emul' if emulate else 'bf16':8s} step {i:4d} loss {loss.item():.4f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dw", choices=["exact", "bfp1"], default="exact",
                    help="dW arithmetic in the emulated run (exact = CPU)")
    args = ap.parse_args()
    global DW_MODE
    DW_MODE = args.dw
    base = run(False, args.steps, args.seed)
    emul = run(True, args.steps, args.seed)
    w = 25
    print("\nwindow    bf16 mean   npu-emul mean   rel gap")
    for s in range(0, args.steps, w):
        a = sum(base[s:s + w]) / len(base[s:s + w])
        b = sum(emul[s:s + w]) / len(emul[s:s + w])
        print(f"{s:4d}-{s + w:4d}  {a:9.4f}   {b:9.4f}      {100 * (b - a) / a:+6.2f}%")


if __name__ == "__main__":
    main()
