#!/usr/bin/env python3
"""CPU islands (xdna_train/cpu_islands.py): LSTM / GRU on XDNA match CPU, incl. gradients.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import copy
import sys

import torch
import xdna_train

xdna_train.register_xdna_device()


def check(name, make, x_shape, hx=False):
    torch.manual_seed(0)
    ref = make().bfloat16()
    dev = copy.deepcopy(ref).to("xdna")
    x = torch.randn(*x_shape).bfloat16()
    xr = x.clone().requires_grad_()
    xd = x.to("xdna").requires_grad_()
    yr, _ = ref(xr)
    yd, _ = dev(xd)
    g = torch.randn_like(yr)
    yr.backward(g)
    yd.backward(g.to("xdna"))
    errs = [(yd.cpu().float() - yr.float()).abs().max().item(),
            (xd.grad.cpu().float() - xr.grad.float()).abs().max().item()]
    for (n, pr), pd in zip(ref.named_parameters(), dev.parameters()):
        errs.append((pd.grad.cpu().float() - pr.grad.float()).abs().max().item())
    worst = max(errs)
    ok = worst <= 1e-2 * max(1.0, yr.float().abs().max().item())
    print(f"{name}: max abs diff {worst:.3g} ({len(errs)} tensors) {'OK' if ok else 'FAIL'}")
    return ok


ok = True
ok &= check("lstm 2-layer batch_first", lambda: torch.nn.LSTM(16, 32, num_layers=2, batch_first=True), (4, 7, 16))
ok &= check("lstm bidirectional", lambda: torch.nn.LSTM(16, 24, bidirectional=True), (5, 3, 16))
ok &= check("gru 2-layer", lambda: torch.nn.GRU(16, 32, num_layers=2, batch_first=True), (4, 7, 16))
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
