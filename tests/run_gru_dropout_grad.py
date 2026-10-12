#!/usr/bin/env python3
"""create_graph must preserve gradients and RNG of the same GRU/LSTM forward."""
import os
import sys
os.environ["XDNA_CPU_ISLANDS"] = "1"
import torch
import xdna_train
xdna_train.register_xdna_device()
failures = []
def check(name, good, detail=""):
    print(f"{name}: {'OK' if good else 'FAIL'} {detail}", flush=True)
    if not good:
        failures.append(name)
for rnn, device, dropout in [(r, d, p) for r in (torch.nn.GRU, torch.nn.LSTM)
                             for d in ("cpu", "xdna") for p in (0., .5)]:
    torch.manual_seed(42)
    model = rnn(3,4,num_layers=2,dropout=dropout).to(device)
    data = torch.randn(5,2,3)
    x = data.to(device).requires_grad_()
    y = model(x)[0]
    ordinary, = torch.autograd.grad(y.sum(),x,retain_graph=True)
    before = torch.get_rng_state().clone()
    differentiable, = torch.autograd.grad(y.sum(),x,create_graph=True)
    rng_same = torch.equal(before,torch.get_rng_state())
    a,b = ordinary.detach().cpu(),differentiable.detach().cpu()
    error = (a-b).abs().max().item()
    relative = ((a-b).norm()/a.norm()).item()
    label = f"{rnn.__name__} {device} dropout={dropout}"
    check(label + " first gradient unchanged", torch.allclose(a,b,rtol=1e-4,atol=1e-5),
          f"max_error={error:.9f} relative_error={relative:.6%}")
    check(label + " differentiable gradient", differentiable.requires_grad)
    check(label + " backward preserves RNG", rng_same)
print("ALL OK" if not failures else f"FAILED: {failures}", flush=True)
sys.exit(1 if failures else 0)
