#!/usr/bin/env python3
"""GRU create_graph=True must retain a differentiable first derivative.

Uses identical CPU-created inputs and parameters on CPU and XDNA. Ordinary
forward/first-gradient agreement is a control; missing higher derivatives fail.
Run with the hardware Python and PYTHONPATH=<checkout>.
"""
import copy
import os
import sys

os.environ["XDNA_CPU_ISLANDS"] = "1"
import torch
import xdna_train

xdna_train.register_xdna_device()
torch.manual_seed(42)
cpu_model = torch.nn.GRU(3, 4)
xdna_model = copy.deepcopy(cpu_model).to("xdna")
data = torch.randn(2, 1, 3)
cpu_x = data.clone().requires_grad_()
xdna_x = data.to("xdna").requires_grad_()
failures = []

def check(name, condition, detail=""):
    print(f"{name}: {'OK' if condition else 'FAIL'} {detail}", flush=True)
    if not condition:
        failures.append(name)

cpu_y = cpu_model(cpu_x)[0]
xdna_y = xdna_model(xdna_x)[0]
check("forward matches CPU",
      torch.allclose(xdna_y.detach().cpu(), cpu_y.detach(), rtol=1e-4, atol=1e-5))
cpu_g, = torch.autograd.grad(cpu_y.sum(), cpu_x, create_graph=True)
xdna_g, = torch.autograd.grad(xdna_y.sum(), xdna_x, create_graph=True)
check("first gradient matches CPU",
      torch.allclose(xdna_g.detach().cpu(), cpu_g.detach(), rtol=1e-4, atol=1e-5))
check("CPU first gradient remains differentiable", cpu_g.requires_grad)
cpu_gg, = torch.autograd.grad(cpu_g.square().sum(), cpu_x)
check("CPU second derivative is finite and nonzero",
      bool(torch.isfinite(cpu_gg).all()) and cpu_gg.norm().item() > 0,
      f"norm={cpu_gg.norm().item():.9f}")
check("XDNA first gradient remains differentiable", xdna_g.requires_grad)
try:
    xdna_gg, = torch.autograd.grad(xdna_g.square().sum(), xdna_x)
except RuntimeError as error:
    check("XDNA second derivative exists", False, str(error))
else:
    check("XDNA second derivative matches CPU",
          torch.allclose(xdna_gg.detach().cpu(), cpu_gg.detach(), rtol=1e-4, atol=1e-5))
print("ALL OK" if not failures else f"FAILED: {failures}", flush=True)
sys.exit(1 if failures else 0)
