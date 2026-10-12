#!/usr/bin/env python3
"""What `import xdna_train` changes stays scoped to XDNA:
Parameter identity is kept for moves to XDNA only, and the step autotuner
ignores CPU optimizers.  Run with PYTHONPATH=<repo>."""
import sys

import torch
import xdna_train
import xdna_train.device as dev

failures = []


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


check("global swap flag untouched", not torch.__future__.get_swap_module_params_on_conversion())

# A CPU optimizer stepping first must not become the autotuner's subject.
c = torch.nn.Linear(4, 4)
co = torch.optim.SGD(c.parameters(), lr=0.1)
for _ in range(5):
    c(torch.randn(2, 4)).sum().backward()
    co.step()
check("autotuner ignores CPU optimizer",
      dev._AUTOTUNER is None or dev._AUTOTUNER.optimizer_id is None)

# Optimizer built before the move still trains the (same) parameters.
m = torch.nn.Linear(4, 4).bfloat16()
w = m.weight
opt = torch.optim.Adam(m.parameters(), lr=0.1)
m.to("xdna")
before = m.weight.detach().cpu().clone()
m(torch.randn(2, 4).bfloat16().to("xdna")).sum().backward()
opt.step()
check("to('xdna') keeps Parameter", m.weight is w and m.weight.device.type == "xdna")
check("optimizer built before the move updates it", not torch.equal(before, m.weight.detach().cpu()))
m2 = torch.nn.Linear(4, 4)
w2 = m2.weight
m2.xdna()
check(".xdna() keeps Parameter", m2.weight is w2)
m3 = torch.nn.Linear(4, 4)
w3 = m3.weight
m3.to(torch.device("xdna"), torch.bfloat16)
check("to(device, dtype) keeps Parameter", m3.weight is w3 and w3.dtype == torch.bfloat16)
check("swap flag restored after moves", not torch.__future__.get_swap_module_params_on_conversion())

# CPU modules keep PyTorch's default (no swap): LSTM.load_state_dict works.
try:
    torch.nn.LSTM(4, 8).load_state_dict(torch.nn.LSTM(4, 8).state_dict())
    torch.nn.LSTM(4, 8).to("xdna").load_state_dict(torch.nn.LSTM(4, 8).state_dict())
    check("LSTM load_state_dict (cpu and xdna)", True)
except RuntimeError as e:
    check(f"LSTM load_state_dict ({e})", False)

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
