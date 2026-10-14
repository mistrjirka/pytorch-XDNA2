#!/usr/bin/env python3
"""Complex numbers and lazy conj / neg views on XDNA must match CPU.

Regression for: complex matmul gradients lost their conjugation (the CPU alias of
an XDNA tensor dropped the lazy-conjugate flag), and any op on a conj / neg view
crashed the process.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import sys

import torch
import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


def close(a, b, tol=1e-5):
    return torch.allclose(a.cpu().resolve_conj(), b, atol=tol, rtol=tol)


# One-element analytic case: x = 1+2j, w = 3+4j, upstream gradient 1.
x = torch.tensor([[1 + 2j]], dtype=torch.complex64, device="xdna", requires_grad=True)
w = torch.tensor([[3 + 4j]], dtype=torch.complex64, device="xdna", requires_grad=True)
y = x @ w
y.backward(torch.ones_like(y))
check("complex mm forward", close(y.detach(), torch.tensor([[-5 + 10j]])))
check("complex mm dx = conj(w)", close(x.grad, torch.tensor([[3 - 4j]])))
check("complex mm dw = conj(x)", close(w.grad, torch.tensor([[1 - 2j]])))

# Larger complex Linear: gradients against CPU.
torch.manual_seed(0)
lin_c = torch.nn.Linear(6, 5, dtype=torch.complex64)
lin_x = torch.nn.Linear(6, 5, dtype=torch.complex64)
lin_x.load_state_dict(lin_c.state_dict())
lin_x = lin_x.to("xdna")
inp = torch.randn(4, 6, dtype=torch.complex64)
g = torch.randn(4, 5, dtype=torch.complex64)
ic = inp.clone().requires_grad_()
ix = inp.to("xdna").requires_grad_()
lin_c(ic).backward(g)
lin_x(ix).backward(g.to("xdna"))
check("complex Linear input grad", close(ix.grad, ic.grad, 1e-4))
check("complex Linear weight grad", close(lin_x.weight.grad, lin_c.weight.grad, 1e-4))
check("complex Linear bias grad", close(lin_x.bias.grad, lin_c.bias.grad, 1e-4))

# Gradient through z * conj(z): d|z|^2 = 2z.
z = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
p = z.to("xdna").requires_grad_()
(p * p.conj()).real.sum().backward()
check("grad of |z|^2", close(p.grad, 2 * z))

# Lazy views must not crash and must read correctly.
zx = z.to("xdna")
check("conj readback", close(zx.conj().cpu(), z.conj()))
check("conj().resolve_conj()", close(zx.conj().resolve_conj(), z.conj()))
check("conj().clone()", close(zx.conj().clone(), z.conj()))
dst = torch.empty(2, dtype=torch.complex64, device="xdna")
dst.copy_(zx.conj())
check("copy_ from a conj view", close(dst, z.conj()))
check("conj(conj(z)) is z", close(zx.conj().conj(), z))
check("hermitian product",
      close(zx.conj().reshape(1, 2) @ zx.reshape(2, 1), z.conj().reshape(1, 2) @ z.reshape(2, 1)))

r = torch.tensor([1.0, -2.0])
rx = r.to("xdna")
check("neg view readback", torch.equal(torch._neg_view(rx).cpu().resolve_neg(), -r))
check("neg view + 1", torch.equal((torch._neg_view(rx) + 1).cpu(), -r + 1))
a = torch.randn(4, 4).bfloat16()
b = torch.randn(4, 4).bfloat16()
check("neg view in mm", torch.allclose((torch._neg_view(a.to("xdna")) @ b.to("xdna")).cpu().float(),
                                       -(a @ b).float(), atol=0.1))

# view_as_real / view_as_complex are true views: in-place updates reach the base
# (Adam on complex parameters updates through view_as_real and silently lost them).
zc = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex64)
zv = zc.to("xdna")
rv = torch.view_as_real(zv)
check("view_as_real values", torch.equal(rv.cpu(), torch.view_as_real(zc)))
rv[0, 1] = 100.0
check("write through view_as_real reaches the complex base", zv.cpu()[0] == 1 + 100j)
check("view_as_complex round trip", torch.equal(torch.view_as_complex(torch.view_as_real(zv)).cpu(), zv.cpu()))
for name, mk in (("SGD", lambda q: torch.optim.SGD(q, lr=0.1)), ("Adam", lambda q: torch.optim.Adam(q, lr=5e-3)),
                 ("AdamW", lambda q: torch.optim.AdamW(q, lr=5e-3)), ("RMSprop", lambda q: torch.optim.RMSprop(q, lr=5e-3))):
    torch.manual_seed(1)
    w0 = torch.randn(3, 3, dtype=torch.complex64)
    gr = torch.randn(3, 3, dtype=torch.complex64)
    got = {}
    for dev in ("cpu", "xdna"):
        q = torch.nn.Parameter(w0.clone().to(dev))
        q.grad = gr.clone().to(dev)
        o = mk([q])
        for _ in range(3):
            o.step()
        got[dev] = q.detach().cpu()
    check(f"{name} on a complex parameter matches CPU", close(got["xdna"], got["cpu"], 1e-5))

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
