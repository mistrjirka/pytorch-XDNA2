#!/usr/bin/env python3
"""Tensor semantics: XDNA must behave like CPU for autograd version checks in CPU
islands (LSTM/GRU), resize_/set_ metadata, out= resizing and out= overlap.

Run with the NIS venv python and PYTHONPATH=<repo>.
"""
import warnings

import torch
import xdna_train

xdna_train.register_xdna_device()
failures = []


def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        failures.append(name)


def run(fn, dev):
    """(None, result) or (exception class name, message) of fn(dev)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return None, fn(dev)
    except (RuntimeError, ValueError) as e:
        return type(e).__name__, str(e)


def same_error(name, fn, needle=None):
    """fn(device) must raise on CPU and on XDNA (with `needle` in the message)."""
    c, x = run(fn, "cpu"), run(fn, "xdna")
    ok = c[0] is not None and x[0] is not None
    if ok and needle:
        ok = needle in c[1] and needle in x[1]
    print(f"{name}: cpu {c[0] or 'accepts'}, xdna {x[0] or 'accepts'} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def same_result(name, fn):
    """fn(device) -> tuple must agree on CPU and XDNA (tensors by value), errors by type."""
    c, x = run(fn, "cpu"), run(fn, "xdna")
    if c[0] or x[0]:
        ok = c[0] == x[0]
    else:
        ok = len(c[1]) == len(x[1]) and all(
            torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
            for a, b in zip(c[1], x[1]))
    print(f"{name}: {'OK' if ok else 'FAIL'}" + ("" if ok else f"  cpu={c[1]}  xdna={x[1]}"))
    if not ok:
        failures.append(name)


# --- CPU-island autograd: in-place mutation between forward and backward ---
VERSION = "modified by an inplace operation"


def rnn_mutation(cls, what):
    def fn(dev):
        torch.manual_seed(13)
        model = cls(4, 5).to(dev)
        x = torch.randn(3, 2, 4, device=dev, requires_grad=True)
        y = model(x)[0]
        with torch.no_grad():
            if what == "input":
                x.add_(1)
            else:
                model.weight_ih_l0.add_(1)
        y.sum().backward()
    return fn


for cls in (torch.nn.LSTM, torch.nn.GRU):
    for what in ("input", "weight"):
        same_error(f"{cls.__name__} backward after in-place {what} mutation",
                   rnn_mutation(cls, what), VERSION)


def rnn_grads(cls):
    out = {}
    for dev in ("cpu", "xdna"):
        torch.manual_seed(13)
        model = cls(4, 5).to(dev)
        x = torch.randn(3, 2, 4, device=dev, requires_grad=True)
        model(x)[0].square().sum().backward()
        out[dev] = (x.grad.cpu(), model.weight_ih_l0.grad.cpu())
    return all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*out.values()))


for cls in (torch.nn.LSTM, torch.nn.GRU):
    check(f"{cls.__name__} backward without mutation matches CPU", rnn_grads(cls))


# --- resize_ ---
def resize_same_shape(dev):
    t = torch.arange(12.0, device=dev).view(3, 4).t()
    r = t.resize_(t.shape)
    return r.stride(), r.cpu()


def resize_other(shape):
    def fn(dev):
        t = torch.arange(12.0, device=dev).view(3, 4).t()
        r = t.resize_(*shape)
        return r.shape, r.stride(), r.storage_offset(), r.cpu()
    return fn


def resize_grow(dev):
    t = torch.arange(6.0, device=dev)
    t.resize_(2, 5)
    return t.shape, t.stride(), t.cpu()[0, :5]


same_result("resize_ same shape keeps transposed strides and values", resize_same_shape)
same_result("resize_ transposed to other shape", resize_other((3, 4)))
same_result("resize_ transposed to smaller shape", resize_other((2, 5)))
same_result("resize_ plain tensor grows", resize_grow)


# --- set_ ---
def set_tensor(dev):
    x = torch.arange(6.0, device=dev)
    y = torch.full((6,), 9.0, device=dev)
    x.set_(y)
    x[0] = 42
    return x.cpu(), y.cpu(), x.data_ptr() == y.data_ptr()


def set_tensor_strided(dev):
    x = torch.zeros(3, device=dev)
    y = torch.arange(12.0, device=dev).view(3, 4).t()
    x.set_(y)
    return x.shape, x.stride(), x.cpu(), x.data_ptr() == y.data_ptr()


def set_storage(dev):
    x = torch.zeros(2, device=dev)
    y = torch.arange(10.0, device=dev)
    x.set_(y.untyped_storage(), 2, (2, 3), (3, 1))
    y[2] = -1
    return x.shape, x.stride(), x.storage_offset(), x.cpu()


def set_empty(dev):
    x = torch.arange(6.0, device=dev)
    x.set_()
    return x.shape, x.numel()


same_result("set_(Tensor) shares storage with source", set_tensor)
same_result("set_(Tensor) copies strides and shape", set_tensor_strided)
same_result("set_(Storage, offset, size, stride)", set_storage)
same_result("set_() empties tensor", set_empty)


# --- out= resizing (deprecated on CPU, but same result) ---
def add_out_resize(dev):
    a, b = torch.arange(4.0, device=dev), torch.ones(4, device=dev)
    out = torch.zeros(2, 2, device=dev)
    torch.add(a, b, out=out)
    return out.shape, out.stride(), out.cpu()


def add_out_resize_larger(dev):
    a, b = torch.arange(6.0, device=dev), torch.ones(6, device=dev)
    out = torch.zeros(2, device=dev)
    torch.add(a, b, out=out)
    return out.shape, out.cpu()


same_result("add(out=) resizes a differently shaped nonempty out", add_out_resize)
same_result("add(out=) grows a smaller out", add_out_resize_larger)

# --- out= overlap ---
OVERLAP = "unsupported operation"


def arange8(d):
    return torch.arange(1.0, 9.0, device=d)


same_error("add(x[:-1], 1, out=x[1:]) overlap",
           lambda d: (lambda x: torch.add(x[:-1], 1, out=x[1:]))(arange8(d)), OVERLAP)
same_error("add(a, x[1:], out=x[:-1]) overlap on the second operand",
           lambda d: (lambda x: torch.add(torch.ones(7, device=d), x[1:], out=x[:-1]))(arange8(d)),
           OVERLAP)
same_error("add(out=) internal overlap (expanded out)",
           lambda d: torch.add(torch.ones(4, device=d), 1, out=torch.zeros(1, device=d).expand(4)),
           OVERLAP)
for opname, op in (("sub", torch.sub), ("mul", torch.mul), ("div", torch.div)):
    same_error(f"{opname}(x[:-1], x[:-1], out=x[1:]) overlap",
               lambda d, op=op: (lambda x: op(x[:-1], x[:-1], out=x[1:]))(arange8(d)), OVERLAP)
same_error("sqrt(out=) overlap",
           lambda d: (lambda x: torch.sqrt(x[:-1], out=x[1:]))(arange8(d)), OVERLAP)
same_error("sigmoid(out=) overlap",
           lambda d: (lambda x: torch.sigmoid(x[:-1], out=x[1:]))(arange8(d)), OVERLAP)
same_error("addcmul(out=) overlap",
           lambda d: (lambda x: torch.addcmul(x[:-1], x[:-1], x[:-1], out=x[1:]))(arange8(d)),
           OVERLAP)
same_error("addcdiv(out=) overlap",
           lambda d: (lambda x: torch.addcdiv(x[:-1], x[:-1], x[:-1] + 1, out=x[1:]))(arange8(d)),
           OVERLAP)
same_error("lerp(out=) overlap",
           lambda d: (lambda x: torch.lerp(x[:-1], x[:-1], 0.5, out=x[1:]))(arange8(d)), OVERLAP)


# --- in-place and non-overlapping out= must keep working ---
def inplace_ok(dev):
    x = torch.arange(8.0, device=dev)
    y = torch.ones(8, device=dev)
    x.add_(y)
    x.mul_(2)
    x[2:6].mul_(3)
    x[:4].sub_(x[4:])
    x.div_(2)
    z = torch.zeros(8, device=dev)
    torch.add(x, y, out=z)
    torch.add(x, y, out=x)
    torch.mul(x, x, out=x)
    torch.sqrt(x, out=x)
    torch.sigmoid(z, out=z)
    torch.addcmul(z, x, x, value=0.5, out=z)
    torch.lerp(z, x, 0.25, out=z)
    w = torch.arange(8.0, device=dev)
    torch.add(w[:4], 1, out=w[4:])           # disjoint slices of one storage
    return x.cpu(), z.cpu(), w.cpu()


c, x = run(inplace_ok, "cpu"), run(inplace_ok, "xdna")
check("in-place ops and disjoint out= keep working",
      c[0] is None and x[0] is None and all(torch.allclose(a, b) for a, b in zip(c[1], x[1])))

print("ALL OK" if not failures else f"FAILED: {failures}")
raise SystemExit(1 if failures else 0)
