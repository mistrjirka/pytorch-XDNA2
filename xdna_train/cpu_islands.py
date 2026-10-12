"""CPU islands: whole ops that PyTorch accelerates only for CPU tensors.

Some CPU kernels are picked by device type, not by the op: `aten::lstm` on a
CPU bf16 tensor runs as one fused oneDNN RNN per layer (mkldnn_rnn_layer), but
on any other device it decomposes into thousands of per-timestep matmuls and
pointwise ops (model zoo LSTM: 48 ms vs 22 ms on CPU).  XDNA storage is host
memory, so such an op can run on CPU views of its XDNA arguments: the op and
its backward form a small CPU autograd graph, wrapped in one autograd node on
the XDNA side.  Results are adopted (or copied) into XDNA tensors.
"""
from __future__ import annotations

import torch

_LIB = None


def _native():
    from . import device
    return device._C


class _CpuIsland(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *tensors):
        C = _native()
        with torch.enable_grad():
            cpu_in = []
            for t in tensors:
                c = C.cpu_alias(t).detach()
                if t.requires_grad and c.is_floating_point():
                    c.requires_grad_()
                cpu_in.append(c)
            outs = fn(*cpu_in)
        # The CPU views share bytes but not version counters; saving the
        # XDNA originals makes backward fail on in-place changes like CPU does.
        ctx.save_for_backward(*tensors)
        # The CPU graph reads the XDNA bytes through non-owning aliases, so
        # the originals must outlive it even if a saved-tensor hook
        # (save_on_cpu) replaces the saved copies and lets their storage go.
        ctx.keep = tensors
        ctx.fn = fn
        ctx.cpu_in, ctx.cpu_out = cpu_in, outs
        return tuple(C.from_cpu(o.detach()) for o in outs)

    @staticmethod
    def backward(ctx, *grads):
        C = _native()
        # Reading saved_tensors raises if an input was modified in place.
        saved = ctx.saved_tensors
        if torch.is_grad_enabled():
            return _differentiable_backward(ctx.fn, saved, grads)
        outs, gs = [], []
        for o, g in zip(ctx.cpu_out, grads):
            if g is not None and o.requires_grad:
                outs.append(o)
                gs.append(C.cpu_alias(g))
        wanted = [c for c in ctx.cpu_in if c.requires_grad]
        res = iter(torch.autograd.grad(outs, wanted, gs, allow_unused=True,
                                       retain_graph=True) if outs and wanted else ())
        out = [None]  # fn
        for c in ctx.cpu_in:
            r = next(res) if c.requires_grad else None
            out.append(None if r is None else C.from_cpu(r))
        return tuple(out)


def _differentiable_backward(fn, inputs, grads):
    """backward with create_graph=True: the CPU graph above is detached from
    the XDNA tensors, so redo the op on CPU copies made with differentiable
    .to() calls, then the gradients (and their graph) lead back to XDNA."""
    cpu_in = [t.to("cpu") for t in inputs]
    outs = fn(*cpu_in)
    sel = [(o, g.to("cpu")) for o, g in zip(outs, grads)
           if g is not None and o.requires_grad]
    wanted = [i for i, (t, c) in enumerate(zip(inputs, cpu_in)) if t.requires_grad]
    res = torch.autograd.grad([o for o, _ in sel], [cpu_in[i] for i in wanted],
                              [g for _, g in sel], allow_unused=True,
                              create_graph=True) if sel and wanted else ()
    out = [None] * (len(inputs) + 1)  # [fn, *inputs]
    for i, r in zip(wanted, res):
        out[i + 1] = None if r is None else r.to(inputs[i].device)
    return tuple(out)


def _lstm_input(input, hx, params, has_biases, num_layers, dropout, train,
                bidirectional, batch_first):
    n_hx = len(hx)

    def fn(*cpu):
        return torch.lstm(cpu[0], list(cpu[1:1 + n_hx]), list(cpu[1 + n_hx:]),
                          has_biases, num_layers, dropout, train, bidirectional,
                          batch_first)
    return _CpuIsland.apply(fn, input, *hx, *params)


def _gru_input(input, hx, params, has_biases, num_layers, dropout, train,
               bidirectional, batch_first):
    def fn(*cpu):
        return torch.gru(cpu[0], cpu[1], list(cpu[2:]), has_biases, num_layers,
                         dropout, train, bidirectional, batch_first)
    return _CpuIsland.apply(fn, input, hx, *params)


def install() -> None:
    """Route the ops above through CPU islands (XDNA_CPU_ISLANDS=0 disables)."""
    global _LIB
    if _LIB is not None:
        return
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("lstm.input", _lstm_input, "AutogradPrivateUse1")
    _LIB.impl("gru.input", _gru_input, "AutogradPrivateUse1")
