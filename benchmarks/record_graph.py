#!/usr/bin/env python3
"""Record the aten op graph of one training step (forward, backward, optimizer).

The graph is the input for fusion planning: which op produces each tensor,
which ops read it, how many bytes it has, and which ops ran on the NPU.  It is
recorded with a TorchDispatchMode below autograd, so backward and optimizer ops
appear as the backend sees them.  No backend change is involved.

    PYTHONPATH=<repo> python benchmarks/record_graph.py --model style_net
    python benchmarks/record_graph.py --factory my_nets.py:lab04 --out graph.json

--model takes a benchmarks/model_zoo.py name.  --factory takes file.py:function
returning (model, make_batch(generator) -> (x, y), loss_fn), like the zoo.
Prints a summary; --out also writes the full graph as JSON.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))

# Op classes for the summary.  Pointwise ops are the candidates for lazy values
# that a consumer applies while reading; norms need whole-batch statistics.
_CLASSES = [
    ("conv", ("convolution",)),
    ("matmul", ("mm", "addmm", "bmm", "matmul", "linear", "baddbmm")),
    ("norm", ("batch_norm", "layer_norm", "group_norm", "native_batch_norm")),
    ("optimizer", ("_fused_adam", "_foreach_", "lerp", "addcdiv", "addcmul")),
    ("layout", ("cat", "upsample", "clone", "copy", "_to_copy", "contiguous", "pad",
                "constant_pad", "slice", "select", "view", "reshape", "permute", "transpose",
                "expand", "squeeze", "unsqueeze", "as_strided", "t", "split", "narrow",
                "detach", "alias", "max_pool", "avg_pool")),
    ("reduction", ("sum", "mean", "amax", "max", "min", "norm", "var", "std")),
    ("pointwise", ("add", "sub", "mul", "div", "relu", "leaky_relu", "gelu", "silu", "sigmoid",
                   "tanh", "threshold", "where", "neg", "abs", "pow", "sqrt", "rsqrt", "exp",
                   "log", "clamp", "fill", "zero", "masked", "sign", "hardtanh", "erf",
                   "_backward")),
]


def op_class(name: str) -> str:
    base = name.split(".")[0].replace("aten::", "")
    for cls, keys in _CLASSES:
        if any(base == k or base.startswith(k) for k in keys):
            return cls
    return "other"


class GraphRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.phase = "forward"
        self._ids = {}
        self._next = 0

    @staticmethod
    def _key(t):
        return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))

    def _id(self, t, new):
        key = self._key(t)
        if new or key not in self._ids:
            self._ids[key] = self._next
            self._next += 1
        return self._ids[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        self.nodes.append({
            "op": str(func.overloadpacket.__name__) + "." + func._overloadname,
            "phase": self.phase,
            "inputs": [self._id(t, False) for t in ins],
            "outputs": [self._id(t, True) for t in outs],
            "shapes": [list(t.shape) for t in outs],
            "dtype": str(outs[0].dtype) if outs else None,
            "bytes": int(sum(t.numel() * t.element_size() for t in outs)),
        })
        return out


def load_factory(spec: str):
    if ":" in spec:
        path, fn = spec.rsplit(":", 1)
        mod_spec = importlib.util.spec_from_file_location("graph_factory", path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return getattr(mod, fn)
    sys.path.insert(0, HERE)
    from model_zoo import MODELS
    return MODELS[spec]


def summarize(nodes, npu_calls):
    produced_by = {}
    consumers = collections.defaultdict(list)
    for i, n in enumerate(nodes):
        for t in n["outputs"]:
            produced_by[t] = i
        for t in n["inputs"]:
            consumers[t].append(i)
    lines = []
    by_class = collections.defaultdict(lambda: [0, 0])
    for n in nodes:
        c = by_class[(n["phase"], op_class(n["op"]))]
        c[0] += 1
        c[1] += n["bytes"]
    lines.append("ops and output MB per phase and class:")
    for (phase, cls), (count, nbytes) in sorted(by_class.items()):
        lines.append(f"  {phase:9s} {cls:10s} {count:5d} ops {nbytes / 2**20:9.1f} MB")

    # Producer -> consumer edges between op classes, skipping view-like ops.
    def real_consumers(t, depth=0):
        out = []
        for j in consumers.get(t, []):
            n = nodes[j]
            if op_class(n["op"]) == "layout" and n["op"].split(".")[0] in (
                    "view", "reshape", "detach", "alias", "_unsafe_view", "t", "expand") and depth < 4:
                for o in n["outputs"]:
                    out += real_consumers(o, depth + 1)
            else:
                out.append(j)
        return out

    edges = collections.Counter()
    for i, n in enumerate(nodes):
        if n["phase"] != "forward":
            continue
        for t in n["outputs"]:
            for j in real_consumers(t):
                edges[(n["op"].split(".")[0], nodes[j]["op"].split(".")[0])] += 1
    lines.append("forward producer -> consumer edges (top 25):")
    for (a, b), k in edges.most_common(25):
        lines.append(f"  {k:4d}  {a} -> {b}")

    # Forward pointwise / norm outputs: does every reader run a conv (the
    # consumers that apply lazy values today), or does something else read it?
    kinds = collections.Counter()
    for i, n in enumerate(nodes):
        if n["phase"] != "forward" or op_class(n["op"]) not in ("pointwise", "norm"):
            continue
        readers = {nodes[j]["op"].split(".")[0] for t in n["outputs"][:1] for j in real_consumers(t)
                   if nodes[j]["phase"] == "forward"}
        conv_only = bool(readers) and readers <= {"convolution", "convolution_overrideable",
                                                  "leaky_relu", "relu"}
        kinds[(n["op"].split(".")[0], "only conv/activation" if conv_only else
               "also " + ",".join(sorted(readers - {"convolution", "convolution_overrideable"}))[:60])] += 1
    lines.append("forward pointwise/norm outputs by readers:")
    for (op, readers), k in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:4d}  {op:28s} {readers}")
    if npu_calls:
        lines.append("backend counters for this step (mapped_cpu_stats calls):")
        for name, calls in sorted(npu_calls.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {calls:5d}  {name}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="style_net")
    ap.add_argument("--factory", help="file.py:function, instead of --model")
    ap.add_argument("--device", default="xdna")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.device == "xdna":
        os.environ.setdefault("XDNA_PROFILE_MAPPED_CPU", "1")
        import xdna_train
        xdna_train.register_xdna_device()
    torch.manual_seed(0)
    model, make_batch, loss_fn = load_factory(args.factory or args.model)()
    model = model.to(device=args.device, dtype=torch.bfloat16).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = make_batch(torch.Generator().manual_seed(1))
    x = x.to(args.device)
    y = y.to(args.device)
    x = x.to(torch.bfloat16) if x.is_floating_point() else x
    y = y.to(torch.bfloat16) if y.is_floating_point() else y

    def step(rec=None):
        if rec: rec.phase = "forward"
        loss = loss_fn(model(x), y)
        if rec: rec.phase = "backward"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if rec: rec.phase = "optimizer"
        opt.step()
        return float(loss.item())

    for _ in range(args.warmup):
        step()
    npu_calls = {}
    if args.device == "xdna":
        xdna_train.wait_for_builds()
        step()
        xdna_train.reset_mapped_cpu_stats()
        lazy0 = {k: v for k, v in xdna_train.allocator_stats().items() if k.startswith("lazy")}
    rec = GraphRecorder()
    with rec:
        step(rec)
    if args.device == "xdna":
        npu_calls = {k: v["calls"] for k, v in xdna_train.mapped_cpu_stats().items()}
        lazy1 = {k: v for k, v in xdna_train.allocator_stats().items() if k.startswith("lazy")}
        npu_calls.update({k: lazy1[k] - lazy0[k] for k in lazy1})
    print(f"model {args.factory or args.model}: {len(rec.nodes)} aten ops in one step")
    print(summarize(rec.nodes, npu_calls))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.factory or args.model, "nodes": rec.nodes}, f)


if __name__ == "__main__":
    main()
