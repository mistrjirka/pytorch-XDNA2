"""Training regions: several ops run as one unit with an explicit contract.

A *region* replaces a pattern of eager ops by one autograd node whose forward
and backward are fused over host memory, while the heavy ops inside (the
convolutions) still dispatch through the normal XDNA backend, so the cost model
keeps choosing NPU or CPU for each of them.

First region: ``frozen_feature_mse`` -- the perceptual loss pattern

    features = frozen Sequential(conv, relu, conv, relu, maxpool, conv, ...)
    loss = mean over all selected taps of (features(x)[tap] - features(t)[tap])**2

where ``x`` needs gradients and ``t`` does not, and no weight of the feature
network is trained.  The eager path builds clones of the taps, concatenates
them, and runs subtract / square / mean (and their backward) over the
concatenation.  The region never materializes the concatenation or the clones:
it reduces each tap in place, and produces the tap gradients, ReLU and max-pool
backward in one pass each.

Contract
  * inputs: ``x`` (bf16 XDNA tensor, requires grad), ``target`` (bf16 XDNA, no
    grad); the weights are read, never modified, and must not require grad.
  * output: a bf16 scalar equal to the mean over the *concatenated* tap
    elements (layers weighted by element count, as the eager loss does).
  * saved for backward: the op outputs of the forward pass (the taps and the
    ReLU / max-pool inputs) and the target taps.  Backward is not
    double-differentiable; ``retain_graph`` works (nothing is freed).
  * numerics: bf16 storage, fp32 arithmetic in the fused kernels, rounded once
    to bf16.  The eager path rounds the difference, the square and the running
    mean to bf16 separately; the region is at least as close to an fp32
    reference (tests/run_regions.py).
  * eligibility is semantic (op types, dtype, device, requires_grad), never a
    model name or shape; anything else uses the eager path unchanged.
    ``XDNA_REGIONS=0`` disables the regions.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def _native():
    from . import device
    device.register_xdna_device()
    return device._C


def enabled() -> bool:
    return os.environ.get("XDNA_REGIONS", "1") != "0"


class _Placer:
    """Where a region's own convolutions run, chosen by measurement.

    The backend's cost model picks NPU or CPU from static rates; a region knows
    its convolutions run back to back with nothing else competing for the host,
    so it times each (conv, direction) on both placements -- alternating CPU /
    NPU calls, a few warmup calls discarded -- and keeps the faster one.  An NPU
    sample only counts if the NPU really ran the call (the stream can still be
    building or the shape ineligible).  XDNA_REGION_PLACEMENT: ``measure``
    (default), ``cost`` (leave the backend's decision), ``npu`` or ``cpu``.
    """

    WARM, TIMED, GIVE_UP = 2, 4, 48
    MAX_KEYS = 64  # (op, direction, input shape) entries kept

    def __init__(self):
        self.mode = os.environ.get("XDNA_REGION_PLACEMENT", "measure")
        self.state: dict = {}
        self.verbose = os.environ.get("XDNA_REGION_VERBOSE", "0") == "1"

    def call(self, key, fn):
        C = _native()
        if self.mode == "cost":
            return fn()
        if self.mode in ("npu", "cpu"):
            return self._run(C, 1 if self.mode == "npu" else 2, fn)[0]
        st = self.state.get(key)
        if st is None:
            if len(self.state) >= self.MAX_KEYS:  # bounded: forget the oldest shape
                self.state.pop(next(iter(self.state)))
            st = self.state[key] = {"cpu": [], "npu": [], "calls": 0, "choice": None}
        if st["choice"] is not None:
            return self._run(C, st["choice"], fn)[0]
        place = 2 if st["calls"] % 2 == 0 else 1  # CPU first, then NPU, alternating
        st["calls"] += 1
        out, dt, on_npu = self._run(C, place, fn)
        if place == 2:
            st["cpu"].append(dt)
        elif on_npu:
            st["npu"].append(dt)
        n_cpu, n_npu = len(st["cpu"]), len(st["npu"])
        if n_cpu >= self.WARM + self.TIMED and n_npu >= self.WARM + self.TIMED:
            cpu = statistics.median(st["cpu"][self.WARM:])
            npu = statistics.median(st["npu"][self.WARM:])
            st["choice"] = 1 if npu < 0.95 * cpu else 2
            if self.verbose:
                print(f"[regions] {key[1]} op{key[0]} {key[2]}: cpu {cpu:.2f} ms, npu {npu:.2f} ms -> "
                      f"{'npu' if st['choice'] == 1 else 'cpu'}")
        elif st["calls"] >= self.GIVE_UP:
            st["choice"] = 0  # the NPU never ran this shape: leave it to the cost model
            if self.verbose:
                print(f"[regions] {key[1]} op{key[0]} {key[2]}: NPU did not run; cost model decides")
        return out

    @staticmethod
    def _run(C, place, fn):
        prev = C.set_conv_placement(place)
        before = C.conv3x3w_npu_runs()
        t = time.perf_counter()
        try:
            out = fn()
        finally:
            C.set_conv_placement(prev)
        dt = (time.perf_counter() - t) * 1e3
        return out, dt, C.conv3x3w_npu_runs() > before


@dataclass
class _Op:
    kind: str  # "conv" | "relu" | "maxpool2"
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    groups: int = 1


@dataclass
class FrozenFeatureChain:
    ops: list[_Op]
    taps: tuple[int, ...]  # op indices whose output is compared (after the op)
    _tapset: frozenset = field(init=False)
    placer: "_Placer" = field(init=False)

    def __post_init__(self):
        self._tapset = frozenset(self.taps)
        self.placer = _Placer()

    @classmethod
    def from_sequential(cls, layers, taps) -> "FrozenFeatureChain | None":
        """Parse ``layers[0 .. max(taps)]``; None if any op is not supported."""
        layers = list(layers)
        taps = tuple(sorted(set(int(t) for t in taps)))
        if not taps or taps[0] < 0 or taps[-1] >= len(layers):
            return None
        ops: list[_Op] = []
        for m in layers[: taps[-1] + 1]:
            if isinstance(m, nn.Conv2d):
                if m.padding_mode != "zeros" or isinstance(m.padding, str):
                    return None
                if m.weight.requires_grad or (m.bias is not None and m.bias.requires_grad):
                    return None
                ops.append(_Op("conv", m.weight, m.bias, tuple(m.stride), tuple(m.padding),
                               tuple(m.dilation), m.groups))
            elif isinstance(m, nn.ReLU):
                ops.append(_Op("relu"))
            elif (isinstance(m, nn.MaxPool2d) and m.kernel_size in (2, (2, 2))
                  and m.stride in (2, (2, 2)) and m.padding in (0, (0, 0))
                  and m.dilation in (1, (1, 1)) and not m.ceil_mode and not m.return_indices):
                ops.append(_Op("maxpool2"))
            else:
                return None
        if ops[0].kind != "conv":
            return None  # keep the contract simple: the first op reads the image
        return cls(ops, taps)

    def eligible(self, x: torch.Tensor, target: torch.Tensor) -> bool:
        if not enabled() or x.device.type != "xdna" or target.device.type != "xdna":
            return False
        if x.dtype != torch.bfloat16 or target.dtype != torch.bfloat16:
            return False
        if x.dim() != 4 or x.shape != target.shape or target.requires_grad:
            return False
        if torch.is_grad_enabled() and not x.requires_grad:
            return False  # the region's backward produces dX only
        for op in self.ops:
            if op.kind == "conv" and (op.weight.device.type != "xdna"
                                      or op.weight.dtype != torch.bfloat16):
                return False
        return True


def _forward_pass(chain: FrozenFeatureChain, x: torch.Tensor, keep_all: bool):
    """Run the chain; returns outs where outs[i] is op i's output (None if dropped)."""
    C = _native()
    outs: list[torch.Tensor | None] = [None] * len(chain.ops)
    cur = x
    i = 0
    n = len(chain.ops)
    while i < n:
        op = chain.ops[i]
        if op.kind == "conv":
            cur = chain.placer.call((i, "fwd", tuple(cur.shape)), lambda cur=cur, op=op: F.conv2d(
                cur, op.weight, op.bias, op.stride, op.padding, op.dilation, op.groups))
            if not cur.is_contiguous():
                cur = cur.contiguous()
            outs[i] = cur
            i += 1
        elif op.kind == "relu":
            fuse_pool = i + 1 < n and chain.ops[i + 1].kind == "maxpool2" and cur.dim() == 4
            r = torch.empty_like(cur)
            if fuse_pool:
                p = torch.empty(cur.shape[0], cur.shape[1], cur.shape[2] // 2, cur.shape[3] // 2,
                                dtype=cur.dtype, device=cur.device)
                C.region_relu_maxpool2(cur, r, p)
                outs[i], outs[i + 1] = r, p
                cur = p
                i += 2
            else:
                C.region_relu(cur, r)
                outs[i] = r
                cur = r
                i += 1
        else:  # standalone maxpool2 (not preceded by relu)
            cur = F.max_pool2d(cur, 2, 2)
            outs[i] = cur
            i += 1
        if not keep_all:
            # drop what neither a tap nor the next op needs
            for j in range(i - 1):
                if j not in chain._tapset:
                    outs[j] = None
    return outs


class _FrozenFeatureMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chain, x, target):
        C = _native()
        with torch.no_grad():
            g_outs = _forward_pass(chain, x, keep_all=True)
            t_outs = _forward_pass(chain, target, keep_all=False)
            total = 0
            sq = 0.0
            for k in chain.taps:
                total += g_outs[k].numel()
                sq += C.region_sqdiff_sum(g_outs[k], t_outs[k])
        # Everything backward reads goes through save_for_backward, so PyTorch's
        # saved-tensor version checks and hooks apply (in-place changes raise).
        gi = [i for i, t in enumerate(g_outs) if t is not None]
        ti = [i for i, t in enumerate(t_outs) if t is not None]
        weights = [op.weight for op in chain.ops if op.kind == "conv"]
        ctx.save_for_backward(x, *[g_outs[i] for i in gi], *[t_outs[i] for i in ti], *weights)
        ctx.meta = (chain, gi, ti, total)
        return torch.full((), sq / total, dtype=torch.bfloat16, device=x.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        C = _native()
        chain, gi, ti, total = ctx.meta
        saved = ctx.saved_tensors  # raises if a saved tensor was modified in place
        x = saved[0]
        g_outs = [None] * len(chain.ops)
        t_outs = [None] * len(chain.ops)
        for n, i in enumerate(gi):
            g_outs[i] = saved[1 + n]
        for n, i in enumerate(ti):
            t_outs[i] = saved[1 + len(gi) + n]
        weights = iter(saved[1 + len(gi) + len(ti):])
        w_of = {k: next(weights) for k, o in enumerate(chain.ops) if o.kind == "conv"}
        scale = 2.0 * float(go) / total
        G = None  # gradient wrt the output of the op being processed (from above)
        last = chain.taps[-1]
        for k in range(last, -1, -1):
            op = chain.ops[k]
            tapped = k in chain._tapset
            out_k = g_outs[k]
            if op.kind == "relu":
                new = torch.empty_like(out_k)
                if tapped:
                    C.region_relu_bwd(out_k, G, out_k, t_outs[k], new, scale)
                else:
                    C.region_relu_bwd(out_k, G, None, None, new, scale)
                G = new
                continue
            if tapped:  # G (+)= scale * (out - target_out)
                new = torch.empty_like(out_k)
                C.region_addtap(G, out_k, t_outs[k], new, scale)
                G = new
            if op.kind == "maxpool2":
                src = g_outs[k - 1] if k > 0 else x
                new = torch.empty_like(src)
                C.region_maxpool2_bwd(src, G, new)
                G = new
            else:  # conv: dX only
                src = g_outs[k - 1] if k > 0 else x
                if not G.is_contiguous():
                    G = G.contiguous()
                G = chain.placer.call((k, "dx", tuple(src.shape)), lambda G=G, src=src, op=op:
                                      torch.ops.aten.convolution_backward(
                                          G, src, w_of[k], None, list(op.stride), list(op.padding),
                                          list(op.dilation), False, [0, 0], op.groups,
                                          [True, False, False])[0])
        return None, G, None


def frozen_feature_mse(chain: FrozenFeatureChain, x: torch.Tensor, target: torch.Tensor):
    """mean over the concatenated tap elements of (feat(x) - feat(target))**2."""
    return _FrozenFeatureMSE.apply(chain, x, target)


def fuse_perceptual_loss(loss_module, features_attr: str = "perceptual_vgg"):
    """Route ``loss_module.compute_perceptual_loss`` through the region when the
    module follows the common pattern: ``getattr(loss_module, features_attr)``
    has ``model.features`` (a Sequential), ``feature_layers`` (tap indices),
    optional ``use_norm`` / ``norm`` (a Normalize with mean/std), and the loss is
    the mean squared tap difference.  Returns True if the region was installed;
    inputs the region does not accept still take the module's own method.
    """
    if not enabled():
        return False
    net = getattr(loss_module, features_attr, None)
    feats = getattr(getattr(net, "model", None), "features", None)
    taps = getattr(net, "feature_layers", None)
    if feats is None or taps is None:
        return False
    chain = FrozenFeatureChain.from_sequential(feats, taps)
    if chain is None:
        return False
    norm = getattr(net, "norm", None) if getattr(net, "use_norm", False) else None
    mean = std = None
    if norm is not None:
        mean, std = getattr(norm, "mean", None), getattr(norm, "std", None)
        if mean is None or std is None:
            return False
    original = loss_module.compute_perceptual_loss

    def compute_perceptual_loss(input, target):
        if not chain.eligible(input, target):
            return original(input, target)
        x, t = input, target
        if mean is not None:
            m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            x = (x - m) / s
            t = (t - m) / s
        return frozen_feature_mse(chain, x, t)

    loss_module.compute_perceptual_loss = compute_perceptual_loss
    return True
