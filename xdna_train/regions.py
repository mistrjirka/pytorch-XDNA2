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
import types
import weakref
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn.modules import module as _modmod
from torch.nn.utils import parametrize
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


def _plain(m, cls) -> bool:
    """m is exactly ``cls`` with nothing around its forward that the region would skip
    (hooks, an instance-level forward, parametrizations)."""
    if type(m) is not cls or "forward" in m.__dict__:
        return False
    if (m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks):
        return False
    if _modmod._global_forward_hooks or _modmod._global_forward_pre_hooks:
        return False
    return not parametrize.is_parametrized(m)


def _layer_sig(m):
    """Everything the parsed chain depends on for one layer, or None if the layer is
    not supported.  Compared on every call, so model surgery after the chain was
    built (a replaced activation, a changed stride, a new hook) is never missed."""
    if _plain(m, nn.Conv2d):
        if m.padding_mode != "zeros" or isinstance(m.padding, str):
            return None
        return ("conv", id(m.weight), id(m.bias), tuple(m.stride), tuple(m.padding),
                tuple(m.dilation), m.groups)
    if _plain(m, nn.ReLU):
        return ("relu",)
    if (_plain(m, nn.MaxPool2d) and m.kernel_size in (2, (2, 2)) and m.stride in (2, (2, 2))
            and m.padding in (0, (0, 0)) and m.dilation in (1, (1, 1)) and not m.ceil_mode
            and not m.return_indices):
        return ("maxpool2",)
    return None


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
    sig: tuple = field(init=False, default=())

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
        sigs = []
        for m in layers[: taps[-1] + 1]:
            sig = _layer_sig(m)
            if sig is None:
                return None
            sigs.append(sig)
            if sig[0] == "conv":
                if m.weight.requires_grad or (m.bias is not None and m.bias.requires_grad):
                    return None
                ops.append(_Op("conv", m.weight, m.bias, tuple(m.stride), tuple(m.padding),
                               tuple(m.dilation), m.groups))
            else:
                ops.append(_Op(sig[0]))
        if ops[0].kind != "conv":
            return None  # keep the contract simple: the first op reads the image
        chain = cls(ops, taps)
        chain.sig = tuple(sigs)
        return chain

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
            if op.kind != "conv":
                continue
            if op.weight.device.type != "xdna" or op.weight.dtype != torch.bfloat16:
                return False
            # the region produces dX only: weights that became trainable after the
            # region was installed must take the eager path (and get their gradients)
            if op.weight.requires_grad or (op.bias is not None and op.bias.requires_grad):
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


def _loss_chain(loss_module):
    """The region chain for the module's CURRENT feature layers (cached on the
    module; rebuilt if the layers or their weights were replaced), or None."""
    cfg = loss_module.__dict__["_xdna_region_cfg"]
    net = getattr(loss_module, cfg["features_attr"], None)
    feats = getattr(getattr(net, "model", None), "features", None)
    taps = getattr(net, "feature_layers", None)
    if feats is None or taps is None:
        return None
    layers = list(feats)[: max(taps) + 1] if len(taps) else []
    chain = loss_module.__dict__.get("_xdna_chain")
    if chain is not None:
        want = tuple(sorted(set(int(t) for t in taps)))
        if (chain.taps != want or chain.sig != tuple(_layer_sig(m) for m in layers)):
            chain = None
    if chain is None:
        chain = FrozenFeatureChain.from_sequential(layers, taps)
        loss_module.__dict__["_xdna_chain"] = chain
    return chain


def _fused_compute_perceptual_loss(self, input, target):
    """Bound to the loss module (types.MethodType), so a deepcopy rebinds it to the
    copy; everything is read from ``self`` at call time."""
    chain = _loss_chain(self)
    if chain is None or not enabled() or not chain.eligible(input, target):
        return type(self).compute_perceptual_loss(self, input, target)
    net = getattr(self, self.__dict__["_xdna_region_cfg"]["features_attr"])
    x, t = input, target
    norm = getattr(net, "norm", None) if getattr(net, "use_norm", False) else None
    if norm is not None:
        mean, std = getattr(norm, "mean", None), getattr(norm, "std", None)
        if mean is None or std is None:
            return type(self).compute_perceptual_loss(self, input, target)
        m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        x = (x - m) / s
        t = (t - m) / s
    return frozen_feature_mse(chain, x, t)


def fuse_perceptual_loss(loss_module, features_attr: str = "perceptual_vgg"):
    """Route ``loss_module.compute_perceptual_loss`` through the region when the
    module follows the common pattern: ``getattr(loss_module, features_attr)``
    has ``model.features`` (a Sequential), ``feature_layers`` (tap indices),
    optional ``use_norm`` / ``norm`` (a Normalize with mean/std), and the loss is
    the mean squared tap difference.  Returns True if the region was installed.

    Every call re-checks eligibility against the module's current state: inputs or
    layers the region does not accept (trainable feature weights, other dtypes,
    replaced layers) take the module's own method, so nothing is silently dropped.
    The wrapper is a method bound to ``loss_module``, so ``copy.deepcopy`` of the
    module gives a copy that uses its own weights.
    """
    if not enabled():
        return False
    net = getattr(loss_module, features_attr, None)
    feats = getattr(getattr(net, "model", None), "features", None)
    taps = getattr(net, "feature_layers", None)
    if feats is None or taps is None or not hasattr(loss_module, "compute_perceptual_loss"):
        return False
    if FrozenFeatureChain.from_sequential(list(feats)[: max(taps) + 1], taps) is None:
        return False
    if getattr(net, "use_norm", False):
        norm = getattr(net, "norm", None)
        if getattr(norm, "mean", None) is None or getattr(norm, "std", None) is None:
            return False
    loss_module.__dict__["_xdna_region_cfg"] = {"features_attr": features_attr}
    loss_module.compute_perceptual_loss = types.MethodType(_fused_compute_perceptual_loss, loss_module)
    return True


# ---------------------------------------------------------------------------
# Training BatchNorm2d + LeakyReLU/ReLU as one op
# ---------------------------------------------------------------------------
class _BatchNormAct(torch.autograd.Function):
    """y = act(batch_norm(x)) with training statistics, on contiguous NCHW bf16.

    Contract: statistics over (N, H, W) of every element, biased variance for
    the normalisation, unbiased variance for the running update (once per call),
    affine weight/bias.  Saved for backward: x, y (the activation mask is y > 0),
    weight, and the fp32 per-channel mean / inverse std.  Not double
    differentiable.  Gradients: input, weight and bias, each only if requested.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps, slope, factor, relu):
        C = _native()
        y = torch.empty_like(x)
        mean = torch.empty(x.shape[1], dtype=torch.float32, device=x.device)
        invstd = torch.empty_like(mean)
        C.region_bn_fwd(x, weight, bias, running_mean, running_var, y, mean, invstd,
                        eps, slope, factor, True)
        ctx.save_for_backward(x, y, weight, mean, invstd)
        ctx.slope = slope
        ctx.relu = relu
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        C = _native()
        x, y, weight, mean, invstd = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0:3]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dx = torch.empty_like(x) if need_x else None
        dw = torch.empty_like(weight) if need_w else None
        db = torch.empty_like(weight) if need_b else None
        C.region_bn_bwd(dy, y, x, weight, mean, invstd, dx, dw, db, ctx.slope, ctx.relu)
        return dx, dw, db, None, None, None, None, None, None


# Below this size the eager BatchNorm + activation is as fast as the fused op (the
# fused kernel wins on large tensors only: 13 MB 3.5 -> 2.3 ms, 6.6 MB 1.2 -> 0.8 ms,
# 3.3 MB and smaller no gain), so small layers keep the eager pair.
_NORM_ACT_MIN_BYTES = int(os.environ.get("XDNA_REGION_NORM_ACT_MIN_BYTES", "4000000"))


def _norm_act_eligible(bn: nn.BatchNorm2d, x: torch.Tensor) -> bool:
    if not enabled() or not bn.training or not bn.track_running_stats or not bn.affine:
        return False
    if bn.momentum is None:  # cumulative moving average: leave to the eager path
        return False
    if not isinstance(x, torch.Tensor) or x.dim() != 4 or x.device.type != "xdna":
        return False
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.numel() // x.shape[1] <= 1:
        return False
    if x.numel() * 2 < _NORM_ACT_MIN_BYTES:
        return False
    for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
        if (t is None or t.device.type != "xdna" or t.dtype != torch.bfloat16
                or not t.is_contiguous() or t.numel() != x.shape[1]):
            return False
    return True


def _act_slope(act):
    # exact types: a subclass may override forward, which the fused op would skip
    if type(act) is nn.LeakyReLU:
        return float(act.negative_slope)
    if type(act) is nn.ReLU:
        return 0.0
    return None


def _fused_bn_forward(self, x):
    """Bound to the BatchNorm module (types.MethodType): a deepcopy rebinds it to
    the copy.  The paired activation is read from ``self`` and its slope at call time."""
    act = self.__dict__.get("_xdna_pair")
    slope = _act_slope(act) if act is not None else None
    if slope is None or slope < 0.0 or not _norm_act_eligible(self, x):
        return type(self).forward(self, x)
    self.num_batches_tracked.add_(1)
    y = _BatchNormAct.apply(x, self.weight, self.bias, self.running_mean, self.running_var,
                            float(self.eps), slope, float(self.momentum), type(act) is nn.ReLU)
    y._xdna_act_for = weakref.ref(act)
    return y


def _checked_act_forward(self, x):
    """Pass through a tensor that the paired fused norm already activated -- once --
    and activate everything else."""
    ref = getattr(x, "_xdna_act_for", None)
    if ref is not None and ref() is self:
        x._xdna_act_for = None  # consumed: a later call on the same tensor activates
        return x
    return type(self).forward(self, x)


def fuse_norm_activation(bn: nn.BatchNorm2d, act: nn.Module) -> bool:
    """Declare that ``act`` is applied directly to the output of ``bn`` (like
    ``torch.ao.quantization.fuse_modules``) and run the pair as one fused op.

    Both modules keep their parameters, buffers and state_dict keys; only their
    ``forward`` is wrapped, by methods bound to the modules themselves, so
    ``copy.deepcopy`` of a model containing the pair gives a working fused pair
    (copy the two modules together: the pairing is stored on the norm).  When the
    fused op is eligible (training mode, bf16 contiguous NCHW on XDNA, affine,
    tracked statistics with a float momentum, a large enough tensor), ``bn(x)``
    returns the *activated* tensor and the next call of ``act`` on exactly that
    tensor passes it through once; anything else runs both modules unchanged.  The
    activation's current ``negative_slope`` is read on every call.

    The pairing is the caller's statement about the model's dataflow.  If the
    activation is not applied to the norm output directly (an add or a branch in
    between), the activation would be applied twice or to the wrong tensor: do
    not declare such a pair.  Returns True if installed.
    """
    if not enabled() or type(bn) is not nn.BatchNorm2d or type(act) not in (nn.LeakyReLU, nn.ReLU):
        return False
    if bn.__dict__.get("_xdna_pair") is not None:
        return bn.__dict__["_xdna_pair"] is act
    object.__setattr__(bn, "_xdna_pair", act)  # plain attribute, not a registered submodule
    bn.forward = types.MethodType(_fused_bn_forward, bn)
    if not isinstance(act.forward, types.MethodType) or act.forward.__func__ is not _checked_act_forward:
        act.forward = types.MethodType(_checked_act_forward, act)
    return True
