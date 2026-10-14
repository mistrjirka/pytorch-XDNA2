"""CPU-resident sparse embeddings that feed a dense model running on XDNA.

XDNA tensors have no sparse layout, so a large embedding table with sparse
gradients cannot live on the device.  The table stays in host memory as an ordinary
CPU parameter, its lookups run on the CPU, and only the dense result crosses to the
device with a normal differentiable copy.  The gradient comes back dense (one row
per lookup), the CPU embedding backward turns it into a sparse COO gradient, and a
sparse-aware CPU optimizer updates only the touched rows.

    emb = CpuSparseEmbedding(1_000_000, 64)            # stays on the CPU
    net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 1)).to("xdna")
    sparse_params, dense_params = split_parameters(nn.ModuleList([emb, net]))
    opt_sparse = torch.optim.SparseAdam(sparse_params, lr=1e-3)
    opt_dense = torch.optim.Adam(dense_params, lr=1e-3)

    out = net(emb(ids, device="xdna"))                  # ids may be CPU or XDNA tensors

Placement is explicit.  ``module.to("xdna")`` on a model that contains these modules
moves the dense layers and leaves the tables where they are (a dtype conversion does
reach the table).  Lookups return CPU tensors unless ``device=`` is given or the
indices already live on an accelerator.

This is a compatibility bridge, not sparse acceleration: the lookup and the sparse
update run on the host.  Per-step cost is the host->device copy of the looked-up rows.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CpuSparseEmbedding", "CpuSparseEmbeddingBag", "split_parameters"]


class _CpuTable(nn.Module):
    """Shared behaviour: the table never leaves the CPU through ``module.to()``."""

    weight: nn.Parameter

    def _apply(self, fn, recurse=True):
        # fn is a device/dtype move built by Module.to()/.bfloat16()/.cuda()...
        # Apply it to a probe to learn the requested dtype; ignore any device change.
        probe = fn(torch.empty(0, dtype=self.weight.dtype))
        if probe.dtype != self.weight.dtype and self.weight.is_floating_point():
            with torch.no_grad():
                self.weight.data = self.weight.data.to(probe.dtype)
        return self

    def _target(self, indices: torch.Tensor, device) -> torch.device:
        if device is not None:
            return torch.device(device)
        if indices.device.type != "cpu":
            return indices.device
        return torch.device("cpu")

    def _deliver(self, out: torch.Tensor, device, dtype) -> torch.Tensor:
        target = device
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)  # differentiable cast on the host (cheaper than on the device)
        if target.type != "cpu":
            out = out.to(target)
        return out


class CpuSparseEmbedding(_CpuTable):
    """``nn.Embedding(sparse=True)`` whose table stays on the CPU.

    Arguments follow ``nn.Embedding``; ``forward(indices, device=None, dtype=None)``
    returns the rows on ``device`` (default: the indices' device if it is not the CPU)
    in ``dtype`` (default: the table's).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, _weight: Optional[torch.Tensor] = None,
                 dtype=None):
        super().__init__()
        self.inner = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                  max_norm=max_norm, norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq, sparse=True,
                                  _weight=_weight, dtype=dtype)

    @property
    def weight(self) -> nn.Parameter:  # type: ignore[override]
        return self.inner.weight

    def forward(self, indices: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        target = self._target(indices, device)
        out = self.inner(indices.cpu() if indices.device.type != "cpu" else indices)
        return self._deliver(out, target, dtype)

    def extra_repr(self) -> str:
        return f"{self.inner.num_embeddings}, {self.inner.embedding_dim}, placement=cpu, sparse_grad=True"


class CpuSparseEmbeddingBag(_CpuTable):
    """``nn.EmbeddingBag(sparse=True)`` (mode sum/mean/max) whose table stays on the CPU."""

    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = "mean",
                 padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
                 norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 include_last_offset: bool = False, dtype=None):
        super().__init__()
        self.inner = nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=max_norm,
                                     norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                     mode=mode, sparse=True, include_last_offset=include_last_offset,
                                     padding_idx=padding_idx, dtype=dtype)

    @property
    def weight(self) -> nn.Parameter:  # type: ignore[override]
        return self.inner.weight

    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None, device=None, dtype=None) -> torch.Tensor:
        target = self._target(indices, device)
        cpu = lambda t: None if t is None else (t.cpu() if t.device.type != "cpu" else t)
        out = self.inner(cpu(indices), cpu(offsets), cpu(per_sample_weights))
        return self._deliver(out, target, dtype)


def split_parameters(module: nn.Module):
    """Return ``(sparse_params, dense_params)`` for optimizer construction.

    Sparse parameters are the tables of the CPU sparse modules (they need a
    sparse-aware optimizer such as ``SparseAdam`` or ``SGD``/``Adagrad``); everything
    else is dense.  Shared parameters appear once.
    """
    sparse_ids = {id(m.weight) for m in module.modules() if isinstance(m, _CpuTable)}
    sparse, dense, seen = [], [], set()
    for p in module.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (sparse if id(p) in sparse_ids else dense).append(p)
    return sparse, dense
