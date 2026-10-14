#!/usr/bin/env python3
"""CPU-resident sparse embeddings feeding a dense XDNA model (xdna_train.cpu_sparse).

Reference: the identical model entirely on CPU (FP32), same init, same optimizers.
Covers placement (.to("xdna") must not move the table), sparse gradients, duplicate
and untouched rows, padding_idx, empty batches, EmbeddingBag, dtype delivery,
gradient accumulation, and checkpoint restore.

Run with a Python with PyTorch 2.14 and PYTHONPATH=<repo>.
"""
import copy
import io
import sys

import torch
import torch.nn.functional as F
from torch import nn

import xdna_train
from xdna_train.cpu_sparse import CpuSparseEmbedding, CpuSparseEmbeddingBag, split_parameters

xdna_train.register_xdna_device()
failures = []


def check(name, cond, detail=""):
    print(f"{name}: {'OK' if cond else 'FAIL'} {detail}")
    if not cond:
        failures.append(name)


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


class Rec(nn.Module):
    def __init__(self, rows=200, dim=16, bag=False):
        super().__init__()
        self.bag = bag
        self.emb = CpuSparseEmbeddingBag(rows, dim, mode="sum") if bag else CpuSparseEmbedding(rows, dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ids, offsets=None, device=None):
        e = self.emb(ids, offsets, device=device) if self.bag else self.emb(ids, device=device)
        return self.mlp(e)


def train(model, ids, y, steps, dev, offsets=None):
    sp, dn = split_parameters(model)
    opts = [torch.optim.SparseAdam(sp, lr=0.05), torch.optim.Adam(dn, lr=0.05)]
    losses = []
    for _ in range(steps):
        for o in opts:
            o.zero_grad()
        pred = model(ids, offsets, device=dev) if offsets is not None else model(ids, device=dev)
        loss = F.mse_loss(pred.float().squeeze(-1), y.to(pred.device))
        loss.backward()
        for o in opts:
            o.step()
        losses.append(loss.item())
    return losses


torch.manual_seed(0)
base = Rec()
ids = torch.randint(0, 200, (64,))
ids[:5] = 7            # duplicate indices
ids[5:8] = 0           # padding index
y = torch.randn(64)

# 1. placement ---------------------------------------------------------------
m = copy.deepcopy(base).to("xdna")
check("table stays on CPU after .to('xdna')", m.emb.weight.device.type == "cpu")
check("dense layers moved", all(p.device.type == "xdna" for p in m.mlp.parameters()))
sp, dn = split_parameters(m)
check("split_parameters", len(sp) == 1 and sp[0] is m.emb.weight and len(dn) == 4)
out = m(ids.to("xdna"), device="xdna")
check("lookup result lives on xdna", out.device.type == "xdna")
check("lookup default follows the indices' device", m.emb(ids.to("xdna")).device.type == "xdna")
check("lookup default on CPU indices stays CPU", m.emb(ids).device.type == "cpu")

# 2. sparse gradient, duplicates, untouched rows ------------------------------------
m.zero_grad()
F.mse_loss(m(ids, device="xdna").float().squeeze(-1), y.to("xdna")).backward()
g = m.emb.weight.grad
check("table gradient is sparse COO", g is not None and g.is_sparse)
touched = set(ids.tolist()) - {0}
rows = set(g.coalesce().indices()[0].tolist())
check("only looked-up rows carry gradient", rows <= set(ids.tolist()) and touched <= rows, f"{len(rows)} rows")

# 3. training tracks the all-CPU reference -------------------------------------------
ref = copy.deepcopy(base)
mx = copy.deepcopy(base).to("xdna")
ref_losses = train(ref, ids, y, 8, None)
x_losses = train(mx, ids, y, 8, "xdna")
check("loss trajectory matches CPU", max(abs(a - b) / max(abs(a), 1e-6) for a, b in zip(ref_losses, x_losses)) < 0.02,
      f"{ref_losses[-1]:.4f} vs {x_losses[-1]:.4f}")
check("table matches CPU reference", rel(mx.emb.weight, ref.emb.weight) < 0.02, f"rel={rel(mx.emb.weight, ref.emb.weight):.4f}")
untouched = torch.tensor(sorted(set(range(200)) - set(ids.tolist())))
check("untouched rows are bit-identical to init",
      torch.equal(mx.emb.weight.detach()[untouched], base.emb.weight.detach()[untouched]))
check("padding row never updated", torch.equal(mx.emb.weight.detach()[0], base.emb.weight.detach()[0]))

# 4. empty batch ---------------------------------------------------------------------------
e = copy.deepcopy(base).to("xdna")
out = e(torch.empty(0, dtype=torch.long), device="xdna")
check("empty batch gives an empty xdna result", out.shape[0] == 0 and out.device.type == "xdna")
out.sum().backward()
check("empty batch backward", e.emb.weight.grad is None or e.emb.weight.grad._nnz() == 0)

# 5. dtype delivery and gradient accumulation ------------------------------------------------
bf = copy.deepcopy(base).to("xdna")
bf.mlp.bfloat16()
o = bf.emb(ids, device="xdna", dtype=torch.bfloat16)
check("delivered in bf16", o.dtype == torch.bfloat16 and o.device.type == "xdna")
bf.zero_grad()
for _ in range(2):
    bf.mlp(bf.emb(ids, device="xdna", dtype=torch.bfloat16)).float().sum().backward()
once = copy.deepcopy(base).to("xdna")
once.mlp.bfloat16()
once.mlp(once.emb(ids, device="xdna", dtype=torch.bfloat16)).float().sum().backward()
check("sparse gradients accumulate over micro-batches",
      rel(bf.emb.weight.grad.to_dense(), 2 * once.emb.weight.grad.to_dense()) < 0.03)
check("dtype conversion reaches the table", copy.deepcopy(base).to(torch.float64).emb.weight.dtype == torch.float64)

# 6. EmbeddingBag ------------------------------------------------------------------------------
torch.manual_seed(3)
bb = Rec(bag=True)
flat = torch.randint(0, 200, (40,))
offs = torch.arange(0, 40, 5)
yb = torch.randn(len(offs))
rb, xb = copy.deepcopy(bb), copy.deepcopy(bb).to("xdna")
lr_ = train(rb, flat, yb, 5, None, offs)
lx_ = train(xb, flat, yb, 5, "xdna", offs)
check("EmbeddingBag trains like CPU", abs(lr_[-1] - lx_[-1]) / abs(lr_[-1]) < 0.02, f"{lr_[-1]:.4f} vs {lx_[-1]:.4f}")

# 7. checkpoint ----------------------------------------------------------------------------------
buf = io.BytesIO()
torch.save(mx.state_dict(), buf)
buf.seek(0)
fresh = Rec().to("xdna")
fresh.load_state_dict(torch.load(buf))
check("state_dict round trip", torch.equal(fresh.emb.weight.detach(), mx.emb.weight.detach())
      and rel(fresh.mlp[0].weight, mx.mlp[0].weight) < 1e-6)

print("ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
