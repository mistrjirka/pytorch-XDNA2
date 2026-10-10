"""Physical check: XDNA ops that run CPU kernels (zero-copy fallback, output
adoption, functional conv) match CPU exactly and keep results on xdna."""
import os
os.environ["XDNA_CONV3X3W"] = "0"  # exact CPU path; conv3x3w has its own test
import torch, torch.nn.functional as F
import xdna_train
xdna_train.register_xdna_device()
torch.manual_seed(0); bf = torch.bfloat16
X = lambda t: t.to("xdna")
def chk(name, a, b, tol=0):
    if isinstance(a, torch.Tensor):
        assert a.device.type == "xdna", (name, a.device)
        a = a.cpu()
    torch.testing.assert_close(a, b, rtol=tol, atol=tol); print("ok", name)
x = torch.randn(4, 8, 12, 12, dtype=bf); xx = X(x)
chk("max_pool2d", F.max_pool2d(xx, 2), F.max_pool2d(x, 2))
v, i = F.max_pool2d(xx, 2, return_indices=True); v0, i0 = F.max_pool2d(x, 2, return_indices=True)
chk("max_pool idx", i, i0); chk("max_pool val", v, v0)
chk("pow", xx ** 2, x ** 2); chk("mean", xx.mean(), x.mean()); chk("mean dim", xx.mean((2, 3)), x.mean((2, 3)))
y = xx.clone(); y.relu_(); y0 = x.clone(); y0.relu_(); chk("relu_ inplace", y, y0)
chk("abs", xx.abs(), x.abs()); chk("eq", xx == 0, x == 0); chk("any", (xx > 5).any(), (x > 5).any())
o = X(torch.empty(4, 8, 12, 12, dtype=bf)); torch.exp(xx, out=o); chk("exp out=", o, torch.exp(x))
o = X(torch.empty(0, dtype=bf)); torch.exp(xx, out=o); chk("exp out=empty(resize)", o, torch.exp(x))
chk("stack(list)", torch.stack([xx, xx]), torch.stack([x, x]))
chk("topk", xx.flatten(1).topk(3).values, x.flatten(1).topk(3).values)
chk("sort idx", xx.flatten().sort().indices, x.flatten().sort().indices)
chk("index", xx[X(torch.tensor([0, 2]))], x[torch.tensor([0, 2])])
chk("clamp", xx.clamp(-0.5, 0.5), x.clamp(-0.5, 0.5)); chk("item", xx.sum().item(), x.sum().item())
big = torch.randn(8, 64, 40, 40, dtype=bf)  # >64 KB results exercise adoption
chk("big exp (adopted)", torch.exp(X(big)), torch.exp(big))
chk("big softmax (adopted)", torch.softmax(X(big), 1), torch.softmax(big, 1))
a = x.clone().requires_grad_(); b = X(x).requires_grad_()
F.max_pool2d(F.relu(a), 2).pow(2).mean().backward(); F.max_pool2d(F.relu(b), 2).pow(2).mean().backward()
chk("grad relu/maxpool/pow/mean", b.grad, a.grad)
chk("non-contig input exp", torch.exp(xx.permute(0, 2, 3, 1)), torch.exp(x.permute(0, 2, 3, 1)))
# Convolutions through the CPU-mapped path (functional + adoption), all masks.
for (ci, co, k, s, bias) in [(16, 32, 3, 1, False), (16, 32, 3, 2, True), (32, 8, 1, 1, True), (195, 64, 3, 1, False)]:
    w = torch.randn(co, ci, k, k, dtype=bf) * 0.1; bb = torch.randn(co, dtype=bf) if bias else None
    inp = torch.randn(4, ci, 20, 20, dtype=bf)
    ref = F.conv2d(inp, w, bb, stride=s, padding=k // 2)
    got = F.conv2d(X(inp), X(w), X(bb) if bias else None, stride=s, padding=k // 2)
    chk(f"conv fwd {ci}->{co} k{k} s{s} bias={bias}", got, ref)
    go = torch.randn_like(ref)
    A = torch.ops.aten.convolution_backward
    for mask in ([True, True, bias], [True, False, False], [False, True, False]):
        r = A(go, inp, w, [co] if bias else None, [s, s], [k // 2] * 2, [1, 1], False, [0, 0], 1, mask)
        g = A(X(go), X(inp), X(w), [co] if bias else None, [s, s], [k // 2] * 2, [1, 1], False, [0, 0], 1, mask)
        for j in range(3):
            if mask[j]:
                chk(f"conv bwd {ci}->{co} k{k} s{s} mask{mask} out{j}", g[j], r[j])
            else:
                assert g[j] is None or not g[j].defined() if hasattr(g[j], "defined") else True
print("ALL OK")
# Recurrent layers route non-CPU devices to fused cells (custom XDNA kernels).
for cls in (torch.nn.LSTM, torch.nn.GRU):
    torch.manual_seed(3)
    ref = cls(16, 32, num_layers=2, batch_first=True)
    import copy
    dev = copy.deepcopy(ref).to("xdna", torch.bfloat16)
    inp = torch.randn(4, 7, 16)
    a = inp.clone().requires_grad_(); b = X(inp.bfloat16()).requires_grad_()
    ya = ref(a)[0]; yb = dev(b)[0]
    assert yb.device.type == "xdna"
    torch.testing.assert_close(yb.float().cpu(), ya, rtol=0.05, atol=0.03); print("ok", cls.__name__, "fwd vs fp32")
    ya.square().sum().backward(); yb.float().square().sum().backward()
    torch.testing.assert_close(b.grad.float().cpu(), a.grad, rtol=0.1, atol=0.05); print("ok", cls.__name__, "input grad vs fp32")
    gref = ref.weight_hh_l0.grad; gdev = dev.weight_hh_l0.grad.float().cpu()
    rel = (gdev - gref).norm() / gref.norm(); assert rel < 0.05, rel; print("ok", cls.__name__, f"weight grad rel err {rel:.4f}")
print("RNN OK")
