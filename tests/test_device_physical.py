from __future__ import annotations

import os

import pytest
import torch

import xdna_train


pytestmark = pytest.mark.skipif(
    os.environ.get("XDNA_TRAIN_PHYSICAL") != "1",
    reason="set XDNA_TRAIN_PHYSICAL=1 to run physical XDNA2 tests",
)


def test_xrt_backed_roundtrip() -> None:
    x_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    x = x_cpu.to("xdna")
    assert x.device.type == "xdna"
    assert xdna_train.is_xrt_backed(x)
    assert xdna_train.bo_handle(x) != 0
    torch.testing.assert_close(x.cpu(), x_cpu)


def test_rectangular_bf16_gemm_matches_cpu() -> None:
    torch.manual_seed(2026)
    a_cpu = (torch.randn(256, 768) * 0.03).bfloat16()
    b_cpu = (torch.randn(768, 3072) * 0.03).bfloat16()
    out = a_cpu.to("xdna") @ b_cpu.to("xdna")
    assert out.device.type == "xdna"
    assert xdna_train.is_xrt_backed(out)
    expected = (a_cpu.float() @ b_cpu.float()).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.01, atol=0.002)


def test_autograd_keeps_gradients_on_xdna() -> None:
    torch.manual_seed(20261003)
    a = (torch.randn(256, 768) * 0.03).bfloat16().to("xdna").requires_grad_(True)
    b = (torch.randn(768, 3072) * 0.03).bfloat16().to("xdna").requires_grad_(True)
    (a @ b).float().square().mean().backward()
    assert a.grad is not None and a.grad.device.type == "xdna"
    assert b.grad is not None and b.grad.device.type == "xdna"


def test_allocator_reuses_storage() -> None:
    xdna_train.empty_cache()
    before = xdna_train.allocator_stats()
    x = torch.empty((256, 768), dtype=torch.bfloat16, device="xdna")
    del x
    first = xdna_train.allocator_stats()
    y = torch.empty((256, 768), dtype=torch.bfloat16, device="xdna")
    del y
    second = xdna_train.allocator_stats()
    assert first["fresh_allocations"] >= before["fresh_allocations"] + 1
    assert second["reuses"] >= first["reuses"] + 1


def test_batch4_cpu_dw_chunking_matches_unsplit_reference(monkeypatch) -> None:
    torch.manual_seed(20261004)
    batch, cin, cout, height, width = 4, 128, 128, 146, 141
    x_cpu = (
        torch.randn(batch, cin, height, width) * 0.01
    ).bfloat16().contiguous(memory_format=torch.channels_last)
    go_cpu = (
        torch.randn(batch, cout, height, width) * 0.01
    ).bfloat16().contiguous(memory_format=torch.channels_last)
    w_cpu = (torch.randn(cout, cin, 3, 3) * 0.02).bfloat16()

    x = x_cpu.to('xdna')
    go = go_cpu.to('xdna')
    w = w_cpu.to('xdna')

    def backward_dw() -> torch.Tensor:
        _, dw, _ = torch.ops.aten.convolution_backward_overrideable(
            go, x, w, [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
            [True, True, False],
        )
        torch.xdna.synchronize()
        return dw.cpu()

    monkeypatch.setenv('XDNA_CPU_DW_BATCH_CHUNK', '0')
    unsplit = backward_dw()
    monkeypatch.setenv('XDNA_CPU_DW_BATCH_CHUNK', '1')
    split = backward_dw()

    torch.testing.assert_close(split, unsplit, rtol=0.02, atol=0.002)


def test_add_out_matches_cpu() -> None:
    torch.manual_seed(20261004)
    a_cpu = (torch.randn(4, 128, 146, 141) * 0.01).bfloat16().contiguous(
        memory_format=torch.channels_last
    )
    b_cpu = (torch.randn(4, 128, 146, 141) * 0.01).bfloat16().contiguous(
        memory_format=torch.channels_last
    )
    a = a_cpu.to('xdna')
    b = b_cpu.to('xdna')
    out = torch.empty_like(a)
    torch.add(a, b, alpha=0.5, out=out)
    torch.xdna.synchronize()
    expected = torch.add(a_cpu, b_cpu, alpha=0.5)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
