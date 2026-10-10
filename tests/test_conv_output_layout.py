"""Numerical and layout regression tests for opt-in patch Conv NCHW output."""
from __future__ import annotations

import os

import pytest
import torch
import xdna_train

pytestmark=pytest.mark.skipif(
    os.environ.get("XDNA_TRAIN_PHYSICAL")!="1",
    reason="run with XDNA_TRAIN_PHYSICAL=1 on XDNA2 hardware",
)


@pytest.mark.parametrize("height",[18,36])
def test_nchw_output_conv_bn_matches_cpu_and_original_geometry(
    monkeypatch: pytest.MonkeyPatch,height:int
) -> None:
    torch.manual_seed(927)
    torch.set_num_threads(10)
    batch,cin,cout=40,256,128
    source=torch.nn.Conv2d(cin,cout,3,padding=1,bias=False).bfloat16()
    reference_bn=torch.nn.BatchNorm2d(cout).bfloat16()
    conv= torch.nn.Conv2d(cin,cout,3,padding=1,bias=False).bfloat16()
    conv.load_state_dict(source.state_dict())
    bn=torch.nn.BatchNorm2d(cout).bfloat16()
    bn.load_state_dict(reference_bn.state_dict())

    generator=torch.Generator().manual_seed(819)
    x_cpu=(torch.randn(batch,cin,height,height,generator=generator)*0.1).bfloat16()
    dy_cpu=(torch.randn(batch,cout,height,height,generator=generator)*0.1).bfloat16()
    x_cpu=x_cpu.detach().requires_grad_(True)
    reference=reference_bn(source(x_cpu))
    reference.backward(dy_cpu)

    conv=conv.to(device="xdna")
    bn=bn.to(device="xdna")
    x=x_cpu.detach().to("xdna").requires_grad_(True)
    dy=dy_cpu.to("xdna")

    # Compare the exact same physical NPU conv with and without the transpose.
    monkeypatch.setenv("XDNA_CONV_OUTPUT_NCHW","0")
    raw=conv(x)
    assert raw.shape==(batch,cout,height,height)
    assert not raw.is_contiguous()
    monkeypatch.setenv("XDNA_CONV_OUTPUT_NCHW","1")
    converted=conv(x)
    assert converted.shape==raw.shape
    assert converted.is_contiguous()
    torch.testing.assert_close(converted.cpu(),raw.cpu(),rtol=0,atol=0)

    out=bn(converted)
    out.backward(dy)
    torch.testing.assert_close(out.cpu(),reference,rtol=0.01,atol=0.02)
    torch.testing.assert_close(conv.weight.grad.cpu(),source.weight.grad,rtol=0.02,atol=0.02)
    torch.testing.assert_close(x.grad.cpu(),x_cpu.grad,rtol=0.02,atol=0.02)
