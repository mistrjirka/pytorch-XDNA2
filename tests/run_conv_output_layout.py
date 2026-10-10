#!/usr/bin/env python3
"""Hardware-only smoke test for XDNA Conv output layout, no pytest dependency."""
import os,torch,xdna_train
torch.manual_seed(927)
torch.set_num_threads(10)
for h in [18,36]:
 b,ci,co=40,256,128
 conv_cpu=torch.nn.Conv2d(ci,co,3,padding=1,bias=False,dtype=torch.bfloat16)
 bn_cpu=torch.nn.BatchNorm2d(co,dtype=torch.bfloat16)
 conv_x=torch.nn.Conv2d(ci,co,3,padding=1,bias=False,dtype=torch.bfloat16)
 conv_x.load_state_dict(conv_cpu.state_dict())
 bn_x=torch.nn.BatchNorm2d(co,dtype=torch.bfloat16)
 bn_x.load_state_dict(bn_cpu.state_dict())
 gen=torch.Generator().manual_seed(819)
 x_cpu=(torch.randn(b,ci,h,h,generator=gen)*0.1).bfloat16().requires_grad_(True)
 dy=(torch.randn(b,co,h,h,generator=gen)*0.1).bfloat16()
 ref=bn_cpu(conv_cpu(x_cpu));ref.backward(dy)
 conv_x.to("xdna");bn_x.to("xdna")
 x=x_cpu.detach().to("xdna").requires_grad_(True)
 dyx=dy.to("xdna")
 os.environ["XDNA_CONV_OUTPUT_NCHW"]="0"
 old=conv_x(x)
 assert not old.is_contiguous(), old.stride()
 os.environ["XDNA_CONV_OUTPUT_NCHW"]="1"
 result=conv_x(x)
 assert result.is_contiguous(),result.stride()
 torch.testing.assert_close(result.cpu(),old.cpu(),rtol=0,atol=0)
 out=bn_x(result)
 out.backward(dyx)
 for key,actual,expected in [
   ("batchnorm_output",out.cpu(),ref),
   ("conv_weight_gradient",conv_x.weight.grad.cpu(),conv_cpu.weight.grad),
   ("conv_input_gradient",x.grad.cpu(),x_cpu.grad),
 ]:
  torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)
  print(f"PASS h={h} {key}; max_abs_diff={(actual.float()-expected.float()).abs().max().item():.6f}",flush=True)
 print("PASS all h",h,flush=True)
