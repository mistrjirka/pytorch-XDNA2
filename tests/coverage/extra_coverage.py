"""Additional shape and domain probes; all results retain limitations."""
import argparse,copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import xdna_train
from network_matrix import Fn,run_case,metrics
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(929);xdna_train.register_xdna_device();xdna_train.set_npu_mode(2);rs=[]
def save(r):rs.append(r);Path(a.out).write_text(json.dumps(rs,indent=2)+'\n');print(r['case'],r.get('status'),str(r.get('errors',r.get('metrics','')))[:500],flush=True)
items=[
 ('conv3d_without_pool',nn.Sequential(nn.Conv3d(2,4,3,padding=1),nn.BatchNorm3d(4),nn.ReLU()),[torch.randn(2,2,4,6,6)*.2],['bf16','fp32']),
 ('instancenorm3d',nn.InstanceNorm3d(4,affine=True),[torch.randn(2,4,4,6,6)*.2],['bf16','fp32']),
 ('ctc_loss',Fn(lambda x: F.ctc_loss(x.log_softmax(2),torch.tensor([[1,2,3],[2,2,1]],device=x.device),torch.tensor([9,8]),torch.tensor([3,3]),blank=0,zero_infinity=True)),[torch.randn(9,2,5)*.3],['fp32']),
]
try:
 from torchvision.ops import roi_align
 boxes=torch.tensor([[0.,1.,1.,7.,6.],[1.,0.,0.,6.,7.]])
 items.append(('roi_align_detection',Fn(lambda x,b:roi_align(x,b,(3,3),sampling_ratio=2,aligned=True)),[torch.randn(2,4,8,8)*.2,boxes],['fp32']))
except Exception as e:save({'case':'roi_align_detection','status':'dependency_unavailable','error':str(e)})
for name,m,x,dtypes in items:
 for dt in dtypes:
  r=run_case({'model':m,'inputs':x},dt);r.update(case=name,dtype=dt);save(r)
# One supported linear projection maps its forward, dX and dW to all3 fixed GEMMs.
x=(torch.randn(256,768)*.03).bfloat16();w=(torch.randn(3072,768)*.03).bfloat16();g=(torch.randn(256,3072)*.03).bfloat16()
xr=x.float().requires_grad_();wr=w.float().requires_grad_();yr=F.linear(xr,wr);yr.backward(g.float())
xd=x.to('xdna').requires_grad_();wd=w.to('xdna').requires_grad_();xdna_train.reset_mapped_cpu_stats()
try:
 yd=F.linear(xd,wd);yd.backward(g.to('xdna'));vals={'output':metrics(yd,yr),'dx':metrics(xd.grad,xr.grad),'dw':metrics(wd.grad,wr.grad)}
 calls=xdna_train.mapped_cpu_stats().get('aten::mm.cpu_mapped',{}).get('calls',0)
 save({'case':'three_fixed_gemm_shapes_linear_training','status':'pass' if all(v['relative_l2']<.01 for v in vals.values()) and calls==0 else 'check','metrics':vals,'cpu_mm_fallback_calls':calls,'gemm_family_configured':xdna_train.device._C.gemm_family_configured()})
except Exception as e:save({'case':'three_fixed_gemm_shapes_linear_training','status':'error','error':str(e)})
