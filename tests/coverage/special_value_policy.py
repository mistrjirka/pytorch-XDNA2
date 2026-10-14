"""Separate non-finite values from ordinary finite numerical error."""
import argparse,copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import xdna_train
from xdna_train import regions
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);xdna_train.register_xdna_device();xdna_train.set_npu_mode(2)
r=[]
def kinds(t):
 t=t.detach().cpu();return {'nan':int(torch.isnan(t).sum()),'positive_inf':int(torch.isposinf(t).sum()),'negative_inf':int(torch.isneginf(t).sum()),'finite':int(torch.isfinite(t).sum())}
for label,number in [('nan',float('nan')),('positive_inf',float('inf')),('negative_inf',float('-inf'))]:
 torch.manual_seed(817);x=(torch.randn(2,64,8,8)*.1).bfloat16();w=(torch.randn(128,64,3,3)*.02).bfloat16();x[0,0,3,3]=number
 xdna_train.reset_mapped_cpu_stats()
 with torch.no_grad():ref=F.conv2d(x.float(),w.float(),padding=1);actual=F.conv2d(x.to('xdna'),w.to('xdna'),padding=1).cpu().float()
 q={'case':'NPU_activation_'+label,'cpu':kinds(ref),'xdna':kinds(actual),'exact_special_class':bool(torch.equal(torch.isnan(ref),torch.isnan(actual)) and torch.equal(torch.isposinf(ref),torch.isposinf(actual)) and torch.equal(torch.isneginf(ref),torch.isneginf(actual))),
  'npu_runs':{k:v['calls'] for k,v in xdna_train.mapped_cpu_stats().items() if k.startswith('npu') and k.endswith('.run')}}
 r.append(q);print(q,flush=True)
regions._NORM_ACT_MIN_BYTES=0
for slope in [0.,.2]:
 torch.manual_seed(13);base=nn.BatchNorm2d(4).bfloat16();base.bias.data.fill_(float('nan'));x=torch.randn(2,4,4,4).bfloat16();g=torch.randn_like(x);outs={}
 for mode in ['cpu','eager_xdna','fused_xdna']:
  dev='cpu' if mode=='cpu' else 'xdna';b=copy.deepcopy(base).to(dev);act=nn.ReLU() if slope==0 else nn.LeakyReLU(slope)
  if mode=='fused_xdna':regions.fuse_norm_activation(b,act)
  z=x.clone().to(dev).requires_grad_();y=act(b(z));y.backward(g.to(dev));outs[mode]={'output':kinds(y),'db':b.bias.grad.detach().cpu().float().tolist(),'dxnorm':float(z.grad.detach().cpu().float().norm())}
 q={'case':'BN_nan_affine_bias','slope':slope,'results':outs};r.append(q);print(q,flush=True)
Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
