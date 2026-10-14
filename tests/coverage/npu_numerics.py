"""Physical packed-convolution output, dX, dW and non-finite behavior probes."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
import xdna_train

def error(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu()
 return {'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30)),'max_abs':float((a-b).abs().max()),'finite':bool(torch.isfinite(a).all())}
def counts():return {k:v['calls'] for k,v in xdna_train.mapped_cpu_stats().items() if k.startswith('npu') and k.endswith('.run')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 torch.set_num_threads(4);xdna_train.register_xdna_device();xdna_train.set_npu_mode(2);r=[]
 for i,(B,Ci,H,W,Co) in enumerate([(2,64,8,8,128),(2,65,7,11,129),(2,128,7,65,67)]):
  torch.manual_seed(280+i);x=(torch.randn(B,Ci,H,W)*.1).bfloat16();w=(torch.randn(Co,Ci,3,3)*.02).bfloat16();g=(torch.randn(B,Co,H,W)*.1).bfloat16()
  xr=x.float().requires_grad_();wr=w.float().requires_grad_();yr=F.conv2d(xr,wr,padding=1);yr.backward(g.float())
  xd=x.to('xdna').requires_grad_();wd=w.to('xdna').requires_grad_();xdna_train.reset_mapped_cpu_stats()
  try:
   yd=F.conv2d(xd,wd,padding=1);yd.backward(g.to('xdna'));gx=xd.grad.cpu();gw=wd.grad.cpu()
   q={'case':'finite_shape','shape':[B,Ci,H,W,Co],'output':error(yd,yr),'dx':error(gx,xr.grad),'dw':error(gw,wr.grad),'npu_runs':counts()}
  except Exception as e:q={'case':'finite_shape','shape':[B,Ci,H,W,Co],'error':str(e)}
  r.append(q);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(q,flush=True)
 # Range checks, same supported g1 stream. These are intentional abnormal numerical inputs.
 for special in ['nan','inf','negative_inf','zero','outlier']:
  torch.manual_seed(815);x=(torch.randn(2,64,8,8)*.1).bfloat16();w=(torch.randn(128,64,3,3)*.02).bfloat16()
  if special=='zero':x.zero_()
  elif special=='outlier':x[0,0,3,3]=10000.
  else:x[0,0,3,3]={'nan':float('nan'),'inf':float('inf'),'negative_inf':float('-inf')}[special]
  xdna_train.reset_mapped_cpu_stats()
  try:
   with torch.no_grad():ref=F.conv2d(x.float(),w.float(),padding=1);actual=F.conv2d(x.to('xdna'),w.to('xdna'),padding=1).cpu().float()
   q={'case':'special_'+special,'cpu_nonfinite':int((~torch.isfinite(ref)).sum()),'xdna_nonfinite':int((~torch.isfinite(actual)).sum()),
      'nonfinite_pattern_matches':bool(torch.equal(torch.isfinite(ref),torch.isfinite(actual))),'npu_runs':counts()}
   if special in ['zero','outlier']:q['error']=error(actual,ref)
  except Exception as e:q={'case':'special_'+special,'exception':str(e),'npu_runs':counts()}
  r.append(q);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(q,flush=True)
if __name__=='__main__':main()
