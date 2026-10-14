import argparse,json
from pathlib import Path
import torch
import xdna_train
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);xdna_train.register_xdna_device();xdna_train.set_npu_mode(2)
rs=[]
for i,(m,k,n) in enumerate([(256,768,3072),(256,3072,768),(3072,256,768)]):
 torch.manual_seed(1900+i);x=(torch.randn(m,k)*.03).bfloat16();w=(torch.randn(k,n)*.03).bfloat16();ref=x.float()@w.float()
 xdna_train.reset_mapped_cpu_stats();got=(x.to('xdna')@w.to('xdna')).cpu().float()
 calls=xdna_train.mapped_cpu_stats().get('aten::mm.cpu_mapped',{}).get('calls',0)
 r={'shape_M_K_N':[m,k,n],'relative_l2':float((got-ref).norm()/ref.norm()),'max_abs':float((got-ref).abs().max()),'cpu_mm_fallback_calls':calls,'family_configured':bool(xdna_train.device._C.gemm_family_configured())}
 rs.append(r);print(r,flush=True)
Path(a.out).write_text(json.dumps(rs,indent=2)+'\n')
