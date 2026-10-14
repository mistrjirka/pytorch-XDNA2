import argparse,copy,json,warnings
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import xdna_train
from xdna_train import regions
from network_matrix import Mobile, cases, metrics

def encode(t):
 t=t.detach().cpu().resolve_conj().resolve_neg()
 return torch.view_as_real(t).tolist() if t.is_complex() else t.tolist()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 torch.set_num_threads(4);xdna_train.register_xdna_device();torch.manual_seed(622);results=[]
 def put(name,fn):
  try:r=fn()
  except Exception as e:r={'error':str(e),'type':type(e).__name__}
  results.append({'case':name,**r});print(name,r,flush=True);Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
 def complex_probe():
  out={}
  for dev in ['cpu','xdna']:
   z=torch.tensor([1+2j,3-4j],dtype=torch.complex64).to(dev)
   x=torch.tensor([[1+2j]],dtype=torch.complex64).to(dev).requires_grad_();w=torch.tensor([[3+4j]],dtype=torch.complex64).to(dev).requires_grad_()
   v=x@w;v.real.sum().backward()
   out[dev]={'conj_to_cpu':encode(z.conj()),'conj_resolved_to_cpu':encode(z.conj().resolve_conj()),
    'forward':encode(v),'dx':encode(x.grad),'dw':encode(w.grad),'hermitian_product':encode(z.conj().reshape(1,2)@z.reshape(2,1))}
  return out
 # Complex conjugate materialization caused a native crash; run separate minimal cases.
 def negative_view():
  x=torch.tensor([1.,-2.]).to('xdna');return {'negview_cpu':torch._neg_view(x).cpu().tolist(),'expected':[-1.,2.]}
 # lazy_negative_view crashed in the initial isolated process; tracked separately.
 def bn_negative():
  torch.manual_seed(712);c=nn.BatchNorm2d(4).bfloat16()
  with torch.no_grad():c.weight.copy_(torch.tensor([-1.,0.,.5,-.3]));c.bias.copy_(torch.tensor([.2,-.1,0.,.3]))
  e=copy.deepcopy(c).to('xdna');d=copy.deepcopy(c).to('xdna');act=nn.LeakyReLU(.2);regions._NORM_ACT_MIN_BYTES=0;regions.fuse_norm_activation(d,act)
  x=torch.randn(3,4,11,13).bfloat16();g=torch.randn_like(x)
  xc=x.clone().requires_grad_();xe=x.to('xdna').detach().requires_grad_();xd=x.to('xdna').detach().requires_grad_()
  yc=F.leaky_relu(c(xc),.2);ye=F.leaky_relu(e(xe),.2);yd=act(d(xd))
  yc.backward(g);ye.backward(g.to('xdna'));yd.backward(g.to('xdna'))
  return {'fused_output_vs_cpu':metrics(yd,yc),'fused_dx_vs_cpu':metrics(xd.grad,xc.grad),
          'eager_dx_vs_cpu':metrics(xe.grad,xc.grad),'fused_dw_vs_cpu':metrics(d.weight.grad,c.weight.grad),
          'fused_db_vs_cpu':metrics(d.bias.grad,c.bias.grad)}
 put('muse_negative_BN_affine_hypothesis',bn_negative)
 def pool_nan():
  C=xdna_train.device._C;rs=[]
  for width in [2,4,10,36]:
   inp=torch.ones((2,3,4,width),dtype=torch.bfloat16);inp[:,:,0,0]=float('nan');inp[:,:,0,1]=float('nan');inp[:,:,1,0]=float('nan')
   inp[:,:,2,0]=-0.;inp[:,:,2,1]=0.
   rr=torch.empty_like(inp,device='xdna');pp=torch.empty((2,3,2,width//2),dtype=torch.bfloat16,device='xdna')
   C.region_relu_maxpool2(inp.to('xdna'),rr,pp)
   cpu=inp.relu();px=cpu.clone().requires_grad_();py=F.max_pool2d(px,2,2);py.backward(torch.ones_like(py))
   grad=torch.empty_like(inp,device='xdna');C.region_maxpool2_bwd(cpu.to('xdna'),torch.ones_like(pp),grad)
   rs.append({'width':width,'nan_masks_match':bool(torch.equal(torch.isnan(pp.cpu()),torch.isnan(py))),
              'backward_bit_exact':bool(torch.equal(grad.cpu().view(torch.int16),px.grad.view(torch.int16)))})
  return {'results':rs}
 put('muse_multiple_nan_pool_hypothesis',pool_nan)
 def mobile_reference():
  torch.manual_seed(622);c=cases()['mobile_depthwise_se'];base=c['model'].bfloat16();x0=c['inputs'][0].bfloat16();res={}
  for label,device,dtype in [('cpu_bf16','cpu',torch.bfloat16),('xdna_bf16','xdna',torch.bfloat16),('cpu_fp64','cpu',torch.float64)]:
   m=copy.deepcopy(base).to(device=device,dtype=dtype);x=x0.to(device=device,dtype=dtype).detach().requires_grad_()
   y=m(x);g=torch.linspace(-.7,.9,y.numel()).reshape(y.shape).bfloat16().to(device=device,dtype=dtype)
   y.backward(g);res[label]={'out':y.detach().cpu(),'dx':x.grad.cpu(),'grad':{n:p.grad.cpu() for n,p in m.named_parameters()}}
  output={}
  for label in ['cpu_bf16','xdna_bf16']:
   output[label]={'output_vs_fp64':metrics(res[label]['out'],res['cpu_fp64']['out']),
    'dx_vs_fp64':metrics(res[label]['dx'],res['cpu_fp64']['dx']),
    'gradients':{n:{'comparison':metrics(g,res['cpu_fp64']['grad'][n]),'abs_norm':float(g.float().norm()),'fp64_norm':float(res['cpu_fp64']['grad'][n].norm())} for n,g in res[label]['grad'].items()}}
  return output
 put('mobile_full_fp64_reference',mobile_reference)
 print('DONE',len(results),flush=True)
if __name__=='__main__':main()
