"""Narrow training-interface smoke tests; do not imply full compiler/distributed support."""
import argparse,copy,io,json,warnings
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import xdna_train
from network_matrix import metrics

class Small(nn.Module):
 def __init__(self):super().__init__();self.seq=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,3))
 def forward(self,x):return self.seq(x)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(83)
 xdna_train.register_xdna_device();rs=[]
 def check(name,fn):
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always')
   try:r=fn()
   except Exception as e:r={'status':'error','error':type(e).__name__,'message':str(e)[:1800]}
  r.update(case=name,warnings=list(dict.fromkeys(str(w.message) for w in ws)));rs.append(r);Path(a.out).write_text(json.dumps(rs,indent=2)+'\n');print(name,r,flush=True)
 def optim(kind):
  torch.manual_seed(514);base=Small().bfloat16();x=torch.randn(4,8).bfloat16();target=torch.randn(4,3).bfloat16();out={}
  for dev in ['cpu','xdna']:
   m=copy.deepcopy(base).to(dev);kw={'lr':.01}
   if kind=='SGD':kw.update(momentum=.9,weight_decay=.01)
   if kind=='AdamW':kw.update(weight_decay=.01,amsgrad=True)
   opt=getattr(torch.optim,kind)(m.parameters(),**kw)
   for _ in range(3):
    opt.zero_grad();loss=F.mse_loss(m(x.to(dev)).float(),target.to(dev).float());loss.backward();opt.step()
   out[dev]={n:p.detach().cpu() for n,p in m.named_parameters()}
  ms={n:metrics(p,out['cpu'][n]) for n,p in out['xdna'].items()};ok=all(v['relative_l2']<.04 or v['max_abs']<.003 for v in ms.values())
  return {'status':'pass' if ok else 'mismatch','metrics':ms}
 for k in ['SGD','Adam','AdamW','RMSprop','Adagrad']:check('optimizer_'+k,lambda k=k:optim(k))
 def cp_test():
  torch.manual_seed(971);base=Small();data=torch.randn(3,8);outs={}
  for dev in ['cpu','xdna']:
   m=copy.deepcopy(base).to(dev);x=data.clone().to(dev).requires_grad_();y=checkpoint(m,x,use_reentrant=False);y.square().sum().backward();outs[dev]=(y.detach().cpu(),x.grad.cpu(),{n:p.grad.cpu() for n,p in m.named_parameters()})
  ms={'y':metrics(outs['xdna'][0],outs['cpu'][0]),'dx':metrics(outs['xdna'][1],outs['cpu'][1])}
  ms.update({n:metrics(g,outs['cpu'][2][n]) for n,g in outs['xdna'][2].items()})
  return {'status':'pass' if all(m['max_abs']<1e-4 for m in ms.values()) else 'mismatch','metrics':ms}
 check('checkpoint_nonreentrant_dense',cp_test)
 def second():
  outs={}
  for dev in ['cpu','xdna']:
   x=torch.linspace(-.8,.7,11).to(dev).requires_grad_();g=torch.autograd.grad(x.sin().square().sum(),x,create_graph=True)[0];h=torch.autograd.grad(g.sum(),x)[0];outs[dev]=h
  m=metrics(outs['xdna'],outs['cpu']);return {'status':'pass' if m['max_abs']<1e-5 else 'mismatch','metrics':m}
 check('higher_order_pointwise',second)
 def vm():
  data=torch.randn(3,8);got=torch.vmap(lambda x:x.sin().sum())(data.to('xdna'));m=metrics(got,data.sin().sum(1));return {'status':'pass' if m['max_abs']<1e-5 else 'mismatch','metrics':m}
 check('vmap_pointwise',vm)
 def jvp():
  x=torch.randn(3,8);t=torch.randn_like(x);_,a=torch.func.jvp(lambda z:z.sin(),(x.to('xdna'),),(t.to('xdna'),));m=metrics(a,x.cos()*t);return {'status':'pass' if m['max_abs']<1e-5 else 'mismatch','metrics':m}
 check('jvp_pointwise',jvp)
 def save():
  x=torch.arange(12,dtype=torch.float32).reshape(3,4).to('xdna');b=io.BytesIO();torch.save({'x':x},b);b.seek(0);r=torch.load(b,map_location='cpu',weights_only=True)['x'];return {'status':'pass' if torch.equal(r,torch.arange(12).reshape(3,4)) else 'mismatch','loaded_device':str(r.device)}
 check('serialize_tensor_to_cpu',save)
 def packed_lstm():
  base=nn.LSTM(5,7,batch_first=True);data=torch.randn(3,6,5);out={}
  for dev in ['cpu','xdna']:
   m=copy.deepcopy(base).to(dev);x=data.clone().to(dev).requires_grad_();p=pack_padded_sequence(x,[6,4,2],batch_first=True,enforce_sorted=True);yp,_=m(p);y,_=pad_packed_sequence(yp,batch_first=True);y.square().sum().backward();out[dev]=(y.detach().cpu(),x.grad.cpu())
  ms={'y':metrics(out['xdna'][0],out['cpu'][0]),'dx':metrics(out['xdna'][1],out['cpu'][1])};return {'status':'pass' if all(m['max_abs']<1e-4 for m in ms.values()) else 'mismatch','metrics':ms}
 check('packed_sequence_lstm_fp32',packed_lstm)
 def amp():
  m=Small().to('xdna');x=torch.randn(3,8).to('xdna')
  with torch.autocast(device_type='xdna',dtype=torch.bfloat16):y=m(x);loss=y.float().square().mean()
  loss.backward();return {'status':'runs','output_dtype':str(y.dtype)}
 check('autocast_xdna',amp)
 def compile_eager():
  m=Small().to('xdna');x=torch.randn(3,8).to('xdna');f=torch.compile(m,backend='eager',fullgraph=True);y=f(x);y.square().sum().backward();return {'status':'runs','note':'Dynamo+eager backend only; not Inductor or NPU graph compilation.'}
 check('compile_frontend_eager_backend',compile_eager)
 print('DONE',len(rs),flush=True)
if __name__=='__main__':main()
