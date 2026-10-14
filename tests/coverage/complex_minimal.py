"""Public complex matmul backward: a one-element independent expected answer."""
import argparse,json
import torch
import xdna_train
p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args()
torch.set_num_threads(2);xdna_train.register_xdna_device()
x=torch.tensor([[1+2j]],dtype=torch.complex64,device=a.device,requires_grad=True)
w=torch.tensor([[3+4j]],dtype=torch.complex64,device=a.device,requires_grad=True)
y=x@w
y.backward(torch.ones_like(y))
def values(t):
 z=t.detach().cpu().tolist()[0][0]
 return [float(z.real),float(z.imag)]
r={'device':a.device,'y':values(y),'dx':values(x.grad),'dw':values(w.grad),
   'expected_y':[-5.,10.],'expected_dx':[3.,-4.],'expected_dw':[1.,-2.]}
print(json.dumps(r),flush=True)
