"""Numerical adversarial and API-contract tests, saved independently of backend."""
import argparse,copy,io,json,math,traceback,warnings
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import xdna_train
from xdna_train import regions

class Loss(nn.Module):
    def __init__(self,taps=(1,)):
        super().__init__();self.perceptual_vgg=nn.Module();self.perceptual_vgg.model=nn.Module()
        self.perceptual_vgg.model.features=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU())
        self.perceptual_vgg.feature_layers=list(taps);self.perceptual_vgg.use_norm=False
        self.requires_grad_(False)
    def compute_perceptual_loss(self,x,t):
        def feats(z):
            vals=[]
            for i,m in enumerate(self.perceptual_vgg.model.features):
                z=m(z)
                if i in self.perceptual_vgg.feature_layers:vals.append(z.clone().flatten(1))
            return torch.cat(vals,1)
        return (feats(x)-feats(t)).square().mean()

def metric(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double()
    return {'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-12)),'max_abs':float((a-b).abs().max()),'finite_a':bool(torch.isfinite(a).all()),'finite_b':bool(torch.isfinite(b).all())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(713);xdna_train.register_xdna_device()
    results=[]
    def emit(name,fn):
        try:r=fn()
        except Exception as e:r={'exception':type(e).__name__,'message':str(e)[:1600],'trace':traceback.format_exc()}
        results.append({'case':name,**r});Path(a.out).write_text(json.dumps(results,indent=2)+'\n');print(name,r,flush=True)
    def mutate(kind):
        m=Loss().to('xdna',torch.bfloat16);regions.fuse_perceptual_loss(m)
        x=torch.randn(2,3,8,8).bfloat16().to('xdna').requires_grad_();t=torch.randn_like(x)
        m.compute_perceptual_loss(x,t).backward();x.grad=None
        if kind=='activation':m.perceptual_vgg.model.features[1]=nn.Sigmoid()
        elif kind=='conv_stride':m.perceptual_vgg.model.features[0].stride=(2,2)
        elif kind=='maxpool':m.perceptual_vgg.model.features[1]=nn.MaxPool2d(2)
        actual=m.compute_perceptual_loss(x,t);ag=torch.autograd.grad(actual,x)[0]
        xx=x.detach().clone().requires_grad_();expected=Loss.compute_perceptual_loss(m,xx,t);eg=torch.autograd.grad(expected,xx)[0]
        return {'output':metric(actual,expected),'input_grad':metric(ag,eg),'actual':float(actual),'expected':float(expected)}
    for k in ['activation','conv_stride','maxpool']:emit('feature_cache_changed_'+k,lambda k=k:mutate(k))
    def dup_taps():
        m=Loss((0,0,1)).to('xdna',torch.bfloat16);regions.fuse_perceptual_loss(m)
        # Module's eager semantics uses membership, so list duplicates don't add weight.
        x=torch.randn(2,3,8,8).bfloat16().to('xdna').requires_grad_();t=torch.randn_like(x)
        return metric(m.compute_perceptual_loss(x,t),Loss.compute_perceptual_loss(m,x,t))
    emit('tap_membership_control',dup_taps)
    def bn_case(level,bias,slope,negative_weight=False):
        x=torch.full((2,4,7,9),level).bfloat16()
        c=nn.BatchNorm2d(4).bfloat16();c.weight.data.fill_(-1 if negative_weight else 1);c.bias.data.fill_(bias)
        e=copy.deepcopy(c).to('xdna');d=copy.deepcopy(c).to('xdna');act=nn.LeakyReLU(slope) if slope else nn.ReLU()
        regions._NORM_ACT_MIN_BYTES=0;regions.fuse_norm_activation(d,act)
        g=torch.randn_like(x);xx=x.to('xdna').requires_grad_();xe=x.to('xdna').requires_grad_();xc=x.clone().requires_grad_()
        y=act(d(xx));ye=F.leaky_relu(e(xe),slope);yc=F.leaky_relu(c(xc),slope)
        y.backward(g.to('xdna'));ye.backward(g.to('xdna'));yc.backward(g)
        return {'level':level,'bias':bias,'slope':slope,'out_vs_cpu':metric(y,yc),'out_vs_eager_xdna':metric(y,ye),
                'dx_vs_cpu':metric(xx.grad,xc.grad),'dweight_vs_cpu':metric(d.weight.grad,c.weight.grad),
                'mean_output':float(y.float().mean()),'cpu_mean_output':float(yc.float().mean())}
    for lev in [0.,1.,1024.,1048576.]:emit('constant_bn_'+str(lev),lambda lev=lev:bn_case(lev,.5,.2))
    for slope in [0.,.001,2.]:emit('bn_slope_'+str(slope),lambda slope=slope:bn_case(1.,-.5,slope,True))
    def pointloss(kind):
        x=torch.tensor([-90.,-20.,0.,20.,90.]).bfloat16();t=torch.tensor([0.,1.,0.,1.,1.]).bfloat16()
        outs={}
        for dev in ['cpu','xdna']:
            a=x.detach().clone().to(dev).requires_grad_();b=t.to(dev)
            if kind=='bce_logits':y=F.binary_cross_entropy_with_logits(a,b)
            elif kind=='mse':y=F.mse_loss(a,b)
            elif kind=='smooth_l1':y=F.smooth_l1_loss(a,b,beta=.5)
            elif kind=='softmax':y=F.log_softmax(a,0).square().mean()
            y.backward();outs[dev]=(y,a.grad)
        return {'output':metric(outs['xdna'][0],outs['cpu'][0]),'gradient':metric(outs['xdna'][1],outs['cpu'][1])}
    for n in ['bce_logits','mse','smooth_l1','softmax']:emit('loss_extreme_'+n,lambda n=n:pointloss(n))
    def repeat_hooks():
        m=Loss().to('xdna',torch.bfloat16);regions.fuse_perceptual_loss(m)
        counts={'n':0}
        def hook(mod,inp,out):counts['n']+=1;return out*.5
        handle=m.perceptual_vgg.model.features[0].register_forward_hook(hook)
        x=torch.randn(2,3,8,8).bfloat16().to('xdna').requires_grad_();t=torch.randn_like(x)
        actual=m.compute_perceptual_loss(x,t);called=counts['n'];expected=Loss.compute_perceptual_loss(m,x,t)
        handle.remove();return {'hook_calls_fused':called,'hook_calls_eager':counts['n']-called,'output':metric(actual,expected)}
    emit('feature_forward_hook_changes_output',repeat_hooks)
    def inplace_pair():
        b=nn.BatchNorm2d(4).to('xdna',torch.bfloat16);act=nn.LeakyReLU(.2,inplace=True);regions.fuse_norm_activation(b,act)
        x=torch.randn(2,4,7,9,dtype=torch.bfloat16,device='xdna',requires_grad=True)
        y=act(b(x));z=act(y)
        try:z.sum().backward();return {'second_call_backward':'success'}
        except RuntimeError as e:return {'second_call_backward':'raises','message':str(e)}
    emit('second_inplace_act_saved_output',inplace_pair)
    print('CASES',len(results),flush=True)
if __name__=='__main__':main()
