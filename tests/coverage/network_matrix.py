"""Release probes: small network motifs, CPU-reference vs XDNA forward/backward.

This is coverage evidence, not a benchmark of full production networks.
Use --case to isolate a failure. No model files or installed packages are changed.
"""
from __future__ import annotations
import argparse, copy, json, math, os, traceback, warnings
from pathlib import Path
from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F

class Fn(nn.Module):
    def __init__(self,fn):super().__init__();self.fn=fn
    def forward(self,*a):return self.fn(*a)
class Mobile(nn.Module):
    def __init__(self):
        super().__init__();self.depth=nn.Conv2d(8,8,3,padding=1,groups=8);self.point=nn.Conv2d(8,16,1);self.bn=nn.BatchNorm2d(16)
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(16,4,1),nn.ReLU(),nn.Conv2d(4,16,1),nn.Sigmoid())
    def forward(self,x):x=F.silu(self.bn(self.point(self.depth(x))));return x*self.se(x)
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.GroupNorm(4,8);self.conv=nn.Conv2d(8,8,3,padding=1);self.time=nn.Linear(6,8)
    def forward(self,x,t):return x+self.conv(F.silu(self.norm(x))+self.time(t)[:,:,None,None])
class TinyViT(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(3,16,4,stride=4);self.encoder=nn.TransformerEncoderLayer(16,4,32,dropout=0.,batch_first=True,norm_first=True);self.head=nn.Linear(16,3)
    def forward(self,x):x=self.patch(x).flatten(2).transpose(1,2);return self.head(self.encoder(x).mean(1))
class Recur(nn.Module):
    def __init__(self,kind):
        super().__init__();self.net=getattr(nn,kind)(6,8,2,batch_first=True,bidirectional=True,dropout=0.)
    def forward(self,x):return self.net(x)[0]
class Graph(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Linear(8,8)
    def forward(self,x,src,dst):
        msgs=self.proj(x.index_select(0,src));out=torch.zeros_like(x);return out.index_add(0,dst,msgs)+x
class Attention(nn.Module):
    def __init__(self,variant):super().__init__();self.variant=variant
    def forward(self,q,k,v,mask=None):return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=self.variant=='causal',dropout_p=0.,enable_gqa=self.variant=='gqa')
class Spectral(nn.Module):
    def forward(self,x):return torch.fft.irfft2(torch.fft.rfft2(x)*.75,s=x.shape[-2:])
class EmbedBag(nn.Module):
    def __init__(self,mode):super().__init__();self.bag=nn.EmbeddingBag(23,8,mode=mode,padding_idx=0)
    def forward(self,ids,offs):return self.bag(ids,offs)
class DenseCheck(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,4))
    def forward(self,x):return self.net(x)

def cases():
    R=lambda *s:torch.randn(*s)*.3
    c={}
    def add(n,m,x,family,dtypes=('bf16','fp32')):c[n]={'model':m,'inputs':x,'family':family,'dtypes':dtypes}
    add('mlp_gelu',DenseCheck(),[R(4,8)],'MLPs / feed-forward networks')
    add('conv1d_temporal',nn.Sequential(nn.Conv1d(4,8,3,padding=2,dilation=2),nn.BatchNorm1d(8),nn.GELU()),[R(2,4,17)],'TCN / audio / time-series')
    add('conv3d_volume',nn.Sequential(nn.Conv3d(2,4,3,padding=1),nn.BatchNorm3d(4),nn.ReLU(),nn.AvgPool3d(2)),[R(2,2,6,8,8)],'3D U-Net / volumetric and video CNNs')
    add('transpose2d_decoder',nn.Sequential(nn.ConvTranspose2d(4,6,3,stride=2,padding=1,output_padding=1),nn.InstanceNorm2d(6,affine=True),nn.LeakyReLU(.2)),[R(2,4,7,9)],'GAN / autoencoder / upsampling decoders')
    add('transpose3d_decoder',nn.ConvTranspose3d(2,3,2,stride=2),[R(1,2,3,4,5)],'Volumetric decoder')
    add('grouped_dilated_conv',nn.Conv2d(8,12,3,padding=2,dilation=2,groups=4),[R(2,8,13,11)],'Grouped and atrous CNNs')
    add('mobile_depthwise_se',Mobile(),[R(2,8,12,10)],'MobileNet-like depthwise and squeeze/excitation')
    add('diffusion_norm_time',Diffusion(),[R(2,8,8,10),R(2,6)],'Diffusion U-Net block')
    add('vit_encoder',TinyViT(),[R(2,3,16,16)],'Vision Transformer encoder')
    add('layer_norm',nn.LayerNorm(16),[R(2,7,16)],'Transformers / sequence models')
    add('rms_norm',nn.RMSNorm(16,eps=1e-5),[R(2,7,16)],'Decoder LLMs / RMSNorm networks')
    for kind in ('LSTM','GRU'):
        add(kind.lower()+'_bidirectional',Recur(kind),[R(2,9,6)],'Recurrent sequence model')
    for affine in (True,False):
        add('batchnorm_affine_'+str(affine),nn.BatchNorm2d(6,affine=affine),[R(3,6,8,7)],'CNN training normalization')
        add('instancenorm1d_affine_'+str(affine),nn.InstanceNorm1d(6,affine=affine),[R(2,6,13)],'Style transfer / sequence normalization')
        add('instancenorm2d_affine_'+str(affine),nn.InstanceNorm2d(6,affine=affine),[R(2,6,8,7)],'Style transfer / image generation')
    add('batchnorm_channels_last',nn.BatchNorm2d(6),[R(2,6,9,7).contiguous(memory_format=torch.channels_last)],'Channels-last CNN training')
    add('batchnorm_transposed',nn.BatchNorm2d(6),[R(2,7,9,6).transpose(1,3)],'Permuted-view normalization')
    for variant in ('causal','bool_mask','float_mask','fully_masked','gqa'):
        q=R(2,4,7,8);k=R(2,2 if variant=='gqa' else 4,7,8);v=k.clone()*1.1
        inp=[q,k,v]
        if variant=='bool_mask':inp+=[torch.ones(7,7,dtype=torch.bool).tril()]
        if variant=='float_mask':inp+=[torch.where(torch.ones(7,7,dtype=torch.bool).tril(),0.,float('-inf'))]
        if variant=='fully_masked':
            mask=torch.ones(7,7,dtype=torch.bool);mask[0]=False;inp+=[mask]
        add('sdpa_'+variant,Attention(variant),inp,'Attention / causal LM / masked sequence models')
    add('embedding_dense',nn.Embedding(23,8,padding_idx=0,max_norm=1.5),[torch.tensor([[1,3,3,0],[2,5,9,0]])],'Language/recommendation embeddings')
    add('embedding_sparse',nn.Embedding(23,8,sparse=True),[torch.tensor([[1,3,3,0],[2,5,9,0]])],'Sparse recommendation embeddings')
    for mode in ('mean','sum','max'):
        add('embeddingbag_'+mode,EmbedBag(mode),[torch.tensor([1,2,3,0,5,5,7]),torch.tensor([0,3,5])],'Recommendation / bag-of-words')
    add('gnn_index_add',Graph(),[R(9,8),torch.tensor([0,1,1,3,5,8]),torch.tensor([1,2,2,5,7,0])],'Graph message passing / scatter aggregation')
    add('sparse_graph_mm',Fn(lambda a,x:torch.sparse.mm(a.to_sparse(),x)),[torch.eye(7)+torch.eye(7).roll(1,0),R(7,4)],'Sparse graph convolution')
    add('spectral_fft',Spectral(),[R(2,3,8,8)],'Fourier neural operators / spectral audio',('fp32',))
    add('complex_linear',nn.Linear(5,4,dtype=torch.complex64),[torch.complex(R(2,5),R(2,5))],'Complex-valued networks',('complex64',))
    add('grid_sample_warp',Fn(lambda x,g:F.grid_sample(x,g,mode='bilinear',padding_mode='border',align_corners=False)),[R(2,3,8,8),torch.tanh(R(2,5,6,2))],'Spatial transformers / optical flow',('fp32',))
    add('resize_bilinear',Fn(lambda x:F.interpolate(x,size=(13,11),mode='bilinear',align_corners=False)),[R(2,4,7,9)],'Segmentation / vision resize')
    add('pixel_shuffle',nn.PixelShuffle(2),[R(2,12,5,7)],'Superresolution')
    add('unfold_fold',Fn(lambda x:F.fold(F.unfold(x,3,padding=1),(7,9),3,padding=1)),[R(2,3,7,9)],'Patch extraction / local attention')
    add('linalg_cholesky',Fn(lambda a:torch.linalg.cholesky(a@a.transpose(-2,-1)+torch.eye(5,device=a.device)*.5)),[R(2,5,5)],'Covariance / Gaussian / differentiable physics',('fp32',))
    add('cumsum_state_scan',Fn(lambda x:torch.cumsum(torch.sigmoid(x),dim=1)),[R(2,13,8)],'Simple scan primitive for state-space models')
    return c

def metrics(a,b):
    a=a.detach().cpu();b=b.detach().cpu()
    if a.is_sparse:a=a.to_dense()
    if b.is_sparse:b=b.to_dense()
    if a.shape!=b.shape:return {'pass':False,'shape_a':list(a.shape),'shape_b':list(b.shape)}
    if a.is_complex():a=a.to(torch.complex128);b=b.to(torch.complex128)
    else:a=a.double();b=b.double()
    finite=bool(torch.equal(torch.isfinite(a),torch.isfinite(b)) and torch.equal(torch.isnan(a),torch.isnan(b)))
    d=(a-b).abs();mask=torch.isfinite(d)
    mx=float(d[mask].max()) if mask.any() else 0.
    rel=float(torch.linalg.vector_norm(torch.nan_to_num(a-b))/torch.linalg.vector_norm(torch.nan_to_num(b)).clamp_min(1e-12))
    return {'relative_l2':rel,'max_abs':mx,'finite_pattern_match':finite}

def run_case(case,dtype):
    dt={'bf16':torch.bfloat16,'fp32':torch.float32,'complex64':torch.complex64}[dtype]
    model=copy.deepcopy(case['model']).to(dtype=dt)
    original=[]
    for x in case['inputs']:
        original.append(x.to(dt) if x.is_floating_point() or x.is_complex() else x)
    outs={};err={}
    import xdna_train
    for device in ('cpu','xdna'):
        stage='construct'
        try:
            m=copy.deepcopy(model).to(device)
            xs=[x.clone(memory_format=torch.preserve_format).to(device).detach().requires_grad_(x.is_floating_point() or x.is_complex()) for x in original]
            stage='forward';y=m(*xs)
            if isinstance(y,tuple):y=y[0]
            g=torch.linspace(-.7,.9,y.numel(),dtype=torch.float32).reshape(y.shape).to(y.dtype).to(device)
            stage='backward';y.backward(g)
            stage='readback'
            outs[device]={'out':y.detach().cpu(), 'input_grads':[x.grad.detach().cpu() if x.grad is not None else None for x in xs],
                'parameter_grads':{n:p.grad.detach().cpu() if p.grad is not None else None for n,p in m.named_parameters()},
                'buffers':{n:b.detach().cpu() for n,b in m.named_buffers()}}
        except Exception as e:err[device]={'stage':stage,'type':type(e).__name__,'message':str(e)[:1300]}
    if 'cpu' in err:return {'status':'reference_unsupported','errors':err}
    if 'xdna' in err:return {'status':'xdna_error','errors':err}
    allmetrics={};a,b=outs['xdna'],outs['cpu'];allmetrics['output']=metrics(a['out'],b['out'])
    for i,(ga,gb) in enumerate(zip(a['input_grads'],b['input_grads'])):
        if gb is None and ga is None:continue
        allmetrics['dx'+str(i)]=metrics(ga,gb) if ga is not None and gb is not None else {'missing_grad':True}
    for key in ('parameter_grads','buffers'):
        for n,bb in b[key].items():
            aa=a[key].get(n)
            if bb is None and aa is None:continue
            allmetrics[key+'/'+n]=metrics(aa,bb) if aa is not None and bb is not None else {'missing_grad':True}
    tol=.05 if dtype=='bf16' else .0003;floor=.005 if dtype=='bf16' else .00003
    bad={k:v for k,v in allmetrics.items() if v.get('missing_grad') or v.get('pass') is False or not v.get('finite_pattern_match',True) or (v.get('relative_l2',0)>tol and v.get('max_abs',0)>floor)}
    return {'status':'numeric_mismatch' if bad else 'pass','gate':{'relative_l2':tol,'absolute_floor':floor},'metrics':allmetrics,'failed':bad}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--case');a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(622)
    import xdna_train
    xdna_train.register_xdna_device();xdna_train.set_npu_mode(2)
    result=[]
    for name,c in cases().items():
        if a.case and name!=a.case:continue
        for dt in c['dtypes']:
            xdna_train.reset_mapped_cpu_stats()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                try:r=run_case(c,dt)
                except Exception as e:r={'status':'harness_error','error':str(e),'traceback':traceback.format_exc()}
            r.update(case=name,dtype=dt,family=c['family'],warnings=list(dict.fromkeys(str(w.message) for w in ws)),
                     npu_runs={k:v['calls'] for k,v in xdna_train.mapped_cpu_stats().items() if k.startswith('npu') and k.endswith('.run')})
            result.append(r);Path(a.out).write_text(json.dumps({'torch':torch.__version__,'backend':xdna_train.__file__,'scope':'small motifs, no full network accuracy or throughput claims','results':result},indent=2)+'\n')
            print(name,dt,r['status'],str(r.get('errors',r.get('failed',{})))[:350],flush=True)
    from collections import Counter
    print('SUMMARY',dict(Counter(r['status'] for r in result)),flush=True)
if __name__=='__main__':main()
