"""Independent FP32 chunk algebra, bounded-memory sequential temporal slabs.
No truncation: each slab consumes its predecessor's full FP32 final state.
All slab boundaries align to64 except each sequence's final tail.
"""
import torch
from torch import nn

class Model(nn.Module):
    def forward(self,q,k,v,g,beta,initial_state,cu_seqlens):
        total,hq,d=q.shape[1:]; hv=v.shape[2]
        result=torch.empty_like(v); finals=[]
        bounds=cu_seqlens.tolist()
        mask=torch.ones(64,64,dtype=torch.bool,device=q.device).tril()
        eye=torch.eye(64,device=q.device)
        for seq,(left,right) in enumerate(zip(bounds,bounds[1:])):
            state=initial_state[seq].float().transpose(-1,-2).contiguous()
            for pos in range(left,right,4096):
                end=min(pos+4096,right); length=end-pos; pad=(-length)%64
                qf=q[0,pos:end].float();kf=k[0,pos:end].float()
                qf=qf*torch.rsqrt(qf.square().sum(-1,keepdim=True)+1e-6)
                kf=kf*torch.rsqrt(kf.square().sum(-1,keepdim=True)+1e-6)
                qf=qf.repeat_interleave(hv//hq,dim=1)
                kf=kf.repeat_interleave(hv//hq,dim=1)
                def chunks(x):
                    if pad:x=torch.cat((x,x.new_zeros((pad,*x.shape[1:]))),0)
                    return x.reshape(-1,64,*x.shape[1:]).transpose(1,2)
                qc,kc,vc=chunks(qf),chunks(kf),chunks(v[0,pos:end].float())
                gc,bc=chunks(g[0,pos:end].float()).cumsum(-1),chunks(beta[0,pos:end].float())
                delta=gc[..., :,None]-gc[...,None,:]
                decay=torch.exp(delta.masked_fill(~mask,float('-inf')))
                system=(bc[..., :,None]*(kc@kc.transpose(-1,-2))*decay).tril(-1)+eye
                rhs=torch.cat((bc[...,None]*vc,bc[...,None]*gc.exp()[...,None]*kc),-1)
                uw=torch.linalg.solve_triangular(system,rhs,upper=False,unitriangular=True)
                uc,wc=uw.split(d,dim=-1)
                attention=(qc@kc.transpose(-1,-2))*decay
                for c in range(qc.shape[0]):
                    vn=uc[c]-wc[c]@state
                    out=gc[c].exp()[...,None]*(qc[c]@state)+attention[c]@vn
                    chunk_left=pos+c*64;valid=min(64,end-chunk_left)
                    result[0,chunk_left:chunk_left+valid].copy_((out*d**-.5).transpose(0,1)[:valid])
                    dk=kc[c]*(gc[c,:,-1:]-gc[c]).exp()[...,None]
                    state=gc[c,:,-1,None,None].exp()*state+dk.transpose(-1,-2)@vn
            finals.append(state.transpose(-1,-2))
        return result,torch.stack(finals).contiguous()
