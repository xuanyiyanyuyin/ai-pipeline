"""用模拟数据测完整 transformer_forward (多层集成, 条件注入)"""
import sys, io, os, math
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
import torch
import torch.nn.functional as F

# 小模型: 模拟 n_layers 层, 每层复用同一权重
H=64; nh=2; hd=32; n_layers=3
from ltx_engine import block_forward, time_embed_forward, rms_norm, linear

def make_layer_weights(seed=0):
    torch.manual_seed(seed)
    lw={}
    lw['_heads']=nh; lw['_head_dim']=hd
    lw['norm1.weight']=torch.randn(H)
    lw['norm2.weight']=torch.randn(H)
    for p in ['attn1','attn2']:
        for proj in ['to_q','to_k','to_v']:
            lw[f'{p}.{proj}.weight']=torch.randn(nh*hd,H)*0.1
            lw[f'{p}.{proj}.bias']=torch.randn(nh*hd)*0.01
        lw[f'{p}.norm_q.weight']=torch.randn(nh*hd)
        lw[f'{p}.norm_k.weight']=torch.randn(nh*hd)
        lw[f'{p}.to_out.0.weight']=torch.randn(H,nh*hd)*0.1
        lw[f'{p}.to_out.0.bias']=torch.randn(H)*0.01
    lw['ff.net.0.proj.weight']=torch.randn(H*4,H)*0.1
    lw['ff.net.0.proj.bias']=torch.randn(H*4)*0.01
    lw['ff.net.2.weight']=torch.randn(H,H*4)*0.1
    lw['ff.net.2.bias']=torch.randn(H)*0.01
    lw['scale_shift_table']=torch.randn(6,H)*0.01
    return lw

def make_global(seed=0):
    torch.manual_seed(seed)
    gw={}
    gw['proj_in.weight']=torch.randn(H,128)*0.1; gw['proj_in.bias']=torch.randn(H)*0.01
    gw['proj_out.weight']=torch.randn(128,H)*0.1; gw['proj_out.bias']=torch.randn(128)*0.01
    gw['scale_shift_table']=torch.randn(2,H)*0.01
    gw['time_embed.emb.timestep_embedder.linear_1.weight']=torch.randn(H,256)*0.1
    gw['time_embed.emb.timestep_embedder.linear_1.bias']=torch.randn(H)*0.01
    gw['time_embed.emb.timestep_embedder.linear_2.weight']=torch.randn(H,H)*0.1
    gw['time_embed.emb.timestep_embedder.linear_2.bias']=torch.randn(H)*0.01
    gw['time_embed.linear.weight']=torch.randn(6*H,H)*0.1; gw['time_embed.linear.bias']=torch.randn(6*H)*0.01
    gw['caption_projection.linear_1.weight']=torch.randn(H,H)*0.1
    gw['caption_projection.linear_1.bias']=torch.randn(H)*0.01
    gw['caption_projection.linear_2.weight']=torch.randn(H,H)*0.1
    gw['caption_projection.linear_2.bias']=torch.randn(H)*0.01
    return gw

def mini_forward(lat, txt, timestep, mask, gw, layer_w):
    B,S,_=lat.shape
    h=linear(lat, gw['proj_in.weight'], gw.get('proj_in.bias'))
    temb, emb_ts=time_embed_forward(timestep,B,gw)
    t=linear(txt, gw['caption_projection.linear_1.weight'], gw['caption_projection.linear_1.bias'])
    t=F.gelu(t,approximate='tanh')
    t=linear(t, gw['caption_projection.linear_2.weight'], gw['caption_projection.linear_2.bias'])
    t=t.view(B,-1,H)
    for lid in range(n_layers):
        h=block_forward(h,t,temb,None,mask,layer_w)
    ssv=gw['scale_shift_table'][None,None].to(emb_ts.dtype)+emb_ts[:,:,None]
    shift,scale=ssv[:,:,0],ssv[:,:,1]
    h=rms_norm(h,None).to(shift.dtype)*(1+scale)+shift
    out=linear(h,gw['proj_out.weight'],gw.get('proj_out.bias'))
    return out

gw=make_global(1); lw=make_layer_weights(2)
torch.manual_seed(3)
B,S=2,16
lat=torch.randn(B,S,128)*0.1
txt_uncond=torch.zeros(B,8,H)
txt_cond=torch.randn(B,8,H)*0.1
ts=torch.tensor([700,700],dtype=torch.long)
# 测试1: 全1 mask
mask=torch.ones(B,8,dtype=torch.long)
o1=mini_forward(lat,txt_uncond,ts,mask,gw,lw)
o2=mini_forward(lat,txt_cond,ts,mask,gw,lw)
print('full 全1mask uncond/cond diff=%.6f'%(o1-o2).abs().mean().item())
# 测试2: 真实mask (uncond全pad, cond前4有效)
mask2=torch.zeros(B,8,dtype=torch.long); mask2[:,:4]=1
o3=mini_forward(lat,txt_uncond,ts,mask2,gw,lw)
o4=mini_forward(lat,txt_cond,ts,mask2,gw,lw)
print('full 真实mask uncond/cond diff=%.6f'%(o3-o4).abs().mean().item())
print('条件注入:', 'OK' if (o3-o4).abs().mean().item()>1e-6 else 'FAIL')
