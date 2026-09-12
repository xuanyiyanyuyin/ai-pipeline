"""用模拟数据快速验证手写引擎逻辑 (不加载真实权重)"""
import sys, io, os, math
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
import torch
import torch.nn.functional as F
from ltx_engine import block_forward, rms_norm, linear

def make_small_weights(seed=0):
    torch.manual_seed(seed)
    H=64; nh=2; hd=32  # 模拟: 64维, 2 heads x 32
    w={}
    w['_heads']=nh; w['_head_dim']=hd
    w['norm1.weight']=torch.randn(H)
    w['norm2.weight']=torch.randn(H)
    for p in ['attn1','attn2']:
        w[f'{p}.to_q.weight']=torch.randn(nh*hd,H)*0.1
        w[f'{p}.to_q.bias']=torch.randn(nh*hd)*0.01
        w[f'{p}.to_k.weight']=torch.randn(nh*hd,H)*0.1
        w[f'{p}.to_k.bias']=torch.randn(nh*hd)*0.01
        w[f'{p}.to_v.weight']=torch.randn(nh*hd,H)*0.1
        w[f'{p}.to_v.bias']=torch.randn(nh*hd)*0.01
        w[f'{p}.norm_q.weight']=torch.randn(nh*hd)
        w[f'{p}.norm_k.weight']=torch.randn(nh*hd)
        w[f'{p}.to_out.0.weight']=torch.randn(H,nh*hd)*0.1
        w[f'{p}.to_out.0.bias']=torch.randn(H)*0.01
    w['ff.net.0.proj.weight']=torch.randn(H*4,H)*0.1
    w['ff.net.0.proj.bias']=torch.randn(H*4)*0.01
    w['ff.net.2.weight']=torch.randn(H,H*4)*0.1
    w['ff.net.2.bias']=torch.randn(H)*0.01
    w['scale_shift_table']=torch.randn(6,H)*0.01
    return w, H, nh, hd

def test_block():
    w,H,nh,hd=make_small_weights()
    torch.manual_seed(1)
    B,S=2,16
    h=torch.randn(B,S,H)*0.1
    txt_uncond=torch.zeros(B,8,H)   # 空文本
    txt_cond=torch.randn(B,8,H)*0.1 # 有文本
    temb=torch.randn(B,1,6,H)*0.01
    # 无 mask (全1): 两个分支都应看到全部文本
    mask_ones=torch.ones(B,8,dtype=torch.long)
    o1=block_forward(h,txt_uncond,temb,None,mask_ones,w)
    o2=block_forward(h,txt_cond,temb,None,mask_ones,w)
    diff_nomask=(o1-o2).abs().mean().item()
    # 有 mask: uncond全pad, cond前4有效
    mask_real=torch.zeros(B,8,dtype=torch.long); mask_real[:,:4]=1
    o3=block_forward(h,txt_uncond,temb,None,mask_real,w)
    o4=block_forward(h,txt_cond,temb,None,mask_real,w)
    diff_mask=(o3-o4).abs().mean().item()
    # 有 mask 时 cond 有效token应保留: 比较 o2(全1) vs o4(真实mask)
    diff_cond_mask=(o2-o4).abs().mean().item()
    print(f'无mask  uncond/cond diff = {diff_nomask:.4f}')
    print(f'真实mask uncond/cond diff = {diff_mask:.4f}')
    print(f'cond分支 全1vs真实mask diff = {diff_cond_mask:.4f} (应>0, 说明mask改变了cond)')
    print('条件注入:', 'OK' if diff_mask>1e-5 else 'FAIL')
    return diff_nomask, diff_mask

if __name__=='__main__':
    test_block()
