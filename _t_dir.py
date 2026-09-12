import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_sched import transformer_forward_sched, compute_rope
# 用不同 latents 测 forward 输出方向
for seed in [42,7,123]:
    torch.manual_seed(seed)
    lat=torch.randn(1,64,128,dtype=torch.bfloat16).cuda()
    txt=torch.randn(1,16,4096,dtype=torch.bfloat16).cuda()
    mask=torch.ones(1,16,dtype=torch.long).cuda()
    ts=torch.tensor([700],dtype=torch.long).cuda()
    rope=compute_rope(lat,4,4,4,8,32,25)
    out=transformer_forward_sched(lat,txt,ts,mask,4,4,4,rope_emb=rope,dtype=torch.bfloat16)
    # 相关性: 如果是 ε 预测, 应和 latents 正相关 (latents含噪声)
    lf=lat.float().flatten(); of=out.float().flatten()
    corr=(lf*of).sum()/(lf.norm()*of.norm()+1e-8)
    print(f'seed{seed}: out_std={of.std().item():.4f} corr(lat,out)={corr.item():.4f}',flush=True)
    # 去噪方向: lat_new = lat + dt*out (dt<0)
    # 如果 corr>0, lat_new = lat - |dt|*out 会减少 latent 幅度(去噪)
    # 如果 corr<0, 会增大
