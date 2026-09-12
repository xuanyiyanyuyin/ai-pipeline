import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, numpy as np
import torch.nn.functional as F
from ltx_sched import transformer_forward_sched, compute_rope, cleanup
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
DEV='cuda'
emb=torch.load(r'D:\ai\_cached_embeds.pt',weights_only=True)
pos=emb['pos'].bfloat16(); neg=emb['neg'].bfloat16()
emb_cat=torch.cat([neg,pos],dim=0).to(DEV)
mask=torch.cat([emb['neg_mask'],emb['pos_mask']],dim=0).to(DEV)
lat_F=3;lat_H=8;lat_W=8;seq=192
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(r'D:\ai\models\LTX-Video',subfolder='scheduler')
steps=20
sigmas=np.linspace(1.0,1/steps,steps)
mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
torch.manual_seed(42)
lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
latents=lat5.permute(0,2,3,4,1).reshape(1,seq,128).to(DEV)
with torch.inference_mode():
    for i,t in enumerate(timesteps):
        li=torch.cat([latents,latents],dim=0).bfloat16()
        ts=t.expand(2).to(DEV)
        rope=compute_rope(li,lat_F,lat_H,lat_W,8,32,25)
        np_=transformer_forward_sched(li,emb_cat,ts,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
        u,c=np_.chunk(2)
        np_=u+3.0*(c-u)
        latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
        if i in [0,9,19]:
            # 空间结构: 各 patch 是否相同
            patches=latents.cpu().float()  # (1,seq,128)
            p0=patches[0,0]; p1=patches[0,1]; p_mid=patches[0,seq//2]
            print(f'step{i}: std={latents.float().std().item():.4f} patch0_std={p0.std().item():.4f} patch0_mid={p_mid.std().item():.4f} patch01_diff={(p0-p1).abs().mean().item():.4f} mean={latents.float().mean().item():.4f}', flush=True)
cleanup()
