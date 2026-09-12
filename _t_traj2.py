import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, torch.nn.functional as F
import numpy as np
from ltx_sched import transformer_forward_sched, compute_rope
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
DEV='cuda'
emb=torch.load(r'D:\ai\_cached_embeds.pt',weights_only=True)
pos=emb['pos'].to(DEV).float(); neg=emb['neg'].to(DEV).float()
pm=emb['pos_mask'].to(DEV).bool(); nm=emb['neg_mask'].to(DEV).bool()
emb2=torch.cat([neg,pos],dim=0); mask2=torch.cat([nm,pm],dim=0)
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(r'D:\ai\models\LTX-Video',subfolder='scheduler')
lat_F=2;lat_H=1;lat_W=1;seq=2
def pack(l5): return l5.permute(0,2,3,4,1).reshape(1,seq,128)
steps=8
sigmas=np.linspace(1.0,1/steps,steps)
mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
def run(sign,label):
    torch.manual_seed(42)
    lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
    latents=pack(lat5).to(DEV)
    print(f'--- {label} ---')
    with torch.inference_mode():
        for i,t in enumerate(timesteps):
            li=torch.cat([latents,latents],dim=0).bfloat16()
            ts=t.expand(2).to(DEV)
            rope=compute_rope(li,lat_F,lat_H,lat_W,8,32,25)
            np_=transformer_forward_sched(li,emb2,ts,mask2,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
            u,c=np_.chunk(2); np_=sign*(u+3.0*(c-u))
            latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
            print(f'step{i} t={t.item():.0f} std={latents.float().std().item():.4f}',flush=True)
run(1,'PLUS')
run(-1,'MINUS')
