import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import numpy as np
# 用官方 reference noise_pred (ts=700) 做一步去噪
ref=torch.load(r'D:\ai\_ref_out2.pt',weights_only=True) if os.path.exists(r'D:\ai\_ref_out2.pt') else None
if ref is None:
    ref=torch.load(r'D:\ai\_ref_out.pt',weights_only=True)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(r'D:\ai\models\LTX-Video',subfolder='scheduler')
# 64x64 9帧 seq=2 (ref是[2,64,128], 用前128 patch)
torch.manual_seed(42)
lat=torch.randn(1,64,128,dtype=torch.float32)
# 假设 noise_pred = ref (2,64,128), 用cond分支
np_=ref[1:2].float()  # [1,64,128]
steps=8
seq=64
sigmas=np.linspace(1.0,1/steps,steps)
mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
# 第一步 t=1000
t=timesteps[0]
before=lat.std().item()
lat_new=sch.step(np_,t,lat,return_dict=False)[0]
print(f't={t.item():.0f} before={before:.4f} after={lat_new.std().item():.4f} dt={sch.sigmas[1].item()-sch.sigmas[0].item():.4f}')
print(f'noise_pred std={np_.std().item():.4f} corr(lat,np)={(lat.flatten()*np_.flatten()).sum()/(lat.norm()*np_.norm()+1e-8).item():.4f}')
