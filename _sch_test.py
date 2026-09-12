import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(r'D:\ai\models\LTX-Video',subfolder='scheduler')
steps=8; seq=8
sigmas=np.linspace(1.0,1/steps,steps)
mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
sch.set_timesteps(steps,mu=mu); sch._step_index=None
print('sigmas:', [f'{s:.3f}' for s in sch.sigmas[:4]], '...', [f'{s:.3f}' for s in sch.sigmas[-3:]])
print('timesteps:', [f'{t:.0f}' for t in timesteps])
# 模拟: 如果 model_output=v=latents(噪声), step 后应降低
torch.manual_seed(0)
lat=torch.randn(1,8,128)
v=lat.clone()  # 假设 v=噪声
for i,t in enumerate(timesteps):
    lat2=sch.step(v.float(),t,lat.float(),return_dict=False)[0]
    print(f'step{i} t={t:.0f} lat_std={lat2.std():.4f}')
    lat=lat2
