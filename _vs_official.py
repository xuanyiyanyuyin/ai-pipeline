import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import numpy as np
# 官方 LTXPipeline 跑 64x64 8步, 记录 latent std 轨迹
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
MODEL=r'D:\ai\models\LTX-Video'
tf=LTXVideoTransformer3DModel.from_pretrained(f'{MODEL}/transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,local_files_only=True)
vae=AutoencoderKLLTXVideo.from_pretrained(f'{MODEL}/vae_1024',torch_dtype=torch.bfloat16,local_files_only=True)
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler')
pipe=LTXPipeline(transformer=tf,scheduler=sch,vae=vae,text_encoder=te,tokenizer=tok)
pipe.enable_sequential_cpu_offload(device='cuda')
# hook 记录每步 latent std
stds=[]
orig=sch.step
def hooked(noise_pred,t,latents,**kw):
    out=orig(noise_pred,t,latents,**kw)
    stds.append(out[0].float().std().item())
    return out
sch.step=hooked
res=pipe(prompt='a bright red apple on a white table',num_inference_steps=8,num_frames=9,
         height=64,width=64,guidance_scale=3.0,frame_rate=25,output_type='latent',
         generator=torch.Generator('cpu').manual_seed(42))
print('official stds:', [f'{s:.4f}' for s in stds])
print('final:', res.frames[0].float().std().item())
