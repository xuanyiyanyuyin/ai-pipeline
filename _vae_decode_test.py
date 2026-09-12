import sys, io, os, time
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from diffusers import AutoencoderKLLTXVideo
vae=AutoencoderKLLTXVideo.from_pretrained(r'D:\ai\models\LTX-Video\vae_1024',torch_dtype=torch.bfloat16,local_files_only=True)
print('VAE loaded')
# 构造最小 latent [B,C,F,H,W] = [1,128,2,1,1] (latent 2帧 1x1)
lat=torch.randn(1,128,2,1,1,dtype=torch.bfloat16)
# 按 pipeline 方式 denormalize
lat_denorm=lat*vae.latents_std.view(1,-1,1,1,1).to(lat.dtype)+vae.latents_mean.view(1,-1,1,1,1).to(lat.dtype)
print('lat_denorm', tuple(lat_denorm.shape))
# pipeline 方式: timestep 是 [B] = [1], 值 0.0
for tval in [0.0]:
    t=torch.tensor([tval],dtype=lat_denorm.dtype)
    try:
        out=vae.decode(lat_denorm, t, return_dict=False)[0]
        print(f'temb={tval} shape {tuple(t.shape)} -> decoded {tuple(out.shape)} OK')
    except Exception as e:
        print(f'temb={tval} shape {tuple(t.shape)} -> ERR {str(e)[:200]}')
