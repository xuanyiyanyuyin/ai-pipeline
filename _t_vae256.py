import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from diffusers import AutoencoderKLLTXVideo
vae=AutoencoderKLLTXVideo.from_pretrained(r'D:\ai\models\LTX-Video\vae_1024',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
# 256x256 17帧 latent: [1,128,3,8,8]
lat=torch.randn(1,128,3,8,8,dtype=torch.bfloat16)
lat_denorm=lat*vae.latents_std.view(1,-1,1,1,1).to(lat.dtype)+vae.latents_mean.view(1,-1,1,1,1).to(lat.dtype)
vae=vae.to('cuda')
with torch.inference_mode():
    t_dec=torch.tensor([0.0],device='cuda',dtype=lat_denorm.dtype)
    out=vae.decode(lat_denorm.to('cuda'),t_dec,return_dict=False)[0]
print('VAE decode OK', tuple(out.shape), flush=True)
