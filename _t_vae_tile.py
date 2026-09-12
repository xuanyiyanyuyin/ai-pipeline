import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from vae_loader import load_vae_pread
vae=load_vae_pread(r'D:\ai\models\LTX-Video\vae_1024')
print('VAE loaded', flush=True)
vae.enable_tiling()  # 在.to(cuda)之前
print('tiling enabled', flush=True)
vae=vae.to('cuda')
# 256x256 9帧 latent [1,128,2,8,8]
lat=torch.randn(1,128,2,8,8,dtype=torch.bfloat16)
m=vae.latents_mean.view(1,-1,1,1,1).to(lat.dtype)
s=vae.latents_std.view(1,-1,1,1,1).to(lat.dtype)
lat_d=lat*s/vae.config.scaling_factor+m
with torch.inference_mode():
    t_dec=torch.tensor([0.0],device='cuda',dtype=lat_d.dtype)
    out=vae.decode(lat_d.to('cuda').to(torch.bfloat16),t_dec,return_dict=False)[0].cpu()
print('256 tiling decode OK', tuple(out.shape), 'range %.3f~%.3f'%(out.min().item(),out.max().item()), flush=True)
