import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from vae_loader import load_vae_pread
print('loading vae...', flush=True)
vae=load_vae_pread(r'D:\ai\models\LTX-Video\vae_1024')
print('vae loaded', flush=True)
print('enabling tiling...', flush=True)
vae.enable_tiling()
print('tiling enabled, use_tiling=', vae.use_tiling, flush=True)
print('tile sizes:', vae.tile_sample_min_height, vae.tile_sample_min_width, vae.tile_sample_min_num_frames, flush=True)
print('to cuda...', flush=True)
vae=vae.to('cuda')
print('on cuda', flush=True)
lat=torch.randn(1,128,2,8,8,dtype=torch.bfloat16)
m=vae.latents_mean.view(1,-1,1,1,1).to(lat.dtype)
s=vae.latents_std.view(1,-1,1,1,1).to(lat.dtype)
lat_d=lat*s/vae.config.scaling_factor+m
print('decoding...', flush=True)
with torch.inference_mode():
    t_dec=torch.tensor([0.0],device='cuda',dtype=lat_d.dtype)
    out=vae.decode(lat_d.to('cuda').to(torch.bfloat16),t_dec,return_dict=False)[0].cpu()
print('decode OK', tuple(out.shape), flush=True)
