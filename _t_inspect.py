import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from diffusers import AutoencoderKLLTXVideo
# 用缓存的 final latent? 没有。重新生成 final latent 代价高。
# 直接看 VAE decode 一个已知 latent 的输出范围
vae=AutoencoderKLLTXVideo.from_pretrained(r'D:\ai\models\LTX-Video\vae_1024',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
# 构造 std=0.97 的 latent (类似去噪最终)
lat=torch.randn(1,128,3,8,8,dtype=torch.bfloat16)*0.97
m=vae.latents_mean.view(1,-1,1,1,1).to(lat.dtype)
s=vae.latents_std.view(1,-1,1,1,1).to(lat.dtype)
lat_denorm=lat*s/vae.config.scaling_factor+m
vae=vae.to('cuda')
with torch.inference_mode():
    t_dec=torch.tensor([0.0],device='cuda',dtype=lat_denorm.dtype)
    out=vae.decode(lat_denorm.to('cuda').to(torch.bfloat16),t_dec,return_dict=False)[0].cpu()
print('decode range: %.3f to %.3f'%(out.min().item(),out.max().item()))
print('decode mean: %.3f'%out.mean().item())
# 直接转 [0,1] 看
f=(out+1)/2
print('after (f+1)/2: %.3f to %.3f'%(f.min().item(),f.max().item()))
