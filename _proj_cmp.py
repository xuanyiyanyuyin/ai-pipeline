import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
from ltx_loader import load_header, read_one
import torch
# single-file 的 proj_in (raw: model.diffusion_model.patchify_proj.weight)
sf=r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors'
h1,b1=load_header(sf)
rawk='model.diffusion_model.patchify_proj.weight'
t_sf=read_one(sf,b1,h1[rawk]['data_offsets'][0],h1[rawk]['data_offsets'][1],h1[rawk]['dtype'],tuple(h1[rawk]['shape']))
print('single proj_in std=%.4f'%t_sf.float().std().item())
# 分片里的 proj_in
sh=r'D:\ai\models\LTX-Video\transformer_13b_shards\diffusion_pytorch_model-1-of-4.safetensors'
h2,b2=load_header(sh)
k2='proj_in.weight'
if k2 in h2:
    t_sh=read_one(sh,b2,h2[k2]['data_offsets'][0],h2[k2]['data_offsets'][1],h2[k2]['dtype'],tuple(h2[k2]['shape']))
    d=(t_sf.float()-t_sh.float()).abs().max().item()
    print('shard proj_in std=%.4f'%t_sh.float().std().item(), 'DIFF=%.6f'%d)
else:
    print('proj_in.weight NOT in shard-1! keys sample:', list(h2.keys())[:10])
