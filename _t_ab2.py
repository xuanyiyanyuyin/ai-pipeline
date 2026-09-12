import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, numpy as np
from ltx_sched import transformer_forward_sched, compute_rope
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
DEV='cuda'
emb=torch.load(r'D:\ai\_cached_embeds.pt',weights_only=True)
pos=emb['pos'].bfloat16(); neg=emb['neg'].bfloat16()
emb_cat=torch.cat([neg,pos],dim=0).to(DEV)
mask=torch.cat([emb['neg_mask'],emb['pos_mask']],dim=0).to(DEV)
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(r'D:\ai\models\LTX-Video',subfolder='scheduler')

def run_ab(w,h,frames=9):
    lat_F=(frames-1)//8+1; lat_H=h//32; lat_W=w//32
    seq=lat_F*lat_H*lat_W
    torch.manual_seed(42)
    lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
    latents=lat5.permute(0,2,3,4,1).reshape(1,seq,128).to(DEV)
    sigmas=np.linspace(1.0,1/20,20)
    mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
    timesteps,_=retrieve_timesteps(sch,20,'cpu',sigmas=sigmas,mu=mu)
    sch.set_timesteps(20,mu=mu); sch._step_index=None
    with torch.inference_mode():
        for i,t in enumerate(timesteps):
            li=torch.cat([latents,latents],dim=0).bfloat16()
            ts=t.expand(2).to(DEV)
            rope=compute_rope(li,lat_F,lat_H,lat_W,8,32,25)
            # dump rope 第一次
            if i==0 and (w,h)==(128,128):
                print('128 rope[0,0,:4]', rope[0][0,0,:4].tolist(), 'shape', tuple(rope[0].shape), flush=True)
            if i==0 and (w,h)==(256,256):
                print('256 rope[0,0,:4]', rope[0][0,0,:4].tolist(), 'shape', tuple(rope[0].shape), flush=True)
            np_=transformer_forward_sched(li,emb_cat,ts,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
            u,c=np_.chunk(2)
            np_=u+3.0*(c-u)
            latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
    lat=latents.float().cpu()
    print(f'[{w}x{h}] seq={seq} mean={lat.mean().item():.4f} std={lat.std().item():.4f} min={lat.min().item():.4f} max={lat.max().item():.4f}', flush=True)
    return lat

l128=run_ab(128,128)
l256=run_ab(256,256)
# 对比通道分布
for name,l in [('128',l128),('256',l256)]:
    ch_std=l.std(dim=(0,1))
    print(f'{name} 通道std: min={ch_std.min().item():.4f} max={ch_std.max().item():.4f} mean={ch_std.mean().item():.4f}', flush=True)
torch.save({'l128':l128,'l256':l256}, r'D:\ai\_ab_lat.pt')
print('saved', flush=True)
