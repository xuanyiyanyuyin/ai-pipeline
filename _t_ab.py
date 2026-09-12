import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch, numpy as np
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from ltx_sched import compute_rope
MODEL=r'D:\ai\models\LTX-Video'
emb=torch.load(r'D:\ai\_cached_embeds.pt',weights_only=True)
pos_emb=emb['pos'].bfloat16(); neg_emb=emb['neg'].bfloat16()
pos_mask=emb['pos_mask']; neg_mask=emb['neg_mask']
tf=LTXVideoTransformer3DModel.from_pretrained(f'{MODEL}/transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler')
# 用官方 forward 直接去噪 (不走 pipeline, 避免 VAE)
# 128 vs 256 各跑20步, 记录 latent 统计
def denoise(seq, lat_F, lat_H, lat_W, frames):
    torch.manual_seed(42)
    lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
    latents=lat5.permute(0,2,3,4,1).reshape(1,seq,128)
    from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
    sigmas=np.linspace(1.0,1/20,20)
    mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
    timesteps,_=retrieve_timesteps(sch,20,'cpu',sigmas=sigmas,mu=mu)
    sch.set_timesteps(20,mu=mu); sch._step_index=None
    with torch.inference_mode():
        for i,t in enumerate(timesteps):
            li=torch.cat([latents,latents],dim=0).bfloat16()
            ts=t.expand(2)
            np_=tf(hidden_states=li,encoder_hidden_states=torch.cat([neg_emb,pos_emb],dim=0),timestep=ts,
                   encoder_attention_mask=torch.cat([neg_mask,pos_mask],dim=0),
                   num_frames=lat_F,height=lat_H,width=lat_W,rope_interpolation_scale=(0.32,32.0,32.0),
                   return_dict=False)[0]
            u,c=np_.chunk(2)
            np_=u+3.0*(c-u)
            latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
    return latents.float().cpu()
# 128 (seq=32, 9帧)
l128=denoise(32,2,4,4,9)
print('128 latent: mean=%.4f std=%.4f min=%.4f max=%.4f'%(l128.mean().item(),l128.std().item(),l128.min().item(),l128.max().item()), flush=True)
# 256 (seq=128, 9帧) 固定frames=9
l256=denoise(128,2,8,8,9)
print('256 latent: mean=%.4f std=%.4f min=%.4f max=%.4f'%(l256.mean().item(),l256.std().item(),l256.min().item(),l256.max().item()), flush=True)
torch.save({'l128':l128,'l256':l256}, r'D:\ai\_ab_lat.pt')
print('saved', flush=True)
