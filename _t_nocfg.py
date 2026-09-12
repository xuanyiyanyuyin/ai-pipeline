import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, numpy as np
from ltx_sched import transformer_forward_sched, compute_rope
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
MODEL=r'D:\ai\models\LTX-Video'; DEV='cuda'
tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
sch=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler')
enc=tok('a bright red apple on a white table',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    emb=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state.bfloat16().to(DEV)
mask=enc.attention_mask.bool().to(DEV)
lat_F=2;lat_H=1;lat_W=1;seq=2
def pack(l5): return l5.permute(0,2,3,4,1).reshape(1,seq,128)
torch.manual_seed(42)
lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
latents=pack(lat5).to(DEV)
steps=8
sigmas=np.linspace(1.0,1/steps,steps)
mu=calculate_shift(seq,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
with torch.inference_mode():
    for i,t in enumerate(timesteps):
        li=latents.float()  # 无CFG, batch=1
        ts=t.expand(1).to(DEV)
        rope=compute_rope(li,lat_F,lat_H,lat_W,8,32,25)
        np_=transformer_forward_sched(li,emb,ts,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
        latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
        print(f'step{i} t={t.item():.0f} std={latents.float().std().item():.4f} np_std={np_.float().std().item():.4f}',flush=True)
