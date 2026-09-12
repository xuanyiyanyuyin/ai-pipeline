import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import torch.nn.functional as F
from diffusers import LTXVideoTransformer3DModel
from ltx_sched import transformer_forward_sched, compute_rope
DEV='cuda'
emb=torch.load(r'D:\ai\_cached_embeds.pt',weights_only=True)
pos=emb['pos'].bfloat16(); neg=emb['neg'].bfloat16()
emb_cat=torch.cat([neg,pos],dim=0)
mask=torch.cat([emb['neg_mask'],emb['pos_mask']],dim=0)
lat_F=2;lat_H=1;lat_W=1;seq=2
# 相同输入
torch.manual_seed(42)
lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
latents=lat5.permute(0,2,3,4,1).reshape(1,seq,128)
li=torch.cat([latents,latents],dim=0).bfloat16()
ts=torch.tensor([1000,1000],dtype=torch.long)
# 官方 CPU
tf=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
with torch.inference_mode():
    np_o=tf(hidden_states=li,encoder_hidden_states=emb_cat,timestep=ts,encoder_attention_mask=mask,
            num_frames=lat_F,height=lat_H,width=lat_W,rope_interpolation_scale=(0.32,32.0,32.0),return_dict=False)[0]
c_o=np_o.chunk(2)[1]
# 手写 GPU
li_g=li.to(DEV); emb_g=emb_cat.bfloat16().to(DEV); ts_g=ts.to(DEV); mask_g=mask.to(DEV)
rope=compute_rope(li_g,lat_F,lat_H,lat_W,8,32,25)
np_h=transformer_forward_sched(li_g,emb_g,ts_g,mask_g,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
c_h=np_h.chunk(2)[1]
co=c_o.float().flatten(); ch=c_h.float().flatten().cpu()
corr=(ch*co).sum()/(ch.norm()*co.norm()+1e-8)
print('t=1000 hand_vs_off corr=%.6f'%corr.item())
print('hand std=%.4f off std=%.4f hand mean=%.4f off mean=%.4f'%(ch.std().item(),co.std().item(),ch.mean().item(),co.mean().item()))
# 差异分布
d=(ch-co).abs()
print('max_abs_diff=%.4f mean_abs_diff=%.4f'%(d.max().item(),d.mean().item()))
