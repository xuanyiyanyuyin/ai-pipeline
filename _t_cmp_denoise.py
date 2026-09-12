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
emb_cat=torch.cat([neg,pos],dim=0).to(DEV)
mask=torch.cat([torch.ones(1,128,dtype=torch.long),torch.ones(1,128,dtype=torch.long)],dim=0).to(DEV)
lat_F=2;lat_H=1;lat_W=1;seq=2
# 官方模型 (CPU forward 慢, 但能加载)
tf=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
# 用半去噪 latent (结构化) 对比
torch.manual_seed(42)
lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
# 加结构
lat5[0,:,0,0,0]+=torch.linspace(-1,1,128).view(-1,1,1,1)*0.3
latents=lat5.permute(0,2,3,4,1).reshape(1,seq,128)
li=torch.cat([latents,latents],dim=0).to(DEV)
ts=torch.tensor([500,500],dtype=torch.long).to(DEV)
# 手写 forward
rope=compute_rope(li.bfloat16(),lat_F,lat_H,lat_W,8,32,25)
np_hand=transformer_forward_sched(li.bfloat16(),emb_cat,ts,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
u_h,c_h=np_hand.chunk(2)
# 官方 forward (CPU)
with torch.inference_mode():
    np_off=tf(hidden_states=li.bfloat16(),encoder_hidden_states=emb_cat,timestep=ts,encoder_attention_mask=mask,
              num_frames=lat_F,height=lat_H,width=lat_W,rope_interpolation_scale=(0.32,32.0,32.0),return_dict=False)[0]
u_o,c_o=np_off.chunk(2)
# 相关性: 手写cond vs 官方cond
ch=c_h.float().flatten().cpu(); co=c_o.float().flatten()
corr=(ch*co).sum()/(ch.norm()*co.norm()+1e-8)
print('hand vs official cond corr: %.4f'%corr.item())
print('hand cond std=%.4f official cond std=%.4f'%(ch.std().item(),co.std().item()))
# 手写 vs latent
cos_h=F.cosine_similarity(c_h,latents.flatten().cpu(),dim=0).item()
cos_o=F.cosine_similarity(co,latents.flatten().cpu(),dim=0).item()
print('hand cos(lat)=%.4f official cos(lat)=%.4f'%(cos_h,cos_o))
