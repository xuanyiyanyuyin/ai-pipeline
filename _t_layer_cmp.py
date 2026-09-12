import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import torch.nn.functional as F
from ltx_sched import block_forward, compute_rope, time_embed_forward, linear, get_global, layer_path, pread_safetensors
from diffusers import LTXVideoTransformer3DModel
# 输入 (小, 固定)
torch.manual_seed(0)
B,S=1,8
h=torch.randn(B,S,4096,dtype=torch.bfloat16)
txt=torch.randn(B,8,4096,dtype=torch.bfloat16)
mask=torch.ones(B,8,dtype=torch.long)
ts=torch.tensor([500],dtype=torch.long)
# 官方单层 (CPU)
tf=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
blk=tf.transformer_blocks[0]
gw=tf.scale_shift_table  # 官方 global scale_shift
# 构造官方 block 输入 (需要 temb 等) - 直接调官方 block forward
# 官方 block forward 需要 temb
import diffusers.models.transformers.transformer_ltx as tl
# 用官方 time_embed
temb_o,_=tf.time_embed(ts.flatten(),batch_size=B,hidden_dtype=torch.bfloat16)
temb_o=temb_o.view(B,-1,temb_o.size(-1))
# 官方 rope
rope_o=tf.rope(h,2,4,1,rope_interpolation_scale=(0.32,32.0,32.0))
with torch.inference_mode():
    h_o=blk(hidden_states=h,encoder_hidden_states=txt,temb=temb_o,image_rotary_emb=rope_o,encoder_attention_mask=mask)
print('official block0 out: std=%.6f'%h_o.float().std().item(), flush=True)
# 手写单层
gw_h=get_global(torch.bfloat16)
gwgpu={k:v for k,v in gw_h.items()}
w=pread_safetensors(layer_path(0))
w={k:v.bfloat16() for k,v in w.items()}
temb_h,ets_h=time_embed_forward(ts,B,gwgpu)
# 手写 rope
rope_h=compute_rope(h,2,4,1,8,32,25)
h_hand=block_forward(h,txt,temb_h,rope_h,mask,w)
print('hand block0 out: std=%.6f'%h_hand.float().std().item(), flush=True)
co=h_o.float().flatten(); ch=h_hand.float().flatten()
corr=(ch*co).sum()/(ch.norm()*co.norm()+1e-8)
print('block0 corr=%.6f'%corr.item())
print('hand_std=%.4f off_std=%.4f'%(ch.std().item(),co.std().item()))
