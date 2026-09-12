import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_sched import transformer_forward_sched, compute_rope
from transformers import T5EncoderModel, T5Tokenizer
MODEL=r'D:\ai\models\LTX-Video'; DEV='cuda'
tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
enc=tok('a bright red apple on a white table',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    emb=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state.bfloat16().to(DEV)
mask=enc.attention_mask.bool().to(DEV)
lat_F=2;lat_H=1;lat_W=1;seq=2
def pack(l5): return l5.permute(0,2,3,4,1).reshape(1,seq,128)
torch.manual_seed(42)
lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
lat_noise=pack(lat5).to(DEV)
# 构造一个"结构化"latent: 平滑版本
lat_struct=(lat_noise*0.3 + torch.ones_like(lat_noise)*0.5)
rope=compute_rope(lat_noise.bfloat16(),lat_F,lat_H,lat_W,8,32,25)
t=torch.tensor([700],dtype=torch.long).to(DEV)
with torch.inference_mode():
    out_noise=transformer_forward_sched(lat_noise.bfloat16(),emb,t,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
    out_struct=transformer_forward_sched(lat_struct.bfloat16(),emb,t,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
print('out_noise std=%.4f mean=%.4f'%(out_noise.float().std().item(),out_noise.float().mean().item()))
print('out_struct std=%.4f mean=%.4f'%(out_struct.float().std().item(),out_struct.float().mean().item()))
# 两个输出差异
diff=(out_noise.float()-out_struct.float()).abs().mean().item()
print('diff mean=%.6f (应大, 若forward响应latent)'%diff)
