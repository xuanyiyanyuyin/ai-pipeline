import sys, io, os, time
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
t0=time.perf_counter()
from diffusers import LTXVideoTransformer3DModel
print('import OK',flush=True)
tf=LTXVideoTransformer3DModel.from_pretrained(
    r'D:\ai\models\LTX-Video\transformer_13b_shards',
    torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
print(f'TF loaded {time.perf_counter()-t0:.0f}s',flush=True)
# 同输入 (和 _t_engine.py 相同)
torch.manual_seed(42)
B,S,D=2,64,128
lat=torch.randn(B,S,D,dtype=torch.bfloat16)
txt=torch.randn(B,16,4096,dtype=torch.bfloat16)
mask=torch.ones(B,16,dtype=torch.long)
ts=torch.tensor([700,700],dtype=torch.long)
with torch.inference_mode():
    out=tf(hidden_states=lat, encoder_hidden_states=txt, timestep=ts,
           encoder_attention_mask=mask, num_frames=4, height=4, width=4, return_dict=False)[0]
print(f'out shape {tuple(out.shape)} std={out.float().std().item():.4f} mean={out.float().mean().item():.4f}',flush=True)
torch.save(out.float(), r'D:\ai\_ref_out.pt')
print('saved',flush=True)
