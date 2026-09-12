import sys, io, os, time
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
t0=time.perf_counter()
from ltx_loader import load_ltx_transformer_pread
print('import OK', flush=True)
tf=load_ltx_transformer_pread(
    r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors',
    r'D:\ai\ltx_config', max_workers=12)
print(f'[pread] loaded in {time.perf_counter()-t0:.0f}s', flush=True)
# 验证关键权重
for name,p in tf.named_parameters():
    if name.endswith('proj_in.weight'):
        print(f'proj_in norm: {p.float().norm().item():.2f}', flush=True)
        break
for name,p in tf.named_parameters():
    if name.endswith('attn1.norm_q.weight'):
        print(f'norm_q norm: {p.float().norm().item():.2f}', flush=True)
        break
print('DONE', flush=True)
