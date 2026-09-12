import sys, io, os, time
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, json
t0=time.perf_counter()
from diffusers import LTXVideoTransformer3DModel
cfg=json.load(open(r'D:\ai\ltx_config\config.json'))
print('config:', cfg, flush=True)
print('创建模型...', flush=True)
tf=LTXVideoTransformer3DModel.from_config(cfg)
print(f'from_config OK {time.perf_counter()-t0:.0f}s', flush=True)
print('DONE', flush=True)
