import sys, io, os, time, gc
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_loader import load_keys_pread, convert_ltx_tf_keys
import struct, json

fn=r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors'
# 读 header
with open(fn,'rb') as f:
    hs=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(hs))
tf_keys=[k for k in header if k.startswith('model.diffusion_model.')]
print(f'keys: {len(tf_keys)}', flush=True)

# 分批读+转换, 但不一次性持有
from diffusers import LTXVideoTransformer3DModel
import json as _json
cfg=_json.load(open(r'D:\ai\ltx_config\config.json'))
print('创建模型...', flush=True)
tf=LTXVideoTransformer3DModel.from_config(cfg)
print('模型创建 OK', flush=True)

# 逐批 load_state_dict (不累积 converted)
BATCH=200
loaded=0
sd={}
for i in range(0,len(tf_keys),BATCH):
    batch=tf_keys[i:i+BATCH]
    tensors=load_keys_pread(fn,batch)
    for k,v in tensors.items():
        newk=convert_ltx_tf_keys(k)
        sd[newk]=v.to(torch.bfloat16)
    # 立即 load 这批
    miss,unexp=tf.load_state_dict(sd, strict=False)
    sd.clear(); del tensors; gc.collect()
    loaded+=len(batch)
    print(f'[{loaded}/{len(tf_keys)}] miss={len(miss)} unexp={len(unexp)}', flush=True)
print('全部加载完成!', flush=True)
for name,p in tf.named_parameters():
    if name.endswith('proj_in.weight'):
        print(f'proj_in norm: {p.float().norm().item():.2f}', flush=True); break
print('DONE', flush=True)
