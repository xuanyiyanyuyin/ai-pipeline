import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
from ltx_loader import load_header, read_one
import torch
# 检查分片文件
import glob
shards=sorted(glob.glob(r'D:\ai\models\LTX-Video\transformer_13b_shards\*.safetensors'))
print('shards:', [os.path.basename(s) for s in shards])
for s in shards:
    try:
        header,base=load_header(s)
        print(f'  {os.path.basename(s)}: {len(header)} keys, size={os.path.getsize(s)/1e9:.2f}GB')
        # 读第一个 key
        k=list(header.keys())[0]
        t=read_one(s, base, header[k]['data_offsets'][0], header[k]['data_offsets'][1], header[k]['dtype'], tuple(header[k]['shape']))
        print(f'    first key {k} shape={tuple(t.shape)} std={t.float().std().item():.2f}')
    except Exception as e:
        print(f'  {os.path.basename(s)}: ERR {type(e).__name__} {str(e)[:100]}')
