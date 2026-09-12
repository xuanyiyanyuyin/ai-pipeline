"""拆分 LTX transformer 为 48 层独立文件, 奇偶分盘 (照用户三级调度方案)
第1级磁盘: C盘(偶数层) + D盘(奇数层), 每层一个 safetensors
用 pread 读 single-file (不走 mmap), 逐层写盘释放
"""
import sys, io, os, struct, json, gc, threading
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from concurrent.futures import ThreadPoolExecutor

SRC=r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors'
C_DIR=r'C:\ai\ltx_shards'
D_DIR=r'D:\ai\ltx_shards'
DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32}
TRANSFORMER_RENAME={'model.diffusion_model.':'','patchify_proj':'proj_in','adaln_single':'time_embed','q_norm':'norm_q','k_norm':'norm_k'}
def conv(k):
    for a,b in TRANSFORMER_RENAME.items(): k=k.replace(a,b)
    return k

# 读 header
with open(SRC,'rb') as f:
    hs=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(hs))
base=8+hs
print(f'header: {len(header)} keys, base={base}',flush=True)

# 按层分组
layer_keys={}  # layer_id -> [(raw_key, ck)]
global_keys=[]
for raw in header:
    if raw=='__metadata__': continue
    if not raw.startswith('model.diffusion_model.'): continue
    ck=conv(raw)
    if ck.startswith('transformer_blocks.'):
        lid=int(ck.split('.')[1])
        layer_keys.setdefault(lid,[]).append((raw,ck))
    else:
        global_keys.append((raw,ck))
print(f'layers: {len(layer_keys)}, global: {len(global_keys)}',flush=True)

# pread 读单个 tensor: 用文件对象 seek+read 循环读满 (Windows 无 os.pread)
_tl=threading.local()
def read_tensor(o0,o1,dt,sh):
    f=getattr(_tl,'f',None)
    if f is None:
        f=open(SRC,'rb'); _tl.f=f
    n=o1-o0
    f.seek(base+o0)
    buf=bytearray()
    while len(buf)<n:
        chunk=f.read(n-len(buf))
        if not chunk: break
        buf.extend(chunk)
    return torch.frombuffer(bytes(buf),dtype=DTYPE_MAP[dt]).reshape(sh)

os.makedirs(C_DIR,exist_ok=True); os.makedirs(D_DIR,exist_ok=True)

# 逐层写盘 (每层独立文件)
for lid in sorted(layer_keys.keys()):
    lk=layer_keys[lid]
    tensors={}
    for raw,ck in lk:
        meta=header[raw]
        t=read_tensor(meta['data_offsets'][0],meta['data_offsets'][1],meta['dtype'],tuple(meta['shape']))
        # 去掉 transformer_blocks.{lid}. 前缀
        short=ck
        p=f'transformer_blocks.{lid}.'
        if short.startswith(p): short=short[len(p):]
        tensors[short]=t
    # 分盘
    drive=C_DIR if lid%2==0 else D_DIR
    path=f'{drive}/layer_{lid:02d}.safetensors'
    from safetensors.torch import save_file
    save_file(tensors,path)
    sz=os.path.getsize(path)/1e6
    del tensors; gc.collect()
    print(f'layer {lid} -> {path} ({len(lk)} keys, {sz:.0f}MB) {drive.split(chr(92))[0]}',flush=True)

# global 权重单独存
g_tensors={}
for raw,ck in global_keys:
    meta=header[raw]
    g_tensors[ck]=read_tensor(meta['data_offsets'][0],meta['data_offsets'][1],meta['dtype'],tuple(meta['shape']))
from safetensors.torch import save_file
save_file(g_tensors,f'{D_DIR}/global.safetensors')
print(f'global -> {D_DIR}/global.safetensors ({len(g_tensors)} keys)',flush=True)
print('DONE',flush=True)
