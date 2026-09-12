import sys, io, os, struct, json, gc
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
import torch
fn=r'D:\ai\models\LTX-Video\text_encoder\model-00001-of-00004.safetensors'
DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool}
with open(fn,'rb') as f:
    hs=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(hs))
    base=8+hs
count=0
for k,meta in header.items():
    if k=='__metadata__': continue
    o0,o1=meta['data_offsets']
    with open(fn,'rb') as f:
        f.seek(base+o0)
        buf=bytearray(f.read(o1-o0))
        t=torch.frombuffer(buf,dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
    print('read', k, tuple(t.shape), t.dtype, flush=True)
    count+=1
    if count>=3: break
print('pread T5 OK', flush=True)
