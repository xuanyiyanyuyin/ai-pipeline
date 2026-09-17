# -*- coding: utf-8 -*-
"""重排专家权重: 每专家 6 个散块 → 1 个连续记录 (2026-09-16)

输出: /v4flash_expack/experts.bin  每专家 13,369,344 字节, 索引 = lid*256+eid
布局 (每个张量 4096 对齐):
  0                w1.weight [2048,2048] int8  4194304 B
  4194304          w1.scale  [2048,128]  e8m0   262144 B
  4456448          w2.weight [4096,1024] int8  4194304 B
  8650752          w2.scale  [4096,64]   e8m0   262144 B
  8912896          w3.weight [2048,2048] int8  4194304 B
  13312000         w3.scale  [2048,128]  e8m0   262144 B
"""
import json
import os
import struct
import time

SRC = '/v4flash_fp4'
DST = '/v4flash_expack'
N_LAYER, N_EXP = 43, 256
TENSORS = [('w1.weight', (2048, 2048), 4194304),
           ('w1.scale', (2048, 128), 262144),
           ('w2.weight', (4096, 1024), 4194304),
           ('w2.scale', (4096, 64), 262144),
           ('w3.weight', (2048, 2048), 4194304),
           ('w3.scale', (2048, 128), 262144)]
EXP_SIZE = sum(t[2] for t in TENSORS)
assert EXP_SIZE == 13369344, EXP_SIZE

os.makedirs(DST, exist_ok=True)
idx = json.load(open(os.path.join(SRC, 'model.safetensors.index.json')))['weight_map']
shards = sorted(set(idx.values()))
# 建 key → (shard, offset_in_shard_file)
loc = {}
hdr_cache = {}
for sh in shards:
    p = os.path.join(SRC, sh)
    with open(p, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
    base = 8 + hs
    hdr_cache[sh] = (p, base, hdr)
    for k, m in hdr.items():
        loc[k] = (sh, base + m['data_offsets'][0], m['data_offsets'][1] - m['data_offsets'][0])

print('[pack] 专家记录 %d 字节, 共 %d 个, 总计 %.1f GB' %
      (EXP_SIZE, N_LAYER * N_EXP, N_LAYER * N_EXP * EXP_SIZE / 2**30), flush=True)

out = open(os.path.join(DST, 'experts.bin'), 'wb')
buf = bytearray(EXP_SIZE)
t0 = time.time()
done = 0
handles = {}
for lid in range(N_LAYER):
    for eid in range(N_EXP):
        off = 0
        for name, shape, nbytes in TENSORS:
            key = 'layers.%d.ffn.experts.%d.%s' % (lid, eid, name)
            sh, o, ln = loc[key]
            assert ln == nbytes, (key, ln, nbytes)
            f = handles.get(sh)
            if f is None:
                f = open(hdr_cache[sh][0], 'rb')
                handles[sh] = f
            f.seek(o)
            mv = memoryview(buf)[off:off + nbytes]
            n = f.readinto(mv)
            assert n == nbytes, (key, n)
            off += nbytes
        out.write(buf)
        done += 1
    if lid % 5 == 4 or lid == N_LAYER - 1:
        el = time.time() - t0
        print('[pack] %d/%d 层, %.1f GB 已写, %.0fs (%.0f MB/s)' %
              (lid + 1, N_LAYER, done * EXP_SIZE / 2**30, el, done * EXP_SIZE / 2**20 / max(el, 1)), flush=True)
out.close()
for f in handles.values():
    f.close()
print('[pack] 完成 %.1fs' % (time.time() - t0), flush=True)
