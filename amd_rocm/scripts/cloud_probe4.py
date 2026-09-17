# -*- coding: utf-8 -*-
"""云端四分段探针 (移植自本机 v4flash_attn_timer.py, 加一路 get_expert 搬运计时)
   用法: 先停掉 8123 服务, 再跑本脚本 (它自己加载一份引擎)
"""
import collections
import os
import sys
import time

os.environ['V4F_BASE'] = '/workspace'
os.environ['V4F_MODEL_DIR'] = '/v4flash_fp4'
os.environ['V4F_SRC_DIR'] = '/workspace/engine/_v4flash_src'
os.environ['V4F_TORCH_FP8'] = '1'
os.environ['V4F_TRITON_FP4'] = '1'
os.environ['V4F_GROUPED_MOE'] = '1'
os.environ['V4F_FUSE_EPI'] = '0'
os.environ['V4F_RESIDENT_LAYERS'] = '1'
os.environ['V4F_GPU_EXP_MAX'] = '2500'
os.environ['V4F_EXP_STORE_DEV'] = ''
os.environ['V4F_LRU_ITEMS'] = '13000'
os.environ['V4F_KEEP_GPU_CACHE'] = '1'
os.environ['V4F_MTP'] = '0'          # 先关 MTP, 只看单 token 前向
sys.path.insert(0, '/workspace/engine')

import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

T = collections.defaultdict(float)
C = collections.defaultdict(int)


def wrap_attn():
    orig = DeepseekV4Attention.forward

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        T['attn'] += time.perf_counter() - t0
        C['attn'] += 1
        return r
    DeepseekV4Attention.forward = timed


def wrap_moe():
    orig = vs.V4LazyMoE.forward

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        T['moe'] += time.perf_counter() - t0
        C['moe'] += 1
        return r
    vs.V4LazyMoE.forward = timed


def wrap_fetch():
    orig = vs.V4FlashEngine.get_expert
    orig_mtp = vs.V4FlashEngine.get_mtp_expert

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        T['fetch'] += time.perf_counter() - t0
        C['fetch'] += 1
        return r

    def timed_mtp(self, *a, **k):
        t0 = time.perf_counter()
        r = orig_mtp(self, *a, **k)
        T['fetch_mtp'] += time.perf_counter() - t0
        C['fetch_mtp'] += 1
        return r
    vs.V4FlashEngine.get_expert = timed
    vs.V4FlashEngine.get_mtp_expert = timed_mtp


def wrap_fwd():
    orig = vs.V4FlashEngine.forward

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        T['fwd'] += time.perf_counter() - t0
        C['fwd'] += 1
        return r
    vs.V4FlashEngine.forward = timed


wrap_attn(); wrap_moe(); wrap_fetch(); wrap_fwd()
print('[cld] 加载引擎...', flush=True)
t0 = time.perf_counter()
eng = vs.V4FlashEngine()
print('[cld] 引擎加载完成 %.0fs' % (time.perf_counter() - t0), flush=True)
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好', return_tensors='pt').input_ids.to(vs.DEVICE)

print('[cld] 预热 forward (触发预热+预填)...', flush=True)
t0 = time.perf_counter()
logits, cache, h = eng.forward(ids, return_hidden=True)
print('[cld] 首次 forward %.1fs' % (time.perf_counter() - t0), flush=True)

for n in range(4):
    for k in list(T.keys()):
        T[k] = 0.0
    for k in list(C.keys()):
        C[k] = 0
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    t0 = time.perf_counter()
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    wall = time.perf_counter() - t0
    fwd = T['fwd']
    attn = T['attn']
    moe = T['moe']
    fetch = T['fetch'] + T['fetch_mtp']
    other = max(0.0, fwd - attn - moe)
    print('  token%d: 墙钟=%.3fs | fwd=%.3fs | attn=%.3fs(%.0f%%) | moe=%.3fs(%.0f%%) | 其中搬运=%.3fs | other=%.3fs(%.0f%%) | 专家查找=%d 次'
          % (n + 1, wall, fwd, attn, attn / max(fwd, 1e-9) * 100, moe, moe / max(fwd, 1e-9) * 100,
             fetch, other, other / max(fwd, 1e-9) * 100, C['fetch'] + C['fetch_mtp']), flush=True)
print('[cld] DONE', flush=True)
