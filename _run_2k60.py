import os, sys, time, io
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
t0=time.perf_counter()
import gen_video as gv
# 2560x1408 2K, 60fps播放, 生成帧率60 — attention_slicing自动启用
fn, size, n = gv.generate_video(
    'a cute yellow orange cat sitting on a wooden table, bright colors, cinematic lighting, detailed fur',
    steps=50, frames=60, base_w=2560, base_h=1408, target=(2560,1408),
    fps=60, frame_rate=60)
print(f'[run] RESULT {fn} {size} {n}f total {time.perf_counter()-t0:.0f}s', flush=True)
gv.free_pipe()
