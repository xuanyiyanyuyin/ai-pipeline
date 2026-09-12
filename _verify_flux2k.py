import os, sys, time, io
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
t0=time.perf_counter()
import gen_image as gi
fn = gi.generate_image('a bright red apple on a white table, photorealistic, high detail', steps=30, out_path='D:/ai/output/images/flux_verify_2k.png', size=2560, height=1408)
print(f'[test] RESULT {fn} total {time.perf_counter()-t0:.0f}s', flush=True)
gi.free_pipe()
