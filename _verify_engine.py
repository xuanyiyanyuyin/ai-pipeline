"""手写引擎验证: 128红苹果(快) / 256(慢) 一键跑
用法: python _verify_engine.py [size]
"""
import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_gen import generate_video
size=int(sys.argv[1]) if len(sys.argv)>1 else 128
frames=9 if size<=128 else 17
print(f'=== 手写引擎验证: {size}x{size} {frames}帧 ===', flush=True)
try:
    fn,sz,n=generate_video('a bright red apple on a white table', steps=20, frames=frames, base_w=size, base_h=size, fps=12, frame_rate=25)
    print('RESULT',fn,sz,n, flush=True)
except Exception:
    import traceback; traceback.print_exc()
