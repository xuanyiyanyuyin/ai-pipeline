import sys, io, os, time
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_gen import generate_video
t0=time.perf_counter()
try:
    fn,size,n=generate_video('a bright red apple on a white table', steps=20, frames=17, base_w=256, base_h=256, fps=12, frame_rate=25)
    print('RESULT',fn,size,n,'total',time.perf_counter()-t0)
except Exception:
    import traceback; traceback.print_exc()
