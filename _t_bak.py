import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from gen_video import generate_video
try:
    fn,size,n=generate_video('a bright red apple', steps=2, frames=17, base_w=32, base_h=32, target=(32,32))
    print('RESULT',fn,size,n)
except Exception:
    import traceback; traceback.print_exc()
