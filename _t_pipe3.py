import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch, time
t0=time.perf_counter()
import gen_video as gv
print('import OK',flush=True)
pipe=gv.get_pipe()
print(f'get_pipe OK {time.perf_counter()-t0:.0f}s',flush=True)
gv.free_pipe()
print('done',flush=True)
