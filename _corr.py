import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_engine import PreadLayerReader, transformer_forward, compute_rope
ref=torch.load(r'D:\ai\_ref_out.pt',weights_only=True)
reader=PreadLayerReader(r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors')
torch.manual_seed(42)
lat=torch.randn(2,64,128,dtype=torch.bfloat16)
txt=torch.randn(2,16,4096,dtype=torch.bfloat16)
mask=torch.ones(2,16,dtype=torch.long)
ts=torch.tensor([700,700],dtype=torch.long)
rope=compute_rope(lat,4,4,4,8,32,25)
out=transformer_forward(reader,lat,txt,ts,mask,4,4,4,rope_emb=rope,dtype=torch.bfloat16).float()
# 相关性
ref_flat=ref.flatten(); out_flat=out.flatten()
dot=(ref_flat*out_flat).sum()
nref=ref_flat.norm(); nout=out_flat.norm()
corr=dot/(nref*nout+1e-8)
print('REF std=%.4f mean=%.4f'%(ref.std().item(),ref.mean().item()))
print('OUT std=%.4f mean=%.4f'%(out.std().item(),out.mean().item()))
print('CORR=%.4f (1=相同方向, -1=相反)'%corr.item())
print('DOT/REFnorm^2=%.4f (应~1 if scale same)'%(dot/(nref*nref+1e-8)).item())
# 逐元素方向一致率
same=((ref_flat*out_flat)>0).float().mean().item()*100
print('SAME_SIGN %.1f%%'%same)
