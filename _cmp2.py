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
out=transformer_forward(reader,lat,txt,ts,mask,4,4,4,rope_emb=rope).float()
diff=(out-ref).abs()
rel=diff/(ref.abs()+1e-6)
print('HAND std=%.4f mean=%.4f'%(out.std().item(),out.mean().item()))
print('REF  std=%.4f mean=%.4f'%(ref.std().item(),ref.mean().item()))
print('ABS  max=%.6f mean=%.6f'%(diff.max().item(),diff.mean().item()))
print('REL  mean=%.6f'%rel.mean().item())
agree=(rel<1e-3).float().mean().item()*100
print('AGREE %.2f%%'%agree)
print('PASS' if diff.mean().item()<0.05 else 'FAIL')
