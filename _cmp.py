import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_engine import PreadLayerReader, transformer_forward, compute_rope
ref=torch.load(r'D:\ai\_ref_out.pt',weights_only=True)  # (2,64,128)
reader=PreadLayerReader(r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors')
torch.manual_seed(42)
B,S,D=2,64,128
lat=torch.randn(B,S,D,dtype=torch.bfloat16)
txt=torch.randn(B,16,4096,dtype=torch.bfloat16)
mask=torch.ones(B,16,dtype=torch.long)
ts=torch.tensor([700,700],dtype=torch.long)
rope=compute_rope(lat,4,4,4,8,32,25)
out=transformer_forward(reader,lat,txt,ts,mask,4,4,4,rope_emb=rope).float()
diff=(out-ref).abs()
rel=diff/(ref.abs()+1e-6)
print(f'手写 std={out.std().item():.4f} mean={out.mean().item():.4f}')
print(f'参考 std={ref.std().item():.4f} mean={ref.mean().item():.4f}')
print(f'绝对差 max={diff.max().item():.6f} mean={diff.mean().item():.6f}')
print(f'相对差 mean={rel.mean().item():.6f}')
# 一致性: 相对误差 <1e-3 算通过
agree=(rel<1e-3).float().mean().item()*100
print(f'一致比例(<0.1%相对差): {agree:.2f}%')
print('✅ 数学正确' if diff.mean().item()<0.05 else '❌ 不一致',flush=True)
