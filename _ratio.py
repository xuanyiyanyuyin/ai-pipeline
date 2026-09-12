import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from ltx_engine import PreadLayerReader, transformer_forward, compute_rope
ref=torch.load(r'D:\ai\_ref_out.pt',weights_only=True)  # 官方 diffusers 输出
reader=PreadLayerReader(r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors')
# 用保存的输入 (与生成 _ref_out.pt 时完全一致)
torch.manual_seed(42)
lat=torch.randn(2,64,128,dtype=torch.bfloat16)
txt=torch.randn(2,16,4096,dtype=torch.bfloat16)
mask=torch.ones(2,16,dtype=torch.long)
ts=torch.tensor([700,700],dtype=torch.long)
rope=compute_rope(lat,4,4,4,8,32,25)
out=transformer_forward(reader,lat,txt,ts,mask,4,4,4,rope_emb=rope,dtype=torch.bfloat16).float().cpu()
# 关键对比
ratio=ref.float().std().item()/out.std().item()
corr=((ref.float().flatten()*out.flatten()).sum())/(ref.float().flatten().norm()*out.flatten().norm()+1e-8)
print('REF std=%.4f'%ref.float().std().item())
print('OUT std=%.4f'%out.std().item())
print('RATIO ref/out=%.4f'%ratio)
print('CORR=%.4f'%corr.item())
# 逐元素缩放一致性
diff_mean=(ref.float()-out.float()).abs().mean().item()
print('ABS diff mean=%.6f'%diff_mean)
print('CONCLUSION: 若ratio~1且diff小 => 幅度一致, 黑屏另有原因')
