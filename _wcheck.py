import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
# 对比分片 from_pretrained 和 from_single_file 的权重
from diffusers import LTXVideoTransformer3DModel
# 分片版 (current gen_video 用)
tf_shard=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
print('shard loaded')
# 用 safe_open 读 single-file 原始 key, 对比关键层
from safetensors import safe_open
fn=r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors'
# 对比 proj_in (patchify_proj), norm_q, caption_projection
checks={
  'proj_in.weight':'model.diffusion_model.patchify_proj.weight',
  'time_embed.linear.weight':'model.diffusion_model.adaln_single.linear.weight',
  'caption_projection.linear_1.weight':'model.diffusion_model.caption_projection.linear_1.weight',
}
with safe_open(fn,framework='pt') as f:
    for model_k, raw_k in checks.items():
        # shard 模型里的权重
        sd=tf_shard.state_dict()
        mw=sd[model_k].float()
        rw=f.get_tensor(raw_k).float()
        d=(mw-rw).abs().max().item()
        print(f'{model_k}: diff={d:.8f} {"OK" if d<1e-6 else "DIFF!"}')
# norm_q (q_norm 重命名)
with safe_open(fn,framework='pt') as f:
    rw=f.get_tensor('model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight').float()
mw=tf_shard.state_dict()['transformer_blocks.0.attn1.norm_q.weight'].float()
d=(mw-rw).abs().max().item()
print(f'norm_q: diff={d:.8f} {"OK" if d<1e-6 else "DIFF!"}')
