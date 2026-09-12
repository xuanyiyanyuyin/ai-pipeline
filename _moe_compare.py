import os, torch, json, struct, re
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch.nn.functional as F
import v4flash_sched as vs

eng = vs.V4FlashEngine()
cfg = eng.config
lid = 0

# 专家 3D 打包 (官方 FP8Experts 格式): 读全部 256 专家 pread
def pack_experts(lid):
    n = cfg.n_routed_experts
    w1s, w2s, w3s, s1s, s2s, s3s = [], [], [], [], [], []
    for e in range(n):
        p = f'layers.{lid}.ffn.experts.{e}'
        w1s.append(vs.pread(f'{p}.w1.weight')); s1s.append(vs.pread(f'{p}.w1.scale'))
        w2s.append(vs.pread(f'{p}.w2.weight')); s2s.append(vs.pread(f'{p}.w2.scale'))
        w3s.append(vs.pread(f'{p}.w3.weight')); s3s.append(vs.pread(f'{p}.w3.scale'))
    gate_up = torch.cat([torch.stack(w1s), torch.stack(w3s)], dim=1)   # [256, 4096, 4096]
    gu_scale = torch.cat([torch.stack(s1s), torch.stack(s3s)], dim=1)  # [256, 32, 32]
    down = torch.stack(w2s)                                            # [256, 4096, 2048]
    down_scale = torch.stack(s2s)                                      # [256, 32, 16]
    return gate_up, gu_scale, down, down_scale

# 官方 layer0 (含 FP8Experts)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
from transformers.utils.quantization_config import FineGrainedFP8Config
fp8cfg = FineGrainedFP8Config(**cfg.quantization_config)
with torch.device('meta'):
    off_layer = DeepseekV4DecoderLayer(cfg, 0)
off_layer.config = cfg
replace_with_fp8_linear(off_layer, quantization_config=fp8cfg, pre_quantized=True, modules_to_not_convert=["compressor"])
off_layer.to_empty(device='cpu')

raw = dict(eng.layer_ram.get(0, {}))
sd = vs.convert_layer_sd(raw)
gate_up, gu_s, down, d_s = pack_experts(0)
sd['mlp.experts.gate_up_proj'] = gate_up
sd['mlp.experts.gate_up_proj_scale_inv'] = gu_s
sd['mlp.experts.down_proj'] = down
sd['mlp.experts.down_proj_scale_inv'] = d_s
off_layer.load_state_dict(sd, strict=True, assign=True)
off_layer.eval().cuda()

my_layer = eng.build_and_inject(0).cuda().eval()

torch.manual_seed(0)
x = torch.randn(1, 4, 4096, device='cuda', dtype=torch.bfloat16)
x4 = x.unsqueeze(2).expand(1, 4, 4, 4096).contiguous()
position_ids = torch.arange(4, device='cuda').unsqueeze(0)
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
mask = create_sliding_window_causal_mask(cfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=position_ids)
pe = {'main': eng.rotary_emb(x, position_ids=position_ids, layer_type='main'),
      'compress': eng.rotary_emb(x, position_ids=position_ids, layer_type='compress')}

with torch.inference_mode():
    off_out = off_layer(x4, position_embeddings=pe, position_ids=position_ids, attention_mask=mask, input_ids=torch.randint(0,1000,(1,4),device="cuda"), past_key_values=DynamicCache(config=cfg))
    my_out = my_layer(x4, position_embeddings=pe, position_ids=position_ids, attention_mask=mask, input_ids=torch.randint(0,1000,(1,4),device="cuda"), past_key_values=DynamicCache(config=cfg))

diff = (off_out.float() - my_out.float()).abs()
print(f'官方 vs 我的 layer0: max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f}')
print(f'官方 std={off_out.float().std().item():.4f} 我的 std={my_out.float().std().item():.4f}')
print(f'比值={my_out.float().std().item()/off_out.float().std().item():.3f}')
