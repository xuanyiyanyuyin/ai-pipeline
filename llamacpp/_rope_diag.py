import sys, io, torch
sys.path.insert(0, r'D:\ai')
from transformers import AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

cfg = AutoConfig.from_pretrained(r'D:\ai_v4flash')
rope = DeepseekV4RotaryEmbedding(cfg)
x = torch.zeros(1, 1, cfg.hidden_size, dtype=torch.bfloat16)

out = []
for lt in ['main', 'compress']:
    pid_dec = torch.tensor([[259]])
    cos_dec, sin_dec = rope(x, pid_dec, layer_type=lt)
    pid_full = torch.arange(260).unsqueeze(0)
    cos_full, sin_full = rope(x, pid_full, layer_type=lt)
    d_cos = (cos_full[:, 259, :] - cos_dec[:, 0, :]).abs().max().item()
    d_sin = (sin_full[:, 259, :] - sin_dec[:, 0, :]).abs().max().item()
    out.append('%s: cos偏差=%.6f sin偏差=%.6f' % (lt, d_cos, d_sin))

io.open(r'D:\ai\llamacpp\_rope_result.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
