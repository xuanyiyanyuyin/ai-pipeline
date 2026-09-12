import sys, io, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import v4flash_sched as vs
from transformers import AutoTokenizer
from encoding_dsv4 import encode_messages
import torch.nn.functional as F
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

eng = vs.V4FlashEngine()
# 强制 eager 注意力(精确, 无 flash 分块)
eng.cfg._attn_implementation = 'eager'
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
DEVICE = vs.DEVICE

TOOLS = [{'type': 'function', 'function': {'name': 'get_time', 'description': 't', 'parameters': {'type': 'object', 'properties': {}}}}]
msgs = [{'role': 'system', 'content': '你是喵朵', 'tools': TOOLS}, {'role': 'user', 'content': '现在几点'}]
prompt = encode_messages(msgs, thinking_mode='chat')
ids = tok.encode(prompt, add_special_tokens=False)
ids_t = torch.tensor([ids], device=DEVICE)

def run_l0(input_ids, past_key_values=None):
    cfg = eng.cfg
    B, S = input_ids.shape
    if past_key_values is None:
        cache = DynamicCache(config=cfg)
        position_ids = torch.arange(S, device=DEVICE).unsqueeze(0)
        pkv_for_mask = None
    else:
        cache = past_key_values
        position_ids = torch.tensor([[cache.get_seq_length()]], device=DEVICE)
        pkv_for_mask = cache
    h = F.embedding(input_ids, eng.embed_w)
    h = h.unsqueeze(2).expand(B, S, cfg.hc_mult, -1).contiguous()
    h_ref = h[:, :, 0, :]
    pe = {'main': eng.rotary_emb(h_ref, position_ids=position_ids, layer_type='main'),
          'compress': eng.rotary_emb(h_ref, position_ids=position_ids, layer_type='compress')}
    mask = create_sliding_window_causal_mask(cfg, inputs_embeds=h_ref, attention_mask=None,
                                             past_key_values=pkv_for_mask, position_ids=position_ids)
    layer = eng.build_and_inject(0).to(DEVICE)
    h = layer(h, position_embeddings=pe, position_ids=position_ids,
              attention_mask=mask, input_ids=input_ids, past_key_values=cache)
    layer.cpu()
    return h[:, :, 0, :].detach(), cache

out = []
with torch.inference_mode():
    logits, _ = eng.forward(ids_t)
    nxt = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    ids2 = torch.cat([ids_t, nxt], dim=-1)
    h_a, _ = run_l0(ids2, None)          # 全量 260
    _, cache_b = eng.forward(ids_t)      # 259 prefill
    h_b, _ = run_l0(nxt, cache_b)        # +1 decode
    d = (h_a[:, -1, :] - h_b[:, -1, :]).abs().max().item()
    out.append('eager 注意力 layer 0 h 偏差 = %.6f' % d)

io.open(r'D:\ai\llamacpp\_eager_result.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
