import sys, io, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import v4flash_sched as vs
from transformers import AutoTokenizer
from encoding_dsv4 import encode_messages

eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
DEVICE = vs.DEVICE

TOOLS = [{'type': 'function', 'function': {'name': 'get_time', 'description': '获取时间', 'parameters': {'type': 'object', 'properties': {}}}}]
msgs = [{'role': 'system', 'content': '你是喵朵', 'tools': TOOLS}, {'role': 'user', 'content': '现在几点'}]
prompt = encode_messages(msgs, thinking_mode='chat')
ids = tok.encode(prompt, add_special_tokens=False)
ids_t = torch.tensor([ids], device=DEVICE)

# 收集每层输出的 h (bf16)
def run_and_dump(input_ids, past_key_values=None):
    # 复用 forward 的逻辑, 手动跑并 dump
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.cache_utils import DynamicCache
    import torch.nn.functional as F
    cfg = eng.cfg
    B, S = input_ids.shape
    if past_key_values is None:
        cache = DynamicCache(config=cfg)
        position_ids = torch.arange(S, device=DEVICE).unsqueeze(0)
        pkv_for_mask = None
    else:
        cache = past_key_values
        past_len = cache.get_seq_length()
        position_ids = torch.tensor([[past_len]], device=DEVICE)
        pkv_for_mask = cache
    h = F.embedding(input_ids, eng.embed_w)
    h = h.unsqueeze(2).expand(B, S, cfg.hc_mult, -1).contiguous()
    h_ref = h[:, :, 0, :]
    pe = {'main': eng.rotary_emb(h_ref, position_ids=position_ids, layer_type='main'),
          'compress': eng.rotary_emb(h_ref, position_ids=position_ids, layer_type='compress')}
    causal_mask = create_sliding_window_causal_mask(cfg, inputs_embeds=h_ref, attention_mask=None,
                                                    past_key_values=pkv_for_mask, position_ids=position_ids)
    hs = []
    for lid in range(cfg.num_hidden_layers):
        if lid > 0 and lid % 1 == 0:
            pass
        layer = eng.build_and_inject(lid).to(DEVICE)
        h = layer(h, position_embeddings=pe, position_ids=position_ids,
                  attention_mask=causal_mask, input_ids=input_ids, past_key_values=cache)
        if lid in (0, 1, 2):
            hs.append((lid, h[:, :, 0, :].detach().clone()))
        layer.cpu()
    return hs, cache, causal_mask

with torch.inference_mode():
    # A: 全量 260
    logits_259, _ = eng.forward(ids_t)
    nxt = logits_259[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    ids2 = torch.cat([ids_t, nxt], dim=-1)
    hs_a, _, mask_a = run_and_dump(ids2, None)

    # B: 259 prefill + 1 decode
    _, cache_b = eng.forward(ids_t)
    hs_b, _, mask_b = run_and_dump(nxt, cache_b)

out = []
out.append('mask A type: %s' % type(mask_a).__name__)
out.append('mask B type: %s' % type(mask_b).__name__)
for (lid, ha), (_, hb) in zip(hs_a, hs_b):
    d = (ha[:, -1, :] - hb[:, -1, :]).abs().max().item()
    out.append('layer %d: h 最后 token 偏差 = %.6f' % (lid, d))

io.open(r'D:\ai\llamacpp\_h_result.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
