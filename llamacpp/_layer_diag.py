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

def safe_max(t):
    if t is None or t.numel() == 0:
        return None
    return t.abs().max().item()

out = []
with torch.inference_mode():
    # 259 prefill
    logits_259, cache_259 = eng.forward(ids_t)
    nxt = logits_259[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    nxt_id = nxt.item()
    # 260 全量 prefill
    ids2 = torch.cat([ids_t, nxt], dim=-1)
    _, cache_full = eng.forward(ids2)
    # 259 prefill + 1 decode
    _, cache_inc = eng.forward(nxt, past_key_values=cache_259)

    out.append('nxt id=%d (%r)' % (nxt_id, tok.decode([nxt_id])))
    for lid in range(eng.cfg.num_hidden_layers):
        la = cache_full.layers[lid]
        lb = cache_inc.layers[lid]
        diffs = []
        if hasattr(la, 'keys') and hasattr(lb, 'keys'):
            d = safe_max(la.keys - lb.keys) if (la.keys is not None and lb.keys is not None) else None
            if d is not None and d > 0.01:
                diffs.append('keys=%.4f' % d)
        for name in ('compressor', 'indexer'):
            ca = la.compressed_kv.get(name) if hasattr(la, 'compressed_kv') else None
            cb = lb.compressed_kv.get(name) if hasattr(lb, 'compressed_kv') else None
            if ca is not None and cb is not None:
                d = safe_max(ca - cb)
                if d is not None and d > 0.01:
                    diffs.append('%s.kv=%.4f' % (name, d))
            elif ca is not None or cb is not None:
                diffs.append('%s.kv 形状不一致(%s vs %s)' % (name, getattr(ca, 'shape', None), getattr(cb, 'shape', None)))
        if hasattr(la, 'entry_count') and hasattr(lb, 'entry_count'):
            if la.entry_count != lb.entry_count:
                diffs.append('entry_count=%s vs %s' % (la.entry_count, lb.entry_count))
        if hasattr(la, 'buffer_kv') and hasattr(lb, 'buffer_kv'):
            for name in la.buffer_kv:
                ba = la.buffer_kv[name]; bb = lb.buffer_kv[name]
                if ba is not None and bb is not None:
                    d = safe_max(ba - bb)
                    if d is not None and d > 0.01:
                        diffs.append('buffer[%s]=%.4f' % (name, d))
                elif ba is not None or bb is not None:
                    diffs.append('buffer[%s] 不一致' % name)
        if diffs:
            out.append('layer %d: %s' % (lid, ' | '.join(diffs)))

io.open(r'D:\ai\llamacpp\_layer_result.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
