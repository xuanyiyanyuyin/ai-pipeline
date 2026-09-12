import sys, io
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import torch
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
print('prompt tokens:', len(ids), flush=True)

out_lines = []
with torch.inference_mode():
    # 全量 prefill
    logits_full, cache = eng.forward(ids_t)
    # 贪心取下一 token
    nxt = logits_full[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    nxt_id = nxt.item()
    # 全量重算: prompt + nxt
    ids2 = torch.cat([ids_t, nxt], dim=-1)
    logits_full2, _ = eng.forward(ids2)
    # 增量: 用 cache 只算 nxt
    logits_inc, _ = eng.forward(nxt, past_key_values=cache)
    diff = (logits_full2[0, -1] - logits_inc[0, -1]).abs().max().item()
    out_lines.append('下一 token id=%d (%r)' % (nxt_id, tok.decode([nxt_id])))
    out_lines.append('全量重算 vs 增量 decode 最大 logits 偏差: %.6f' % diff)
    out_lines.append('全量 top5: %s' % logits_full2[0, -1].topk(5).indices.tolist())
    out_lines.append('增量 top5: %s' % logits_inc[0, -1].topk(5).indices.tolist())

io.open(r'D:\ai\llamacpp\_incr_result.txt', 'w', encoding='utf-8').write('\n'.join(out_lines))
print('done', flush=True)
