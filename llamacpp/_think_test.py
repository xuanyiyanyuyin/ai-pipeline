import sys, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import v4flash_sched as vs
from transformers import AutoTokenizer
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
DEVICE = vs.DEVICE
EOS = '<｜end▁of▁sentence｜>'

msgs = [{'role': 'system', 'content': vs.CHAT_SYSTEM}, {'role': 'user', 'content': '1+1等于几？'}]
prompt = encode_messages(msgs, thinking_mode='thinking')
ids = tok.encode(prompt, add_special_tokens=False)
ids_t = torch.tensor([ids], device=DEVICE)

with torch.inference_mode():
    out = eng.generate(ids_t, max_new_tokens=150, temperature=0.6, top_k=0, top_p=1.0,
                       repetition_penalty=1.0, banned_ids=None)
new_tok = out[0][ids_t.shape[1]:]
text = tok.decode(new_tok, skip_special_tokens=False)
print('生成 token 数:', len(new_tok), flush=True)
print('=== 原始输出 ===', flush=True)
print(repr(text), flush=True)
i = text.find(EOS)
print('EOS 位置:', i, flush=True)
if i >= 0:
    text = text[:i + len(EOS)]
    try:
        parsed = parse_message_from_completion_text(text, 'thinking')
        print('=== parse 成功 ===', flush=True)
        print('reasoning:', repr(parsed.get('reasoning_content'))[:300], flush=True)
        print('content:', repr(parsed.get('content')), flush=True)
    except Exception as e:
        print('parse error:', e, flush=True)
