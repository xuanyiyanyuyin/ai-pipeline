import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from t5_loader import load_t5_pread
from transformers import T5Tokenizer
print('loading T5 pread...',flush=True)
te=load_t5_pread(r'D:\ai\models\LTX-Video\text_encoder')
print('T5 loaded, dtype:', next(te.parameters()).dtype, flush=True)
tok=T5Tokenizer.from_pretrained(r'D:\ai\models\LTX-Video\tokenizer',local_files_only=True)
enc=tok('a bright red apple on a white table',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    e=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state
print('encode OK', tuple(e.shape), flush=True)
