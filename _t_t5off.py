import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import T5EncoderModel, T5Tokenizer
te=T5EncoderModel.from_pretrained(r'D:\ai\models\LTX-Video\text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
print('T5 loaded', flush=True)
tok=T5Tokenizer.from_pretrained(r'D:\ai\models\LTX-Video\tokenizer',local_files_only=True)
enc=tok('a bright red apple',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    e=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state
print('encode OK', tuple(e.shape), flush=True)
