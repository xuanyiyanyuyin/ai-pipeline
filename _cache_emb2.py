import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import T5EncoderModel, T5Tokenizer
MODEL=r'D:\ai\models\LTX-Video'
tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
enc=tok('a bright red apple on a white table',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
encn=tok('',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    pos=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state.bfloat16()
    neg=te(input_ids=encn.input_ids,attention_mask=encn.attention_mask).last_hidden_state.bfloat16()
torch.save({'pos':pos.cpu(),'neg':neg.cpu(),'pos_mask':enc.attention_mask.bool(),'neg_mask':encn.attention_mask.bool()}, r'D:\ai\_cached_embeds.pt')
print('cached with mask', flush=True)
print('pos_mask sum', enc.attention_mask.sum().item(), 'neg_mask sum', encn.attention_mask.sum().item(), flush=True)
