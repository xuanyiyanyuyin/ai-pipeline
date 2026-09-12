import sys, io, os, gc
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import T5EncoderModel, T5Tokenizer
# 先 T5 (小), 验证稳定
print('step1: load T5',flush=True)
tok=T5Tokenizer.from_pretrained(r'D:\ai\models\LTX-Video\tokenizer',local_files_only=True)
te=T5EncoderModel.from_pretrained(r'D:\ai\models\LTX-Video\text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
print('step2: T5 OK, encode',flush=True)
enc=tok('a bright red apple on a white table',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
with torch.inference_mode():
    emb=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state
print('step3: encode OK',tuple(emb.shape),flush=True)
# 卸载 T5
print('step4: del T5',flush=True)
del te; gc.collect()
torch.cuda.empty_cache()
print('step5: T5 released',flush=True)
# 再加载 transformer
print('step6: load TF',flush=True)
from diffusers import LTXVideoTransformer3DModel
tf=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
print('step7: TF OK',flush=True)
