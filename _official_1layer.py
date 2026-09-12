import os, torch
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None
import importlib.util
_KD = r'C:/Users/Jinqi Wei/.cache/huggingface/hub/kernels--kernels-community--finegrained-fp8/snapshots/7cdb05d472d6c954c7d03182ed836ebfd4610df0/build/torch-cuda'
spec = importlib.util.spec_from_file_location('_k', _KD+'/__init__.py', submodule_search_locations=[_KD])
m = importlib.util.module_from_spec(spec); import sys; sys.modules['_k']=m; spec.loader.exec_module(m)
import transformers.integrations.finegrained_fp8 as _ff8
_ff8.lazy_load_kernel = lambda *a, **k: m
from transformers import AutoTokenizer, AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
MODEL = r'D:\ai_v4flash'
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True); cfg.num_hidden_layers = 1
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
# 用 safetensors 非 mmap 后端
model = DeepseekV4ForCausalLM.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True)
model.eval()
ids = tok('你好，请介绍一下你自己', return_tensors='pt').input_ids
with torch.inference_mode():
    out = model(ids).logits
print('官方1层 首token top8:', [tok.decode([i]) for i in out[0,-1].topk(8).indices.tolist()], flush=True)
print('官方1层 logits std:', round(out.float().std().item(),3), flush=True)
