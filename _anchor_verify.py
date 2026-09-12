import os, torch, json
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None
import transformers.integrations.finegrained_fp8 as _ff8
import importlib.util
_KERNEL_DIR = r'C:/Users/Jinqi Wei/.cache/huggingface/hub/kernels--kernels-community--finegrained-fp8/snapshots/7cdb05d472d6c954c7d03182ed836ebfd4610df0/build/torch-cuda'
spec = importlib.util.spec_from_file_location('_k', _KERNEL_DIR+r'/__init__.py', submodule_search_locations=[_KERNEL_DIR])
m = importlib.util.module_from_spec(spec); import sys; sys.modules['_k']=m; spec.loader.exec_module(m)
_ff8.lazy_load_kernel = lambda *a, **k: m
from transformers import AutoTokenizer, AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

MODEL = r'D:\ai_v4flash'
cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
cfg.num_hidden_layers = 1
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
print('加载官方 2 层 (device_map=auto, cpu offload)...', flush=True)
model = DeepseekV4ForCausalLM.from_pretrained(MODEL, config=cfg, device_map='auto',
    max_memory={0:'4GB','cpu':'24GB','disk':'50GB'}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
ids = tok('你好，请介绍一下你自己', return_tensors='pt').input_ids
with torch.inference_mode():
    out = model(ids).logits
nxt = out[0,-1].argmax().item()
print(f'官方 2层 首token: {tok.decode([nxt])!r} (id={nxt})', flush=True)
print(f'官方 2层 logits std={out.float().std().item():.4f} top5={[tok.decode([i]) for i in out[0,-1].topk(5).indices.tolist()]}', flush=True)
