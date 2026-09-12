import sys, io, os
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
from diffusers import LTXVideoTransformer3DModel
# 分片加载
tf_shard=LTXVideoTransformer3DModel.from_pretrained(r'D:\ai\models\LTX-Video\transformer_13b_shards',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
# single-file 加载 (如果能)
try:
    tf_sf=LTXVideoTransformer3DModel.from_single_file(r'D:\ai\models\LTX-Video\ltxv-13b-0.9.8-dev.safetensors',torch_dtype=torch.bfloat16,config=r'D:\ai\ltx_config',low_cpu_mem_usage=True)
    # 对比 proj_in
    d=(tf_shard.proj_in.weight.float()-tf_sf.proj_in.weight.float()).abs().max().item()
    print('proj_in diff:', d)
    d2=(tf_shard.state_dict()['transformer_blocks.0.attn1.to_q.weight'].float()-tf_sf.state_dict()['transformer_blocks.0.attn1.to_q.weight'].float()).abs().max().item()
    print('attn1.to_q diff:', d2)
    # 全部 key 对比
    sd1=tf_shard.state_dict(); sd2=tf_sf.state_dict()
    maxd=0; bad=[]
    for k in sd1:
        if k in sd2 and sd1[k].shape==sd2[k].shape:
            d=(sd1[k].float()-sd2[k].float()).abs().max().item()
            if d>maxd: maxd=d
        elif k not in sd2: bad.append(('missing',k))
        elif sd1[k].shape!=sd2[k].shape: bad.append(('shape',k,sd1[k].shape,sd2[k].shape))
    print('max diff all keys:', maxd)
    print('mismatch keys:', bad[:5], 'count', len(bad))
except Exception as e:
    print('single-file load ERR:', str(e)[:150])
