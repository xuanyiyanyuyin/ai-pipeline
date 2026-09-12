"""
FLUX.1-dev 魔改生图 — 12GB显存流式版
策略: diffusers官方计算逻辑 + 逐模块CPU卸载(GPU只放当前块, 算完搬回)
用法: D:\ai\python\python.exe gen_image.py "一只在月亮上打太极的橘猫" [steps] [size]
"""
import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
MODEL=r'D:\ai\models\FLUX.1-dev'
OUT=r'D:\ai\output\images'
os.makedirs(OUT, exist_ok=True)

def main():
    prompt=sys.argv[1] if len(sys.argv)>1 else '一只在月亮上打太极的橘猫, 电影级光影'
    steps=int(sys.argv[2]) if len(sys.argv)>2 else 50
    iw=int(sys.argv[3]) if len(sys.argv)>3 else 2560
    ih=int(sys.argv[4]) if len(sys.argv)>4 else 1408
    print(f'[gen] prompt: {prompt}',flush=True)
    print(f'[gen] steps={steps} size={iw}x{ih}',flush=True)

    from diffusers import FluxPipeline
    t0=time.perf_counter()
    pipe=FluxPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True)
    print(f'[gen] pipeline loaded {time.perf_counter()-t0:.0f}s, 启用逐块显存搬运...',flush=True)
    # 魔改核心: 每个子模块(每个transformer块/T5块)用时上GPU, 用完回CPU
    # GPU峰值 = 单个块(~400MB) + 激活, 12GB绰绰有余
    pipe.enable_sequential_cpu_offload(device='cuda')
    print(f'[gen] offload ready, 开始去噪...',flush=True)

    t1=time.perf_counter()
    res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
             height=ih,width=iw,max_sequence_length=4096,
             generator=torch.Generator('cpu').manual_seed(42))
    dt=time.perf_counter()-t1
    img=res.images[0]
    fn=f'{OUT}/flux_{int(time.time())}.png'
    img.save(fn)
    print(f'[gen] done in {dt:.0f}s ({dt/steps:.1f}s/step) -> {fn}',flush=True)
    print(f'[gen] VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB',flush=True)

_pipe=None
def get_pipe():
    """全局单例FLUX pipeline, 避免重复加载"""
    global _pipe
    if _pipe is not None:
        return _pipe
    from diffusers import FluxPipeline
    _pipe=FluxPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True)
    _pipe.enable_sequential_cpu_offload(device='cuda')
    return _pipe

def free_pipe():
    """释放FLUX引擎, 归还显存给122B"""
    global _pipe
    if _pipe is not None:
        del _pipe; _pipe=None
        import torch; torch.cuda.empty_cache()
        print('[gen] FLUX released',flush=True)

# ═══ IP-Adapter 加装(不新建引擎, 复用 _pipe 单例) ═══
_IPCACHE={'ipadapter_loaded':False, 'image_encoder_cpu':None}
IPADAPTER_PATH=r'D:\ai\models\adapter\flux-ip-adapter'
CLIP_VIT_PATH=r'D:\ai\models\adapter\clip-vit-large-patch14'

def add_ipadapter():
    """给已加载的 _pipe 加装 IP-Adapter 权重(一次性, 不新建引擎)"""
    global _IPCACHE
    if _IPCACHE['ipadapter_loaded']:
        return True
    torch=__import__('torch')
    pipe=get_pipe()
    import os
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    pipe.load_ip_adapter(
        IPADAPTER_PATH, weight_name='ip_adapter.safetensors',
        image_encoder_pretrained_model_name_or_path=CLIP_VIT_PATH,
        image_encoder_dtype=torch.bfloat16,
    )
    _IPCACHE['ipadapter_loaded']=True
    print('[gen] IP-Adapter added',flush=True)
    return True

def _encode_ip_reference(img_pil):
    """用 image_encoder 算参考图 embeds. image_encoder 被 offload 到 meta, 无法用;
    故单独加载一份到 CPU 算(只编码1次, 用完释放, 不占显存)"""
    torch=__import__('torch')
    import transformers
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    import torch.nn.functional as F
    # 单独加载 image_encoder (CPU, fp32, 只算embeds)
    enc=CLIPVisionModelWithProjection.from_pretrained(CLIP_VIT_PATH, torch_dtype=torch.bfloat16).to('cpu')
    enc.eval()
    proc=CLIPImageProcessor.from_pretrained(CLIP_VIT_PATH)
    img=img_pil.convert('RGB')
    inputs=proc(images=img, return_tensors='pt')
    inputs={k:v.to('cpu') for k,v in inputs.items()}
    with torch.no_grad():
        embeds=enc(**inputs).image_embeds
    embeds=embeds.unsqueeze(1).to('cpu')   # (1,1,dim)
    del enc
    torch.cuda.empty_cache()
    return embeds

def generate_image(prompt, steps=50, out_path=None, size=2560, height=1408, ip_reference=None):
    """可导入的生图函数(供 server 调用). ip_reference=PIL图 时启用 IP-Adapter 角色一致"""
    from PIL import Image
    t0=time.perf_counter()
    pipe=get_pipe()
    kwargs={}
    if ip_reference is not None:
        add_ipadapter()                      # 加装 IP-Adapter(复用_pipe)
        embeds=_encode_ip_reference(ip_reference)   # 算embeds(用完即释放image_encoder)
        kwargs['ip_adapter_image_embeds']=embeds
    res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
             height=height,width=size,max_sequence_length=4096,
             generator=torch.Generator('cpu').manual_seed(int(time.time())%10000),
             **kwargs)
    img=res.images[0]
    if out_path is None:
        fn=f'{OUT}/chat_{int(time.time())}.png'
    else:
        fn=out_path
    img.save(fn)
    print(f'[gen] {fn} ({img.size}) {time.perf_counter()-t0:.0f}s',flush=True)
    return fn

def generate_batch(prompt, steps=50, count=4, size=2560, height=1408):
    """批量生成 count 张图(不同seed), 返回路径列表"""
    t0=time.perf_counter()
    pipe=get_pipe()
    base=int(time.time())%100000
    paths=[]
    for i in range(count):
        fn=f'{OUT}/gen_{base}_{i}.png'
        res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
                 height=height,width=size,max_sequence_length=4096,
                 generator=torch.Generator('cpu').manual_seed(base+i))
        res.images[0].save(fn)
        paths.append(fn)
        print(f'[gen] {i+1}/{count} {fn} ({time.perf_counter()-t0:.0f}s)',flush=True)
    return paths

if __name__=='__main__':
    main()
