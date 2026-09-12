# -*- coding: utf-8 -*-
"""临时: 独立后台跑 FLUX 生成喵朵立绘(提示词写死在文件里, 避免命令行中文乱码)"""
import sys
sys.path.insert(0, r'D:\ai')
import torch
from diffusers import FluxPipeline

MODEL = r'D:\ai\models\FLUX.1-dev'
PROMPT = ("喵朵, 银白色长发, 粉色发尾渐变, 粉尖猫耳, 蓝色大眼睛, "
          "蓝白色女仆装, 白色围裙搭配蓝色连衣裙, 猫尾巴, 站姿, 全身立绘, "
          "柔光背景, 干净简洁, 精致二次元, 柔和高光, 清新治愈, "
          "高质量插画, 竖版构图, 突出人物")
STEPS = 50
W, H = 832, 1248

import time, os
t0 = time.perf_counter()
pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'[loaded] {time.perf_counter()-t0:.0f}s', flush=True)
pipe.enable_sequential_cpu_offload(device='cuda')
print('[offload ready] start denoising', flush=True)

res = pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=3.5,
           height=H, width=W, max_sequence_length=4096,
           generator=torch.Generator('cpu').manual_seed(int(time.time()) % 100000))
img = res.images[0]
fn = f'D:/ai/output/images/miao_{int(time.time())}.png'
img.save(fn)
print(f'[done] {fn} {img.size} in {time.perf_counter()-t0:.0f}s', flush=True)
