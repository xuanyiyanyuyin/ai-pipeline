# -*- coding: utf-8 -*-
"""复用原本生图引擎 + 加装 IP-Adapter 测试(不新建引擎)"""
import sys, os
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import gen_image
from PIL import Image

REF=r'D:\ai\output\角色库\miao_1788351147.png'
ref=Image.open(REF).convert('RGB')

print('[A] 调用 get_pipe (原本引擎)...', flush=True)
pipe=gen_image.get_pipe()
print('[B] get_pipe OK, 加装 IP-Adapter...', flush=True)

# 小尺寸测试(避免长时间) + IP参考
out=gen_image.generate_image(
    prompt="同一个银发青和服少女, 站在樱花树下回眸, 手持红玫瑰, 青色和服裙, 精致二次元插画, 柔和高光, 高质量",
    steps=20, size=512, height=512,  # 小尺寸快速验证
    ip_reference=ref,
)
print(f'[C] 完成: {out}', flush=True)
gen_image.free_pipe()
print('[D] 已释放引擎', flush=True)
