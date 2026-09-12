# -*- coding: utf-8 -*-
"""耳朵变形demo v2: 用 PIL rotate 绕耳根转(插值+抗锯齿, 不丢像素)"""
import math
import numpy as np
from psd_tools import PSDImage
from PIL import Image

psd = PSDImage.open(r'D:\ai\output\images\image_178881725249173.psd')
body = np.array(psd[0].composite().convert('RGBA'))
H, W = body.shape[:2]
ear_L = psd[1].composite().convert('RGBA')  # 左耳 off(719,13) anchor(45,71)
ear_R = psd[2].composite().convert('RGBA')  # 右耳 off(596,34) anchor(19,68)

def place(body, ear, off, angle, anchor):
    """绕耳根anchor旋转, 再叠加到 off"""
    # 绕 anchor(图层内坐标) 旋转, BICUBIC 插值抗锯齿
    e = ear.rotate(angle, resample=Image.BICUBIC, center=anchor, expand=False)
    ex, ey = off
    ea = np.array(e)
    # alpha=0 的不叠加(不挡背景)
    mask = ea[:,:,3] > 0
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        gx, gy = ex+x, ey+y
        if 0 <= gx < W and 0 <= gy < H:
            body[gy, gx] = ea[y, x]
    return body

for ang, name in [(0,'0'),(15,'p15'),(-15,'n15'),(30,'p30')]:
    b = body.copy()
    b = place(b, ear_L, (719,13), ang, (45,71))
    b = place(b, ear_R, (596,34), ang, (19,68))
    Image.fromarray(b).save(rf'D:\ai\output\ears\demo2_{name}.png')
    print(f'demo2_{name}.png angle={ang}')
print('完成')
