# -*- coding: utf-8 -*-
"""耳朵变形demo: 立绘 + 双耳, 支持倾斜(rotate)/抖动(wiggle), 输出效果图"""
from psd_tools import PSDImage
import numpy as np
from PIL import Image
import math

psd = PSDImage.open(r'D:\ai\output\images\image_178881725249173.psd')
# 立绘背景(图层0)
body = np.array(psd[0].composite().convert('RGBA'))
H, W = body.shape[:2]

# 两只耳朵: 图层1(off 719,13), 图层2(off 596,34), 各带耳根锚点
ears = [
    {'img': psd[1].composite().convert('RGBA'), 'off': (719,13), 'anchor': (45,71), 'name':'L'},  # 左耳, 耳根在图层内(45,71)
    {'img': psd[2].composite().convert('RGBA'), 'off': (596,34), 'anchor': (19,68), 'name':'R'},  # 右耳
]

def rotate_about(img, anchor, angle):
    """绕 anchor 点旋转 angle 度, 返回同尺寸RGBA"""
    from PIL import Image as I
    ang = math.radians(angle)
    ca, sa = math.cos(ang), math.sin(ang)
    out = I.new('RGBA', img.size, (0,0,0,0))
    src = img.convert('RGBA')
    sa_arr = np.array(src)
    ox, oy = anchor
    # 逐像素旋转(锚点不动)
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            dx, dy = x-ox, y-oy
            nx = int(ox + dx*ca - dy*sa)
            ny = int(oy + dx*sa + dy*ca)
            if 0 <= nx < img.size[0] and 0 <= ny < img.size[1]:
                out.putpixel((x,y), tuple(sa_arr[ny,nx]))
    return out

def place(img, body, off, angle=0):
    """把(可能旋转的)耳朵叠加到 body 的 off 位置"""
    e = img if angle==0 else rotate_about(img, (img.size[0]//2, img.size[1]), angle)
    ex, ey = off
    ea = np.array(e)
    for y in range(e.size[1]):
        for x in range(e.size[0]):
            if ea[y,x,3] > 0:
                gx, gy = ex+x, ey+y
                if 0 <= gx < W and 0 <= gy < H:
                    body[gy,gx] = ea[y,x]
    return body

# 生成 0°(原样) / +15° / -15° 三个倾斜效果
for ang, name in [(0,'0'), (15,'p15'), (-15,'n15')]:
    b = body.copy()
    for ear in ears:
        b = place(ear['img'], b, ear['off'], ang)
    Image.fromarray(b).save(rf'D:\ai\output\ears\demo_{name}.png')
    print(f'demo_{name}.png 角度{ang} 完成')
print('全部生成')
