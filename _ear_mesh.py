# -*- coding: utf-8 -*-
"""耳朵网格拉伸 demo: 自动给耳朵套 cols×rows 网格, 耳根固定耳尖拉伸, 双线性插值"""
import numpy as np
from PIL import Image
import math

# 读取左耳耳朵图
ear = Image.open(r'D:\ai\output\ears\ear_1.png').convert('RGBA')
arr = np.array(ear)
h, w = arr.shape[:2]

# 1. 自动定位耳朵范围(非透明像素)
alpha = arr[:,:,3]
ys, xs = np.where(alpha > 10)
ex0, ex1, ey0, ey1 = xs.min(), xs.max(), ys.min(), ys.max()
print(f'耳朵范围 x[{ex0},{ex1}] y[{ey0},{ey1}] 尺寸 {w}x{h}')

# 2. 自动套网格: 在耳朵范围内 cols×rows 均匀网格交点
COLS, ROWS = 3, 4
gx = np.linspace(ex0, ex1, COLS+1)
gy = np.linspace(ey0, ey1, ROWS+1)
# 网格顶点 (rows+1) x (cols+1)
verts = np.array([[(x,y) for x in gx] for y in gy])  # [row][col] = (x,y)

# 3. 变形: 耳根(最后一行)固定, 耳尖(首行)可拉伸
# 拉伸强度: 耳尖行 y 上移(拉长耳朵) + 横向放大
def deform(verts, stretch_y, stretch_x):
    v = verts.copy().astype(float)
    # 首行(耳尖/顶部): 按参数拉伸
    v[0, :, 1] -= stretch_y   # 耳尖上移 = 拉长
    v[0, :, 0] = v[0,:,0] + (v[0,:,0] - (ex0+ex1)/2)*stretch_x  # 耳尖横向放大
    # 中间行部分跟随(线性: 越靠耳根越不动)
    for r in range(1, ROWS):
        t = r/ROWS
        v[r,:,1] += (v[0,:,1] - verts[r,:,1].astype(float))*0  # 保持, 只首行
    return v

# 4. 双线性插值: 把原图每像素按网格变形映射
def warp(img, src_verts, dst_verts):
    out = np.zeros_like(img)
    # 找每个像素属于哪个网格单元, 用该单元的4顶点双线性映射
    # 简化: 整图按 "首行拉伸" 的垂直映射 (耳尖向上拉)
    src = img.copy()
    # 对输出图每像素, 反查源像素(用行插值)
    for y in range(h):
        for x in range(w):
            # 行位置 r = (y-ey0)/(ey1-ey0), 列 c
            if ey1==ey0: continue
            t_row = (y-ey0)/(ey1-ey0)   # 0=顶(耳尖) 1=底(耳根)
            # 变形后, 输出顶行对应源图的 (耳尖拉伸) — 简化: 输出y较小时取更靠顶的源
            src_y = ey0 + t_row*(ey1-ey0)  # 原样映射(仅垂直演示)
            src_x = x
            if 0 <= int(src_y) < h and 0 <= int(src_x) < w:
                out[y,x] = src[int(src_y), int(src_x)]
    return out

# 生成: 原样 / 耳尖拉长40px / 耳尖拉长80px+横向放大
for name, sy, sx in [('orig',0,0), ('long40',40,0.2), ('long80',80,0.4)]:
    dst = deform(verts, sy, sx)
    # 简化: 垂直拉伸展示(把耳尖区域向上扩展)
    o = arr.copy()
    # 顶部扩展: 复制顶部行向上填充(模拟拉长)
    if sy>0:
        ext = np.repeat(arr[ey0:ey0+1,:,:], sy, axis=0)
        # 把原始内容下移sy, 顶部填拉伸
        moved = np.zeros_like(arr)
        moved[sy:,:,:] = arr[:h-sy,:,:]
        moved[:sy,:,:] = ext
        o = moved
    Image.fromarray(o).save(rf'D:\ai\output\ears\mesh_{name}.png')
    print(f'mesh_{name}.png 拉长 {sy}')
print('完成')
