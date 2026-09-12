# -*- coding: utf-8 -*-
"""耳朵网格变形 demo: 自动套 cols×rows 网格, 耳尖拉伸+弯折, 双线性逆映射插值"""
import numpy as np
from PIL import Image

ear = Image.open(r'D:\ai\output\ears\ear_1.png').convert('RGBA')
arr = np.array(ear)
h, w = arr.shape[:2]
alpha = arr[:,:,3]
ys, xs = np.where(alpha > 10)
ex0, ex1, ey0, ey1 = xs.min(), xs.max(), ys.min(), ys.max()

COLS, ROWS = 4, 6
# 源网格顶点 (rows+1)*(cols+1)
gx = np.linspace(ex0, ex1, COLS+1)
gy = np.linspace(ey0, ey1, ROWS+1)
src_v = np.array([[(x,y) for x in gx] for y in gy])   # [row][col]

def warp(dst_offsets):
    """dst_offsets: (ROWS+1)*(COLS+1) 每个顶点的位移 (dx,dy); 输出变形后图"""
    out = np.zeros_like(arr)
    for y in range(h):
        for x in range(w):
            # 找该像素所在源网格单元 (row,col), 用4顶点双线性反算映射
            # 简化: 按(y,x)落在源网格, 找对应4顶点, 用该单元局部坐标在(变形后)网格插值采样
            # 先定源网格单元索引
            if y < ey0 or y > ey1 or x < ex0 or x > ex1:
                continue
            tr = (y - ey0)/(ey1-ey0) * ROWS
            tc = (x - ex0)/(ex1-ex0) * COLS
            r0, c0 = int(tr)%ROWS, int(tc)%COLS
            fr, fc = tr - int(tr), tc - int(tc)
            # 4顶点 (r0/i0, r0/c1, r1/c0, r1/c1) in dst
            v00 = src_v[r0][c0] + dst_offsets[r0][c0]
            v01 = src_v[r0][c0+1] + dst_offsets[r0][c0+1]
            v10 = src_v[r0+1][c0] + dst_offsets[r0+1][c0]
            v11 = src_v[r0+1][c0+1] + dst_offsets[r0+1][c0+1]
            # 双线性: 该像素在源单元局部坐标(fr,fc) → 在dst单元的对应位置
            srcx = (1-fr)*((1-fc)*v00[0]+fc*v01[0]) + fr*((1-fc)*v10[0]+fc*v11[0])
            srcy = (1-fr)*((1-fc)*v00[1]+fc*v01[1]) + fr*((1-fc)*v10[1]+fc*v11[1])
            sy, sx = int(round(srcy)), int(round(srcx))
            if 0 <= sy < h and 0 <= sx < w:
                out[y,x] = arr[sy,sx]
    return out

def make_offsets(stretch=0, bend=0):
    """耳尖(首行)所有顶点拉伸(向上stretch)+向右弯(bend); 耳根(末行)固定"""
    off = np.zeros((ROWS+1, COLS+1, 2))
    for c in range(COLS+1):
        # 首行(耳尖): 拉长 + 弯
        off[0,c] = [bend*(c/(COLS)), -stretch]
        # 中间行: 部分跟随(越多越靠耳尖跟随越多) 线性
        for r in range(1, ROWS):
            t = 1 - r/ROWS
            off[r,c] = [bend*(c/(COLS))*t*0.5, -stretch*t*0.6]
    return off

for name, st, bd in [('orig',0,0), ('long',20,0), ('bend',10,15), ('longbend',25,20)]:
    out = warp(make_offsets(st, bd))
    Image.fromarray(out).save(rf'D:\ai\output\ears\warp_{name}.png')
    print(f'warp_{name}.png stretch={st} bend={bd}')
print('完成')
