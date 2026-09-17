# -*- coding: utf-8 -*-
"""精确测 P2P: ① 12.75MB(一个专家) 小拷贝往返延迟/带宽 ② 256MB 大拷贝带宽 (各 3 轮)
   目的是决定: 计算卡放 GPU0 还是 GPU1 (仓库卡放另一张)。"""
import torch, time

def bench(direction, nbytes, reps):
    n = nbytes // 4
    src_dev, dst_dev = direction
    src = torch.zeros(n, dtype=torch.float32, device=src_dev)
    dst = torch.zeros(n, dtype=torch.float32, device=dst_dev)
    # 预热
    for _ in range(3):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / reps, nbytes / (dt / reps) / 1e9

print("=== 一个专家的大小 (12.75MB) 拷贝, 各 40 次 ===")
for name, d in (("GPU1->GPU0", ('cuda:1', 'cuda:0')), ("GPU0->GPU1", ('cuda:0', 'cuda:1'))):
    ms, gbs = bench(d, 13_369_344, 40)
    print("  %-10s 每次 %.2f ms  (等效 %.1f GB/s)" % (name, ms * 1e3, gbs))

print("=== 大拷贝 (256MB), 各 3 次 ===")
for name, d in (("GPU1->GPU0", ('cuda:1', 'cuda:0')), ("GPU0->GPU1", ('cuda:0', 'cuda:1'))):
    ms, gbs = bench(d, 268_435_456, 3)
    print("  %-10s %.1f GB/s" % (name, gbs))

print("=== 从主机内存 (pageable) 上到两卡 (对照) ===")
h = torch.zeros(268_435_456 // 4, dtype=torch.float32)
for dev in ('cuda:0', 'cuda:1'):
    d = torch.zeros_like(h, device=dev)
    for _ in range(2):
        d.copy_(h, non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        d.copy_(h, non_blocking=False)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print("  host->%s %.1f GB/s" % (dev, 268_435_456 / dt / 1e9))
