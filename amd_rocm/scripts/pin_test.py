# -*- coding: utf-8 -*-
"""验证: ① 能不能 pin 大内存 ② 忙线程下 pinned 是否比 pageable 快"""
import threading
import time

import torch

print('=== ① pin 内存上限实测 ===', flush=True)
bufs = []
try:
    for i in range(64):                    # 64 × 128MB = 8GB
        b = torch.empty(32 * 1024 * 1024, dtype=torch.float32, pin_memory=True)
        b.fill_(1.0)
        bufs.append(b)
        if (i + 1) % 16 == 0:
            print('  已 pin %.1f GB' % ((i + 1) * 128 / 1024), flush=True)
    print('  ✓ pin 8GB 成功' if False else '  OK: pin 到 %.1f GB' % (len(bufs) * 128 / 1024), flush=True)
except Exception as e:
    print('  ✗ 失败于 %.1f GB: %s' % (len(bufs) * 128 / 1024, str(e)[:160]), flush=True)
del bufs
torch.cuda.empty_cache()

MB = 1024 * 1024
PER = 2.125
stop = [False]


def busy():
    s = 0
    while not stop[0]:
        for i in range(20000):
            s += i * i


def mk(pinned, n=6):
    return [torch.empty(int(PER * MB) // 4, dtype=torch.float32, pin_memory=pinned) for _ in range(n)]


def worker(srcs, dsts, iters, res, i):
    t0 = time.perf_counter()
    for _ in range(iters):
        for a, b in zip(srcs, dsts):
            b.copy_(a, non_blocking=True)
    torch.cuda.synchronize(0)
    res[i] = time.perf_counter() - t0


def bench(tag, srcs, dsts, nthreads, iters=40, load=False):
    res = [0.0] * nthreads
    if load:
        stop[0] = False
        threading.Thread(target=busy, daemon=True).start()
    ths = [threading.Thread(target=worker, args=(srcs, dsts, iters, res, i)) for i in range(nthreads)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    stop[0] = True
    print('  %-26s %.2f 专家/秒  (每专家 %.2f ms)'
          % (tag, nthreads * iters / wall, wall / (nthreads * iters) * 1e3), flush=True)


print('=== ② 忙线程(GIL竞争)下的搬运对比 ===', flush=True)
for pinned in (False, True):
    srcs = mk(pinned)
    dsts = [torch.empty_like(t, device='cuda:0') for t in srcs]
    bench('pageable' if not pinned else 'pinned  ', srcs, dsts, 3, load=True)
    del srcs, dsts
    torch.cuda.empty_cache()
print('done', flush=True)
