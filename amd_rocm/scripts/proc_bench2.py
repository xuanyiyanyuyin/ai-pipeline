# -*- coding: utf-8 -*-
"""4 进程独立计算图 决定性测试 (spawn, 带超时/异常回传)
   用法: python proc_bench2.py <进程数>
"""
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault('V4F_MODEL_DIR', '/v4flash_fp4')
os.environ.setdefault('V4F_BASE', '/workspace')
os.environ.setdefault('V4F_SRC_DIR', '/workspace/engine/_v4flash_src')

N1, K1 = 2048, 4096
N2, K2 = 4096, 2048
ITERS = 100


def worker(d, q, iters):
    try:
        import sys as _sys
        _sys.path.insert(0, '/workspace/engine')      # ★ spawn 子进程需要
        import torch
        import v4flash_chunked_sched as vs
        torch.cuda.set_device(d)
        dev = f'cuda:{d}'
        w1 = torch.zeros(N1, K1 // 2, dtype=torch.int8, device=dev)
        s1 = torch.zeros(N1, K1 // 32, dtype=torch.uint8, device=dev)
        w3 = torch.zeros(N1, K1 // 2, dtype=torch.int8, device=dev)
        s3 = torch.zeros(N1, K1 // 32, dtype=torch.uint8, device=dev)
        w2 = torch.zeros(N2, K2 // 2, dtype=torch.int8, device=dev)
        s2 = torch.zeros(N2, K2 // 32, dtype=torch.uint8, device=dev)
        x = (torch.randn(1, K1) * 0.1).to(torch.bfloat16).to(dev)

        def one():
            g = vs._triton_mxfp4_matmul(x, w1, s1)
            u = vs._triton_mxfp4_matmul(x, w3, s3)
            f = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
            return vs._triton_mxfp4_matmul(f, w2, s2)

        for _ in range(5):
            one()
        torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize(d)
        q.put(('ok', d, (time.perf_counter() - t0) / iters * 1e3))
    except Exception as e:
        q.put(('err', d, str(e)[:200]))


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(i, q, ITERS)) for i in range(n)]
    for p in ps:
        p.start()
    got = []
    for _ in range(n):
        try:
            got.append(q.get(timeout=420))
        except Exception as e:
            got.append(('timeout', -1, str(e)[:80]))
    for p in ps:
        p.join(timeout=30)
    oks = [g for g in got if g[0] == 'ok']
    if oks:
        each = max(g[2] for g in oks)
        print('  %d 进程: 每卡 %.3f ms/专家 → 聚合 %.0f 专家/秒  %s'
              % (n, each, len(oks) / (each / 1e3), ' '.join('cuda%d=%.3f' % (g[1], g[2]) for g in sorted(oks, key=lambda z: z[1]))), flush=True)
    else:
        print('  %d 进程: 全部失败 %s' % (n, got), flush=True)
