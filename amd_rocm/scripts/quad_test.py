# -*- coding: utf-8 -*-
"""4 实例并发测试: 同时向 4 个端口各发一个 64-token 请求, 量总吞吐"""
import json
import threading
import time
import urllib.request

PORTS = [8123, 8124, 8125, 8126]
STEPS = 64
res = {}


def ask(port):
    body = json.dumps({"model": "v4flash",
                       "messages": [{"role": "user", "content": "请从1一直数到200，只输出数字"}],
                       "stream": False, "max_tokens": STEPS, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode("utf-8", "replace"))
        txt = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        txt = "ERR " + str(e)[:60]
    res[port] = (time.time() - t0, len(txt), txt[:30].replace("\n", " "))


print('=== 4 实例并发: 每个实例一个 64-token 请求 ===', flush=True)
ths = [threading.Thread(target=ask, args=(p,)) for p in PORTS]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.perf_counter() - t0
tot = 0
for p in PORTS:
    if p in res:
        dt, n, s = res[p]
        print('  端口 %d: %.1fs, %d 字 | %s' % (p, dt, n, s), flush=True)
        tot += 1
print('  4 路并发墙钟 %.1fs (各自单跑约 60-70s) → 并发加速 ≈ %.2fx' % (wall, 4 * 65 / wall), flush=True)
print('[quad] DONE', flush=True)
