# -*- coding: utf-8 -*-
"""可控基准: 固定 prompt + 贪心 + 固定长度, 报 秒/步 与 token/秒 (排除 MTP 抽签干扰)"""
import json
import re
import time
import urllib.request

URL = "http://127.0.0.1:8123/v1/chat/completions"
LOG = "/workspace/v4f_server_g.log"


def ask(mt=64):
    body = json.dumps({"model": "v4flash",
                       "messages": [{"role": "user", "content": "请从1一直数到200，只输出数字"}],
                       "stream": False, "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode("utf-8", "replace"))
        txt = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        txt = "ERR " + str(e)[:60]
    return time.time() - t0, len(txt)


def steps():
    """从日志数出本轮生成了多少个 token (每 4 个一条)"""
    try:
        s = open(LOG, 'rb').read().decode('utf-8', 'replace')
    except Exception:
        return 0, 0
    tok = [int(m) for m in re.findall(r'\[gen\] token (\d+) done', s)]
    mtp = [int(m) for m in re.findall(r'MTP\+(\d+)', s)]
    return (max(tok) if tok else 0), (max(mtp) if mtp else 0)


print("=== 可控基准: 3 次相同请求 (贪心, 64 token) ===", flush=True)
for i in range(3):
    b_tok, b_mtp = steps()
    dt, nchar = ask(64)
    a_tok, a_mtp = steps()
    gen = a_tok - b_tok          # 主路径 token 增量
    extra = a_mtp - b_mtp        # MTP 额外产出
    total = gen + extra
    print("  第%d次: 墙钟 %6.1fs  主路径%3d + MTP%3d = %3d token  →  %.3f s/token, %.2f tok/s"
          % (i + 1, dt, gen, extra, total, dt / max(total, 1), total / dt), flush=True)

# 再取一次 status 看 moe 累计
try:
    s = json.loads(urllib.request.urlopen("http://127.0.0.1:8123/api/status", timeout=20).read().decode())
    print("status.exp.tm =", s.get('exp', {}).get('tm'), flush=True)
except Exception as e:
    print("status err", str(e)[:60], flush=True)
