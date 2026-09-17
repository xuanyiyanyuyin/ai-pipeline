# -*- coding: utf-8 -*-
"""云端速度实测: 流式按 token 打时间戳 → TTFT / TPOT / tok/s"""
import json, time, urllib.request

URL = "http://127.0.0.1:8123/v1/chat/completions"


def stream_ask(msg, mt=32):
    body = json.dumps({"model": "v4flash", "messages": [{"role": "user", "content": msg}],
                       "stream": True, "max_tokens": mt}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ts, text = [], ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if not p or p == "[DONE]":
                continue
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("done"):
                break
            if "tok" in d:
                ts.append(time.time() - t0)
                text += d["tok"]
    return ts, text


def report(tag, ts, text):
    if not ts:
        print("%-16s 未收到 token 块" % tag)
        return
    ttft = ts[0]
    if len(ts) > 1:
        tpot = (ts[-1] - ts[0]) / (len(ts) - 1)
        print("%-16s TTFT=%6.1fs  块数=%3d  TPOT=%.3fs/token  (%.2f tok/s)  文本%d字  %r"
              % (tag, ttft, len(ts), tpot, 1.0 / tpot, len(text), text[:40]))
    else:
        print("%-16s TTFT=%6.1fs  仅 1 个 token 块  文本%d字 %r" % (tag, ttft, len(text), text[:40]))


# 预热 + 两次测量 (第二次是同实例热态)
for i, q in enumerate(["从1数到30", "从1数到20"]):
    ts, text = stream_ask(q, mt=32)
    report("第%d次(%s)" % (i + 1, q), ts, text)
