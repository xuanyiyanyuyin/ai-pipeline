"""Qwen3.6 Q4_K_M GGUF 分块多线程下载 — 8并发×512MB块, Range断点续传"""
import os, urllib.request, threading, time
URL = 'https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
dst = r'D:\ai\models\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
os.makedirs(os.path.dirname(dst), exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0'}
# 总大小
req = urllib.request.Request(URL, headers=UA)
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers.get('Content-Length', 0))
print(f'总大小 {total/1e9:.2f}GB', flush=True)
CHUNK = 512 * 1024 * 1024  # 512MB
chunks = [(i, min(i+CHUNK, total)-1) for i in range(0, total, CHUNK)]
parts = [dst + f'.part{i:02d}' for i in range(len(chunks))]
def dl(idx, start, end):
    p = parts[idx]
    if os.path.exists(p) and os.path.getsize(p) == end-start+1:
        return
    for attempt in range(10):
        try:
            req = urllib.request.Request(URL, headers={**UA, 'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                with open(p, 'wb') as f:
                    while True:
                        b = r.read(8*1024*1024)
                        if not b: break
                        f.write(b)
            if os.path.getsize(p) == end-start+1:
                return
        except Exception as e:
            time.sleep(2)
    print(f'块{idx} 失败', flush=True)
with threading.Lock():
    done = 0
    lock = threading.Lock()
    def w(i, s, e):
        global done
        dl(i, s, e)
        with lock:
            global done
            done += 1
            print(f'{done}/{len(chunks)} ({done*512/1024:.1f}GB/{total/1e9:.2f}GB)', flush=True)
    ts = [threading.Thread(target=w, args=(i,s,e)) for i,(s,e) in enumerate(chunks)]
    for t in ts: t.start()
    for t in ts: t.join()
# 拼接
with open(dst, 'wb') as out:
    for p in parts:
        with open(p, 'rb') as f:
            while True:
                b = f.read(16*1024*1024)
                if not b: break
                out.write(b)
        os.remove(p)
print(f'完成 {os.path.getsize(dst)/1e9:.2f}GB', flush=True)
