"""V4-Flash 对话版 双盘并发下载 — 手动多线程 + 断点续传 + 失败重试
8 并发(避免打挂 CDN), 每文件独立线程, Range 断点续传, 失败自动重试 5 次
"""
import os, sys, json, urllib.request, threading, time, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash/resolve/main/'
REPO_API = 'https://hf-mirror.com/api/models/deepseek-ai/DeepSeek-V4-Flash/tree/main'
UA = {'User-Agent': 'Mozilla/5.0'}

def get_files():
    req = urllib.request.Request(REPO_API, headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30))
    sts = sorted([f['path'] for f in data if f.get('type') == 'file' and f['path'].endswith('.safetensors')])
    meta = [f['path'] for f in data if f.get('type') == 'file' and not f['path'].endswith('.safetensors')]
    return sts, meta

def download_one(path, local_dir, retries=5):
    url = BASE + path
    dst = os.path.join(local_dir, path)
    tmp = dst + '.part'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(retries):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = 'ab' if have > 0 else 'wb'
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(tmp, dst)
            return path
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ✗ {path}: {str(e)[:60]}', flush=True)
                return None
            time.sleep(2 * (attempt + 1))
    return None

def run(parts, local_dir, tag, workers=8):
    os.makedirs(local_dir, exist_ok=True)
    total = len(parts)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, p, local_dir): p for p in parts}
        for fut in as_completed(futs):
            done += 1
            if done % 5 == 0 or done == total:
                el = time.time() - t0
                sz = sum(os.path.getsize(os.path.join(local_dir, p)) for p in parts if os.path.exists(os.path.join(local_dir, p)))
                print(f'  [{tag}] {done}/{total} ({sz/1e9:.1f}GB, {sz/1e6/el:.0f}MB/s)', flush=True)
    return done

if __name__ == '__main__':
    sts, meta = get_files()
    split = 14
    c_files, d_files = sts[:split], sts[split:]
    print(f'对话版: C{len(c_files)} + D{len(d_files)} 片 + {len(meta)} 元文件', flush=True)
    # 元文件
    for m in meta:
        download_one(m, r'D:\ai_v4flash')
    print('元文件完成, 双盘并发...', flush=True)
    t1 = threading.Thread(target=run, args=(c_files, r'C:\ai_v4flash', 'C'))
    t2 = threading.Thread(target=run, args=(d_files, r'D:\ai_v4flash', 'D'))
    t1.start(); t2.start(); t1.join(); t2.join()
    # 合并 C 到 D
    for f in os.listdir(r'C:\ai_v4flash'):
        src = os.path.join(r'C:\ai_v4flash', f); dst = os.path.join(r'D:\ai_v4flash', f)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    print('✅ 下载+合并完成', flush=True)
