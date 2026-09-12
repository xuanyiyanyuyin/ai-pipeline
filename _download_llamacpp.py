"""llama.cpp Windows CUDA 13.3 下载 (RTX 5070 Blackwell sm_120 需 13.x)"""
import os, urllib.request, time, zipfile
BASE = 'https://github.com/ggml-org/llama.cpp/releases/download/b10405'
files = [
    ('llama-b10405-bin-win-cuda-13.3-x64.zip', 'llamacpp/'),
    ('cudart-llama-bin-win-cuda-13.3-x64.zip', 'llamacpp/'),
]
os.makedirs(r'D:\ai\llamacpp', exist_ok=True)
for fn, sub in files:
    url = f'{BASE}/{fn}'
    dst = os.path.join(r'D:\ai\llamacpp', fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
        print(f'{fn} 已存在', flush=True); continue
    print(f'下载 {fn} ...', flush=True)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            with open(dst, 'wb') as f: f.write(data)
            print(f'完成 {fn} {os.path.getsize(dst)/1e6:.0f}MB', flush=True)
            break
        except Exception as e:
            print(f'重试{attempt} {str(e)[:50]}', flush=True)
            time.sleep(3)
# 解压
for fn, sub in files:
    zpath = os.path.join(r'D:\ai\llamacpp', fn)
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(r'D:\ai\llamacpp')
        print(f'解压 {fn}', flush=True)
print('llama.cpp 下载+解压完成', flush=True)
