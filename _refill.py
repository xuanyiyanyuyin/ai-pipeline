import os, urllib.request, time
UA = {'User-Agent': 'Mozilla/5.0'}
BASE = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash/resolve/main/'
files = ['model-00010-of-00046.safetensors', 'model-00044-of-00046.safetensors']
for f in files:
    url = BASE + f
    dst = f'D:/ai_v4flash/{f}'
    tmp = dst + '.part'
    if os.path.exists(tmp): os.remove(tmp)
    for attempt in range(10):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = 'ab' if have > 0 else 'wb'
                with open(tmp, mode) as fh:
                    while True:
                        c = r.read(8*1024*1024)
                        if not c: break
                        fh.write(c)
            os.replace(tmp, dst)
            print(f'✅ {f} 完成 {os.path.getsize(dst)/1e9:.2f}GB', flush=True)
            break
        except Exception as e:
            print(f'  {f} 第{attempt+1}次失败: {str(e)[:60]}', flush=True)
            time.sleep(3)
