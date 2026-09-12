import os, urllib.request, time
BASE = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash/resolve/main/'
UA = {'User-Agent': 'Mozilla/5.0'}
shard = 'model-00044-of-00046.safetensors'
dst = r'D:\ai_v4flash\model-00044-of-00046.safetensors'
tmp = dst + '.part'
for attempt in range(10):
    try:
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(BASE + shard, headers={**UA, 'Range': f'bytes={have}-'})
        with urllib.request.urlopen(req, timeout=300) as r:
            total = int(r.headers.get('Content-Length', 0)) + have
            print(f'重试{attempt}: 从{have/1e9:.2f}GB续传, 总{total/1e9:.2f}GB', flush=True)
            with open(tmp, 'ab' if have else 'wb') as f:
                while True:
                    chunk = r.read(16 * 1024 * 1024)
                    if not chunk: break
                    f.write(chunk)
        got = os.path.getsize(tmp)
        if total > 0 and got == total:
            os.replace(tmp, dst)
            print(f'✓ 完整 {got/1e9:.2f}GB', flush=True)
            break
        else:
            print(f'✗ 不完整 {got/1e9:.2f}/{total/1e9:.2f}GB, 重试', flush=True)
    except Exception as e:
        print(f'✗ 异常 {str(e)[:60]}', flush=True)
    time.sleep(3)
