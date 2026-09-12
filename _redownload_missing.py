"""补下缺失/不完整的 shard — 带完整性校验(下载后比对预期大小), 失败重试"""
import os, json, struct, urllib.request, time
BASE = 'https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash/resolve/main/'
UA = {'User-Agent': 'Mozilla/5.0'}
DIR = r'D:\ai_v4flash'

def expected_size(shard):
    """从 index.json 算 shard 预期大小"""
    idx = json.load(open(os.path.join(DIR, 'model.safetensors.index.json'), encoding='utf-8'))
    wm = idx['weight_map']
    # 该 shard 里所有 tensor 的最大 data_offsets[1]
    with open(os.path.join(DIR, shard), 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
        max_end = max(m['data_offsets'][1] for m in hdr.values() if isinstance(m, dict) and 'data_offsets' in m)
    return 8 + hs + max_end

def download_verified(shard, retries=8):
    url = BASE + shard
    dst = os.path.join(DIR, shard)
    exp = expected_size(shard)
    for attempt in range(retries):
        try:
            tmp = dst + '.part'
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=300) as r:
                total = int(r.headers.get('Content-Length', 0)) + have
                mode = 'ab' if have > 0 else 'wb'
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            # 校验: 文件大小必须等于预期
            got = os.path.getsize(tmp)
            if got == exp:
                os.replace(tmp, dst)
                print(f'✓ {shard} 完整 {got/1e9:.2f}GB', flush=True)
                return True
            else:
                print(f'  ✗ {shard} 不完整 {got/1e9:.2f}GB vs 预期 {exp/1e9:.2f}GB, 重试...', flush=True)
                os.remove(tmp)
        except Exception as e:
            print(f'  ✗ {shard} 异常 {str(e)[:60]}, 重试...', flush=True)
        time.sleep(3 * (attempt + 1))
    return False

if __name__ == '__main__':
    missing = ['model-00010-of-00046.safetensors', 'model-00044-of-00046.safetensors']
    for s in missing:
        download_verified(s)
    print('补下完成', flush=True)
