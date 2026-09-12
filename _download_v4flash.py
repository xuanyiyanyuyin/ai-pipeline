"""V4-Flash-Base 双盘并发下载 (C盘前14片 + D盘后32片)"""
import sys, json, urllib.request, os

# 获取分片列表
req = urllib.request.Request('https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Flash-Base/repo/files',
                             headers={'User-Agent':'Mozilla/5.0'})
data = json.load(urllib.request.urlopen(req, timeout=20))
files = [f['Path'] for f in data.get('Data',{}).get('Files',[]) if f.get('Type')=='blob']
sts = sorted([f for f in files if f.endswith('.safetensors')])
print(f'共 {len(sts)} 个 safetensors')

# 分片: C盘 前14, D盘 后32
split = 14
c_files = sts[:split]
d_files = sts[split:]
print(f'C盘: {len(c_files)} 片 ({c_files[0]} ... {c_files[-1]})')
print(f'D盘: {len(d_files)} 片 ({d_files[0]} ... {d_files[-1]})')

# 保存分片列表供下载进程用
json.dump({'c':c_files,'d':d_files,'all':sts}, open(r'D:\ai\_v4flash_parts.json','w'))

def download(parts, local_dir, tag):
    from modelscope import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    print(f'[{tag}] 开始下载 {len(parts)} 片到 {local_dir}', flush=True)
    snapshot_download(
        'deepseek-ai/DeepSeek-V4-Flash-Base',
        local_dir=local_dir,
        allow_patterns=parts,  # 只下指定分片
        max_workers=8,
    )
    print(f'[{tag}] 下载完成', flush=True)

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'c':
        download(c_files, r'C:\ai_v4flash', 'C')
    elif which == 'd':
        download(d_files, r'D:\ai_v4flash', 'D')
    else:
        # 模拟列表
        print('用: python _download_v4flash.py c|d')
