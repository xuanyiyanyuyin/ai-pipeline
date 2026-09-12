# -*- coding: utf-8 -*-
"""从 cherry web_fetch 落盘的 JSON 里提取关键信息"""
import json, re, sys

p = sys.argv[1]
raw = json.load(open(p, encoding='utf-8'))
try:
    items = json.loads(raw[0]['text'])
except Exception:
    items = raw

KWS = ['gfx1100', 'gfx11', 'RDNA3', 'torch-rocm', 'rocm', 'scaled_mm',
       'fp8_supported', 'is_fp8', 'supported_arch', 'arch_', 'raise', 'not supported']
for it in items:
    print('=' * 72)
    print('TITLE:', str(it.get('title', '?'))[:110])
    c = it.get('content', '') or ''
    print('LEN:', len(c))
    for kw in KWS:
        hits = list(re.finditer(re.escape(kw), c, re.I))
        if not hits:
            continue
        print(f'\n--- [{kw}] {len(hits)} 处 ---')
        for m in hits[:2]:
            s, e = max(0, m.start() - 260), min(len(c), m.end() + 260)
            seg = re.sub(r'\s+', ' ', c[s:e])
            print(f'  ...{seg}...')
