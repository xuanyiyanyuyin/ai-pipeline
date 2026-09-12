"""合并 FACT.md + JOURNAL.jsonl 到完整记忆库 txt (不带删减)"""
import json, datetime

MEM_DIR = r'C:\Users\Jinqi Wei\AppData\Roaming\CherryStudio\Data\Agents\t-default\memory'
OUT = r'D:\ai\完整记忆库_2026-08-13.txt'

with open(MEM_DIR + r'\FACT.md', encoding='utf-8') as f:
    fact = f.read()

with open(MEM_DIR + r'\JOURNAL.jsonl', encoding='utf-8') as f:
    journal_lines = f.readlines()

# 时间戳转北京时间 (UTC+8)
def to_beijing(ts):
    dt = datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
    return (dt + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')

journal_entries = []
for line in journal_lines:
    line = line.strip()
    if not line: continue
    try:
        e = json.loads(line)
        journal_entries.append({
            'time': to_beijing(e['ts']),
            'tags': ','.join(e.get('tags',[])),
            'text': e['text']
        })
    except: pass

parts = []
parts.append('='*70)
parts.append('完整记忆库 — 汇总导出')
parts.append(f'导出时间: {to_beijing(datetime.datetime.utcnow().isoformat()+"Z")}')
parts.append(f'来源: FACT.md(长期记忆) + JOURNAL.jsonl(事件日志 {len(journal_entries)} 条)')
parts.append('='*70)
parts.append('')
parts.append('#'*70)
parts.append('# 第一部分: FACT.md (长期记忆/持久知识)')
parts.append('#'*70)
parts.append('')
parts.append(fact)
parts.append('')
parts.append('#'*70)
parts.append(f'# 第二部分: JOURNAL (事件日志, {len(journal_entries)} 条, 时间升序)')
parts.append('#'*70)
parts.append('')
for i, e in enumerate(journal_entries, 1):
    parts.append('-'*60)
    parts.append(f'[{i:02d}] {e["time"]}  tags: {e["tags"]}')
    parts.append(e['text'])
    parts.append('')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(parts))

print(f'已生成: {OUT}')
print(f'FACT.md: {len(fact)} 字符, JOURNAL: {len(journal_entries)} 条')
import os
print(f'文件大小: {os.path.getsize(OUT)/1024:.0f} KB')
