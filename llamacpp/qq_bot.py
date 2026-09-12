"""QQ 机器人适配层 (OneBot v11 协议)
兼容 NapCat / LLOneBot / Lagrange 等 OneBot 实现。
- 接收框架 HTTP 上报的消息事件 (post_type=message)
- 群里: 仅 @机器人 或 消息含触发词 才回; 私聊: 都回
- 调 8081 的 /v1/chat/completions (persona=wife) 拿 文字+语音
- 通过 OneBot HTTP API 发 文字 + 语音 回群

配置区在下方, 改完重启即可。
"""
import json, urllib.request, time, os, threading
import http.server
from http.server import BaseHTTPRequestHandler, HTTPServer

# ═══════════ 配置 ═══════════
LLM_URL    = 'http://127.0.0.1:8081'     # 本地 LLM+语音 代理
ONEBOT_API = 'http://127.0.0.1:3000'     # OneBot 框架 HTTP API 地址 (NapCat/LLOneBot 默认 3000)
BOT_PORT   = 8899                         # 本适配层监听端口 (框架把事件 POST 到这里; 避开 QQ 占用的 8082/8083)
SELF_QQ    = 3900148698                   # 机器人自己的 QQ 号 (填了才能 @机器人 触发 + 防回环)
GROUP_TRIGGERS = ('喵朵', 'miaoduo', '暄依嫣', '依嫣', '青璃')  # 群里不 @ 时, 消息含这些词也回
MAX_TOKENS = 32768                        # 模型思考很啰嗦(~4000-8000 token), 要给足额度否则正文被截断
SEND_TEXT  = True                         # 文字 + 语音一起发（用户要求文字也显示）
SEND_VOICE = True                         # 是否发语音
MAX_HISTORY = 128                         # 持续对话: 保留最近 128 轮(256K上下文,保守一半)
VOICE_DIR  = r'D:\ai\output\audio\voice'
LOG_FILE   = r'D:\ai\llamacpp\qqbot.log'
# ═════════════════════════

def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

def onebot(action, params):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(ONEBOT_API + '/' + action,
                data=json.dumps(params, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json'})
            r = urllib.request.urlopen(req, timeout=30)
            return json.loads(r.read())
        except Exception as e:
            last = e
            time.sleep(2)
    raise last

def text_of(segments):
    """转文字给模型看。注意: @ 段不能喂原始QQ号 — 模型会把号码原样写进回复,
    变成"假@"(纯文本, QQ里不提醒)。用语义标记代替。"""
    out = []
    for s in segments or []:
        t = s.get('type')
        if t == 'text':
            out.append(s['data'].get('text', ''))
        elif t == 'at':
            qq = str(s['data'].get('qq', ''))
            out.append('[@我]' if qq == str(SELF_QQ) else '[@群友]')
    return ''.join(out)

def image_of(segments):
    """提取消息里的图片 file 列表（OneBot image 段的 file 字段）"""
    out = []
    for s in segments or []:
        if s.get('type') == 'image':
            out.append(s['data'].get('file', ''))
    return out

def at_others(segments):
    """提取消息里 @ 的其他 QQ（非机器人自己），用于让机器人回复时 @ 对方"""
    out = []
    for s in segments or []:
        if s.get('type') == 'at':
            qq = str(s['data'].get('qq', ''))
            if qq and qq != str(SELF_QQ):
                out.append(qq)
    return out

# ═══ 看图 (VL-30B 视觉分析) ═══
ENGINE_MGR = 'http://127.0.0.1:8321'  # engine_manager 统一入口

def restore_main_model():
    """VL/生图等任务完成后自动恢复对话主力引擎（35B qwen36）。
    因为 engine_manager 跑任务时互斥停了对话引擎，用完后必须恢复才能继续对话/质检回复"""
    try:
        payload = json.dumps({'name': 'qwen36'}).encode()
        req = urllib.request.Request(ENGINE_MGR + '/api/start', data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
        log('视觉/生图任务完成，自动恢复 35B 对话引擎...')
    except Exception as e:
        log('自动恢复 35B 失败: %s' % str(e)[:120])

def run_vl_task(image_path, question):
    """提交 VL 任务到 engine_manager 并等待结果（engine_manager 自动互斥停其他引擎）"""
    import time
    try:
        payload = json.dumps({'name': 'vl', 'params': {'image': image_path, 'question': question}}).encode()
        req = urllib.request.Request(ENGINE_MGR + '/api/task', data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        return '提交视觉分析失败: %s' % str(e)[:150]
    # 轮询任务结果（VL 加载+分析可能几分钟）
    for _ in range(240):
        time.sleep(5)
        try:
            d = json.loads(urllib.request.urlopen(ENGINE_MGR + '/api/engines', timeout=8).read())
            task = d.get('engines', {}).get('vl', {}).get('task', {})
            st = task.get('status')
            if st == 'done':
                return task.get('result', {}).get('answer', '(无描述)')
            if st == 'error':
                return '视觉分析出错: %s' % str(task.get('error', ''))[:200]
        except Exception:
            pass
    return '视觉分析超时（>20分钟）'

def get_image_local_path(file_ref):
    """通过 OneBot get_image 拿图片本地路径；失败则尝试直接当路径用"""
    try:
        gi = onebot('get_image', {'file': file_ref})
        data = gi.get('data', {}) if isinstance(gi, dict) else {}
        for k in ('path', 'file', 'url'):
            if data.get(k) and os.path.exists(data[k]):
                return data[k]
        # 有的实现返回相对路径
        if data.get('path'):
            return data['path']
    except Exception:
        pass
    # 兜底：file 本身就是本地路径
    if os.path.exists(file_ref):
        return file_ref
    return None

def handle_image(msg_type, target_id, img_file, user_text):
    """后台线程：拿图 → VL 分析 → 回复文字"""
    try:
        path = get_image_local_path(img_file)
        if not path:
            log('看图失败: 无法定位图片 %s' % img_file[:40])
            return
        question = '请用中文详细描述这张图片的内容。'
        if user_text and user_text.strip():
            question += ' 用户的问题或上下文: ' + user_text.strip()[:200]
        log('VL 看图: %s' % path)
        ans = run_vl_task(path, question)
        log('看图结果: %s' % ans[:50])
        # 不自动恢复 35B — 手动切换（引擎面板）
        segs = [{'type': 'text', 'data': {'text': ans[:2000]}}]
        if msg_type == 'group':
            onebot('send_group_msg', {'group_id': target_id, 'message': segs})
        else:
            onebot('send_private_msg', {'user_id': target_id, 'message': segs})
    except Exception as e:
        log('看图失败: %s' % e)


# ═══ 生图 (35B 生成纯英文提示词 → FLUX 生图 → 发图到 QQ) ═══
def handle_image_gen(msg_type, target_id, prompt):
    """后台线程：提交 FLUX 生图任务 → 轮询 → 图片发到 QQ"""
    try:
        import time
        payload = json.dumps({'name': 'flux',
                              'params': {'prompt': prompt, 'steps': 50, 'size': 2560, 'height': 1408}}).encode()
        req = urllib.request.Request(ENGINE_MGR + '/api/task', data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
        log('生图任务已提交: %s' % prompt[:60])
        # 轮询结果（FLUX 2K 约 3-4 分钟）
        for _ in range(360):
            time.sleep(5)
            try:
                d = json.loads(urllib.request.urlopen(ENGINE_MGR + '/api/engines', timeout=8).read())
                task = d.get('engines', {}).get('flux', {}).get('task', {})
                st = task.get('status')
                if st == 'done':
                    img = task.get('result', {}).get('image', '')
                    log('生图完成: %s' % img)
                    # 不自动恢复 35B — 手动切换（引擎面板）
                    if img and os.path.exists(img):
                        segs = [{'type': 'image', 'data': {'file': 'file:///' + img.replace(os.sep, '/')}}]
                    else:
                        segs = [{'type': 'text', 'data': {'text': '🖼️ 图片已生成: %s' % img}}]
                    if msg_type == 'group':
                        onebot('send_group_msg', {'group_id': target_id, 'message': segs})
                    else:
                        onebot('send_private_msg', {'user_id': target_id, 'message': segs})
                    return
                if st == 'error':
                    log('生图失败: %s' % task.get('error', '')[:200])
                    segs = [{'type': 'text', 'data': {'text': '❌ 生图失败: %s' % task.get('error', '')[:150]}}]
                    if msg_type == 'group':
                        onebot('send_group_msg', {'group_id': target_id, 'message': segs})
                    else:
                        onebot('send_private_msg', {'user_id': target_id, 'message': segs})
                    return
            except Exception:
                pass
        log('生图超时')
    except Exception as e:
        log('生图流程失败: %s' % e)

# ═══ 持续对话: 会话上下文缓存 (每群/每用户独立, 保留 MAX_HISTORY 轮, 持久化到文件) ═══
CONV_DIR = r'D:\ai\llamacpp\memory'       # 会话上下文文件目录 (与记忆同目录)
_conversations = {}   # {cid: [messages]}
_conv_lock = threading.Lock()

def _conv_path(cid):
    return os.path.join(CONV_DIR, f'conv_{cid}.json')

def _get_conv(cid):
    """取会话上下文 — 优先内存, 否则从文件加载(重启恢复)"""
    with _conv_lock:
        if cid in _conversations:
            return list(_conversations[cid])
    try:
        with open(_conv_path(cid), encoding='utf-8') as f:
            hist = json.load(f)
        with _conv_lock:
            _conversations[cid] = hist
        return list(hist)
    except Exception:
        return []

def _save_conv(cid, hist):
    """保存会话上下文 — 截断到 MAX_HISTORY 轮, 写内存 + 落盘"""
    hist = hist[-MAX_HISTORY * 2:]  # 保留最近 N 轮(每轮2条)
    with _conv_lock:
        _conversations[cid] = hist
    try:
        os.makedirs(CONV_DIR, exist_ok=True)
        with open(_conv_path(cid), 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False)
    except Exception:
        pass

def pick_persona(text):
    """喊谁谁回: 暄依嫣(deep) / 青璃(qingli) / 默认喵朵(wife)"""
    t = text or ''
    if ('暄依嫣' in t) or ('依嫣' in t) or ('大姐' in t):
        return 'deep'
    if ('青璃' in t) or ('三妹' in t):
        return 'qingli'
    return 'wife'

def llm_chat(text, persona='wife', history=None):
    """调 8081 (按 persona 选人设+音色), 返回 (文字, 语音本地路径或None)。
    persona: wife=喵朵 / deep=暄依嫣 / qingli=青璃
    history: 多轮会话上下文, 无则单轮"""
    messages = []
    if history:
        messages.extend(history)  # 历史对话 (role user/assistant)
    messages.append({'role': 'user', 'content': text})
    payload = {'model': 'qwen', 'persona': persona,
               'messages': messages,
               'max_tokens': MAX_TOKENS}
    # 连接断开/长生成超时重试 (35B 生成慢, 偶发连接被关; 重试 3 次, 间隔递增)
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(LLM_URL + '/v1/chat/completions',
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json'})
            r = urllib.request.urlopen(req)  # 无超时: 模型生成不受限
            d = json.loads(r.read())
            content = (d['choices'][0]['message'].get('content') or '').strip()
            audio_url = d.get('audio')
            audio_path = os.path.join(VOICE_DIR, os.path.basename(audio_url)) if audio_url else None
            return content, audio_path
        except Exception as e:
            last_err = e
            log('LLM 第%d/3 次失败: %s' % (attempt + 1, str(e)[:120]))
            time.sleep(2 * (attempt + 1))  # 2s, 4s 递增
    raise last_err

def _send(msg_type, target_id, segs):
    if msg_type == 'group':
        return onebot('send_group_msg', {'group_id': target_id, 'message': segs})
    return onebot('send_private_msg', {'user_id': target_id, 'message': segs})

def strip_fake_at(text):
    """删除回复里的"假@"(纯文本 @QQ号) — 真@靠 at 段, 文本里的号码只会碍眼/不提醒"""
    import re as _re
    t = _re.sub(r'`?\s*@\d{5,}\s*`?', '', text or '')
    t = _re.sub(r'[ \t]{2,}', ' ', t)
    t = _re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def send_reply(msg_type, target_id, text, audio_path, at_list=None):
    """发送回复: 第1条 = @ + 文字, 第2条 = 语音。
    ⚠️ 别想着合并成一条: 实测 2026-09-12 把 [at, text, record] 放进一条消息,
    NapCat/QQ 只渲染 record, 文字和 @ 全部丢失 — 所以必须分两条发。"""
    text = strip_fake_at(text)
    # 第一条: @指定的人 + 文字（@ 段在开头，确保对方收到 @ 提醒）
    if at_list or (SEND_TEXT and text):
        segs = [{'type': 'at', 'data': {'qq': q}} for q in (at_list or [])]
        if SEND_TEXT and text:
            segs.append({'type': 'text', 'data': {'text': text}})
        if segs:
            _send(msg_type, target_id, segs)
    # 第二条: 语音单独发
    if SEND_VOICE and audio_path and os.path.exists(audio_path):
        _send(msg_type, target_id,
              [{'type': 'record', 'data': {'file': 'file:///' + audio_path.replace(os.sep, '/')}}])
    return True

def handle(event):
    msg_type = event.get('message_type')
    target_id = event.get('group_id') or event.get('user_id')
    sender_id = event.get('user_id')       # 发送者 QQ (群里 = 发这条消息的人/机器人)
    segments = event.get('message')
    text = text_of(segments)
    imgs = image_of(segments)
    log('处理消息 [%s:%s] %s' % (msg_type, target_id, text[:40]))
    # 有图片 → 看图（VL-30B 视觉分析），后台线程不阻塞文字对话
    if imgs:
        log('检测到 %d 张图片，开始看图' % len(imgs))
        threading.Thread(target=handle_image, args=(msg_type, target_id, imgs[0], text), daemon=True).start()
        # 先回复"正在看图"（可选）
        try:
            segs = [{'type': 'text', 'data': {'text': '🔍 正在用视觉模型看图，稍等~'}}]
            if msg_type == 'group':
                onebot('send_group_msg', {'group_id': target_id, 'message': segs})
            else:
                onebot('send_private_msg', {'user_id': target_id, 'message': segs})
        except Exception:
            pass
        return
    # 会话上下文 (伙伴@机制已取消 — 用户要求不再自动@其他人/其他bot)
    _cid = str(target_id)
    hist = _get_conv(_cid)
    persona = pick_persona(text)                   # 喊谁谁回: 喵朵/暄依嫣/青璃
    log('人设=%s' % persona)
    # 生成中占位提示 (35B 生成慢, 让群里知道谁在忙, 避免反复催)
    _who = {'wife': '喵朵', 'deep': '暄依嫣', 'qingli': '青璃'}.get(persona, '喵朵')
    try:
        _send(msg_type, target_id, [{'type': 'text', 'data': {'text': '⏳ %s正在思考，稍等~' % _who}}])
    except Exception:
        pass
    try:
        content, audio_path = llm_chat(text, persona, hist)
        log('LLM 回复 %d 字, 语音=%s' % (len(content), os.path.basename(audio_path) if audio_path else None))
    except Exception as e:
        log('LLM 失败: %s' % e)
        return
    # 更新会话上下文 (持续对话连贯性)
    hist.append({'role': 'user', 'content': text})
    hist.append({'role': 'assistant', 'content': content})
    _save_conv(_cid, hist)
    # 解析生图标记 [IMAGE_PROMPT:纯英文提示词] → 后台 FLUX 生图 + 发图
    import re as _re
    m = _re.search(r'\[IMAGE_PROMPT:(.*?)\]', content, _re.S)
    if m:
        prompt = m.group(1).strip()
        content = content.replace(m.group(0), '').strip() or '🖼️ 正在为你生成图片，请稍等~'
        threading.Thread(target=handle_image_gen, args=(msg_type, target_id, prompt), daemon=True).start()
    try:
        # 群聊: 回复谁就 @ 谁 — 一次只 @ 当前说话的这个, 不记忆不持久 (伙伴机制已废)
        _at = [sender_id] if (msg_type == 'group' and sender_id and str(sender_id) != str(SELF_QQ)) else None
        r = send_reply(msg_type, target_id, content, audio_path, _at)
        log('发送结果: %s (at=%s)' % (r, _at))
    except Exception as e:
        log('发送失败(检查 OneBot 框架是否启动/端口是否对): %s' % e)

BUSY = False

def _run(ev):
    try:
        handle(ev)
    finally:
        global BUSY
        BUSY = False

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _read_body(self):
        # NapCat 用 chunked 传输编码(无 Content-Length), 需手动解码
        if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
            body = b''
            while True:
                line = self.rfile.readline()
                size = int(line.strip().split(b';')[0], 16)
                if size == 0:
                    self.rfile.readline()
                    break
                body += self.rfile.read(size)
                self.rfile.readline()
            return body
        ln = int(self.headers.get('Content-Length', 0) or 0)
        return self.rfile.read(ln)

    def do_GET(self):
        """监督接口: 查看喵朵会话上下文 / 记忆
        GET /api/conv            → 所有会话概览 (cid + 轮数 + 最后一句)
        GET /api/conv?cid=xxx    → 指定会话完整上下文"""
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path == '/api/conv':
            qs = parse_qs(u.query)
            cid = qs.get('cid', [''])[0]
            if cid:
                hist = _get_conv(cid)
                body = json.dumps({'cid': cid, 'rounds': len(hist) // 2, 'messages': hist},
                                  ensure_ascii=False).encode()
            else:
                convs = []
                for c, h in _conversations.items():
                    last = h[-1]['content'][:50] if h else ''
                    convs.append({'cid': c, 'rounds': len(h) // 2, 'last': last})
                body = json.dumps({'conversations': convs}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global BUSY
        body = self._read_body()
        try:
            ev = json.loads(body.decode('utf-8', errors='replace'))
        except Exception:
            ev = {}
        if ev.get('post_type') == 'message':
            msg_type = ev.get('message_type')
            uid = ev.get('user_id')
            if SELF_QQ and uid == SELF_QQ:
                pass  # 忽略自己
            elif msg_type == 'group':
                segs = ev.get('message', [])
                at_me = SELF_QQ and any(s.get('type') == 'at' and str(s['data'].get('qq')) == str(SELF_QQ) for s in segs)
                text = text_of(segs)
                if (at_me or any(k in text for k in GROUP_TRIGGERS)) and not BUSY:
                    BUSY = True
                    threading.Thread(target=_run, args=(ev,), daemon=True).start()
            elif msg_type == 'private' and not BUSY:
                BUSY = True
                threading.Thread(target=_run, args=(ev,), daemon=True).start()
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    log('QQ 机器人适配层启动 http://127.0.0.1:%d → LLM %s, OneBot %s' % (BOT_PORT, LLM_URL, ONEBOT_API))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', BOT_PORT), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
