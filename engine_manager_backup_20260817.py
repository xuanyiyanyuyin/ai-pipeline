"""
AI 引擎统一管理器 — 多模型互斥调度
架构: 所有引擎共享同一GPU/RAM, 一次只能运行一个
  - V4-Flash 对话  (v4flash_server.py, 常驻服务, 8123)
  - Qwen3.6 快速聊天 (qwen36_server.py, 常驻服务, 8081)
  - FLUX 生图  (gen_image.py, 任务型)
  - LTX 生视频 (gen_video.py, 任务型)
  - TTS/ASR   (gen_audio.py, 任务型)
  - VL 视觉   (vl_vqa.py, 任务型)
面板: http://127.0.0.1:8321
用法: D:\\ai\\python\\python.exe engine_manager.py
"""
import http.server, socketserver, json, os, sys, subprocess, threading, time, socket, signal
from pathlib import Path

BASE = Path(r'D:\ai')
PORT = 8321
PY = r'D:\ai\python\python.exe'

# ═══ 引擎注册表 ═══
# type: service=常驻HTTP, task=一次性任务
ENGINES = {
    'v4flash': {
        'name': 'V4-Flash 对话 (千文)',
        'type': 'service',
        'script': 'v4flash_server.py',
        'py': r'D:\ai\python_ltx25\Scripts\python.exe',
        'port': 8123,
        'desc': 'DeepSeek-V4-Flash 156GB · FP4 · 7s/tok · CherryStudio接8123',
        'ram_gb': 24,
        'color': '#e040fb',
    },
    'qwen36': {
        'name': 'Qwen3.6-35B 快速聊天',
        'type': 'service',
        'script': 'qwen36_server.py',
        'py': r'D:\ai\python_ltx25\Scripts\python.exe',
        'port': 8081,
        'extra_ports': [8080],
        'desc': '20-25 tok/s · 262K上下文 · 联网搜索 · CherryStudio接8081',
        'ram_gb': 25,
        'color': '#00bcd4',
    },
    'flux': {
        'name': 'FLUX 生图',
        'type': 'task',
        'func': 'gen_image.py:generate_image',
        'desc': '2560×1408 原生2K · 需英文提示词',
        'ram_gb': 12,
        'color': '#ff6b35',
    },
    'ltx': {
        'name': 'LTX-13B 生视频',
        'type': 'task',
        'func': 'gen_video.py:generate_video',
        'desc': '2560×1408 原生2K · 17帧 · 3-4分钟',
        'ram_gb': 26,
        'color': '#ff9d6b',
    },
    'audio': {
        'name': 'Qwen3 TTS/ASR',
        'type': 'task',
        'func': 'gen_audio.py:main',
        'desc': '语音合成 + 识别 · 双模型闭环',
        'ram_gb': 8,
        'color': '#58a6ff',
    },
    'vl': {
        'name': 'Qwen3-VL-30B 视觉',
        'type': 'task',
        'func': 'vl_vqa.py:ask_image',
        'desc': '看图问答 · 官方推理 · 5分钟/次',
        'ram_gb': 25,
        'color': '#3fb950',
    },
}

# ═══ 状态管理 ═══
class EngineState:
    def __init__(self):
        self.services = {}  # name -> Popen
        self.task_lock = threading.Lock()
        self.task_status = {}  # name -> {status, result, error}

state = EngineState()

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except OSError:
            return True

def is_service_running(name):
    eng = ENGINES[name]
    if eng['type'] != 'service':
        return False
    return port_in_use(eng['port']) or name in state.services and state.services[name].poll() is None

def stop_service(name):
    """停止常驻服务"""
    eng = ENGINES[name]
    if name in state.services:
        proc = state.services.pop(name)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    # 端口占用则杀进程 (含 extra_ports, 如 qwen36 的 llama-server 8080)
    import psutil
    ports = [eng['port']] + eng.get('extra_ports', [])
    for p in psutil.process_iter(['pid', 'name']):
        try:
            conns = p.connections()
            if any(c.laddr.port in ports for c in conns):
                p.terminate()
        except Exception:
            pass
    return True

def stop_all_services():
    for name in list(state.services):
        stop_service(name)

def start_service(name):
    """启动常驻服务"""
    eng = ENGINES[name]
    # 互斥: 先停所有其他服务
    stop_all_services()
    py = eng.get('py', PY)
    proc = subprocess.Popen(
        [py, '-u', str(BASE / eng['script'])],
        cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    state.services[name] = proc
    # 等待端口就绪
    for _ in range(60):
        if port_in_use(eng['port']):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return port_in_use(eng['port'])

def run_task(name, params):
    """跑一次性任务 (生图/视频/音频/VL)"""
    eng = ENGINES[name]
    with state.task_lock:
        # 互斥: 停所有常驻服务释放资源
        stop_all_services()
        state.task_status[name] = {'status': 'running', 'params': params}
        try:
            result = _exec_task(name, params)
            state.task_status[name] = {'status': 'done', 'result': result}
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.task_status[name] = {'status': 'error', 'error': str(e)}
            raise

def _exec_task(name, params):
    """执行任务型引擎, 通过 subprocess 调用独立脚本"""
    eng = ENGINES[name]
    if name == 'flux':
        prompt = params.get('prompt', 'a red apple on white table, photorealistic')
        steps = min(int(params.get('steps', 50)), 200)  # 默认50(官方2K推荐上限), 最高200
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            f"import gen_image as gi\n"
            f"fn=gi.generate_image({prompt!r}, steps={steps}, size=2560, height=1408)\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'ltx':
        prompt = params.get('prompt', 'a red apple rotating on white table')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            f"import gen_video as gv\n"
            f"fn,size,n=gv.generate_video({prompt!r}, steps={min(int(params.get('steps',50)),200)}, frames=17, base_w=2560, base_h=1408, do_upscale=False)\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'video': line[7:], 'prompt': prompt}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'audio':
        text = params.get('text', '你好, 我是本地部署的语音助手。')
        voice = params.get('voice', '温柔清晰的年轻女性声音')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import gen_audio\n"
            f"gen_audio.main.__globals__['__name__']='x'\n"
            "import runpy\n"
            f"runpy.run_path(r'D:/ai/gen_audio.py', run_name='__main__')"
        )
        # gen_audio 用 argv, 通过环境传参
        env = dict(os.environ)
        out = subprocess.run([PY, 'gen_audio.py', text, voice], cwd=str(BASE),
                            capture_output=True, text=True, timeout=1200, env=env)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        # 找最新音频
        audio_dir = BASE / 'output' / 'audio'
        files = sorted(audio_dir.glob('*.wav'), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            return {'audio': str(files[0])}
        return {'log': out.stdout[-500:]}
    elif name == 'vl':
        img = params.get('image', '')
        question = params.get('question', '请用中文详细描述这张图片的内容。')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            f"import vl_vqa\n"
            f"ans=vl_vqa.ask_image({img!r}, {question!r})\n"
            "print('RESULT:'+ans)"
        )
        out = subprocess.run([PY, '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1200)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'answer': line[7:], 'image': img}
        raise RuntimeError('no output: '+out.stdout[-500:])
    raise ValueError(f'unknown engine {name}')

# ═══ HTTP 服务 ═══
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def _send(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors(); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/', '/index.html', '/panel'):
            body = (BASE / 'engine_panel.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self._cors(); self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/engines':
            data = {}
            for name, eng in ENGINES.items():
                running = is_service_running(name) if eng['type']=='service' else state.task_status.get(name,{}).get('status')=='running'
                data[name] = {**eng, 'running': running,
                              'task': state.task_status.get(name, {})}
            self._send({'engines': data})
        elif self.path == '/api/status':
            self._send({'services': {n: is_service_running(n) for n in ENGINES if ENGINES[n]['type']=='service'},
                        'tasks': {n: s.get('status') for n,s in state.task_status.items()}})
        else:
            # 静态图片/视频/音频
            if self.path.startswith('/output/'):
                fp = BASE / self.path.lstrip('/').replace('/', os.sep)
                if fp.exists():
                    ext = fp.suffix.lower()
                    ctype = {'png':'image/png','jpg':'image/jpeg','gif':'image/gif','wav':'audio/wav','mp4':'video/mp4'}.get(ext,'application/octet-stream')
                    body = fp.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Content-Length', str(len(body)))
                    self._cors(); self.end_headers()
                    self.wfile.write(body)
                    return
                self._send({'error':'not found'},404)
            else:
                self._send({'error':'unknown'},404)
    def do_POST(self):
        ln = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(ln).decode('utf-8')) if ln else {}
        except:
            req = {}
        if self.path == '/api/start':
            name = req.get('name','')
            if name not in ENGINES or ENGINES[name]['type'] != 'service':
                self._send({'error':'bad engine'},400); return
            # 后台线程启动, 避免阻塞HTTP
            def _start():
                start_service(name)
            threading.Thread(target=_start, daemon=True).start()
            self._send({'ok': True, 'engine': name, 'starting': True})
        elif self.path == '/api/stop':
            name = req.get('name','')
            if name not in ENGINES:
                self._send({'error':'bad engine'},400); return
            stop_service(name)
            self._send({'ok': True, 'engine': name, 'running': False})
        elif self.path == '/api/stop_all':
            stop_all_services()
            self._send({'ok': True})
        elif self.path == '/api/task':
            name = req.get('name','')
            params = req.get('params', {})
            if name not in ENGINES or ENGINES[name]['type'] != 'task':
                self._send({'error':'bad engine'},400); return
            # 后台线程跑, 避免阻塞HTTP
            def _run():
                try:
                    result = run_task(name, params)
                    state.task_status[name] = {'status':'done','result':result}
                except Exception as e:
                    state.task_status[name] = {'status':'error','error':str(e)}
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._send({'ok': True, 'status': 'running', 'engine': name})
        else:
            self._send({'error':'unknown'},404)

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def main():
    print('='*50)
    print('  AI 引擎统一管理器')
    print(f'  面板: http://127.0.0.1:{PORT}')
    print('  引擎互斥: 一次只能运行一个')
    print('='*50)
    # 确保 psutil 可用
    try:
        import psutil
    except ImportError:
        print('[warn] psutil not installed, stop_service may be limited')
    srv = ThreadedServer(('127.0.0.1', PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_all_services()

if __name__ == '__main__':
    main()
