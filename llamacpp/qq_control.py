# -*- coding: utf-8 -*-
"""QQ 机器人一键启动/关闭
用法: python qq_control.py start | stop
"""
import subprocess, sys, socket, time, os

PY = r'D:\ai\python_ltx25\Scripts\python.exe'
LLAMACPP = r'D:\ai\llamacpp'
MODEL = r'D:/ai/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
BOT = r'D:\ai\llamacpp\qq_bot.py'
PROXY = r'D:\ai\llamacpp\websearch_proxy.py'
NAPCAT = r'D:\ai\napcat\napcat\launcher-user.bat'
NAPCAT_CWD = r'D:\ai\napcat\napcat'
QQ_NUM = '3900148698'
CREATE_NEW_PROCESS_GROUP = 0x00000200

def port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(('127.0.0.1', port))
        return True
    except Exception:
        return False
    finally:
        s.close()

def proc_exists(name_substr):
    import psutil
    for p in psutil.process_iter(['name']):
        try:
            if p.info['name'] and name_substr.lower() in p.info['name'].lower():
                return True
        except Exception:
            pass
    return False

def kill_port(port):
    try:
        out = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return
    pids = []
    for line in out.splitlines():
        if (':' + str(port) + ' ') in line and 'LISTENING' in line:
            parts = line.split()
            if parts and parts[-1] not in pids:
                pids.append(parts[-1])
    for pid in pids:
        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
        print('  killed port %s (pid %s)' % (port, pid))

def kill_napcat():
    import psutil
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            name = (p.info['name'] or '').lower()
            cmd = ' '.join(p.info['cmdline'] or []).lower()
            if name == 'napcatwinbootmain.exe' or (name == 'qq.exe' and 'enable-logging' in cmd):
                p.terminate()
                print('  killed %s (pid %s)' % (p.info['name'], p.pid))
        except Exception:
            pass

def start():
    # 0. engine_manager (8321) — 引擎统一管理中枢（看图/生图/互斥都走它）
    if not port_open(8321):
        print('[0/4] starting engine_manager (8321)...')
        subprocess.Popen([PY, '-u', r'D:\ai\engine_manager.py'], cwd=r'D:\ai',
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=CREATE_NEW_PROCESS_GROUP)
        for _ in range(20):
            if port_open(8321):
                print('    engine_manager ready')
                break
            time.sleep(1)
    else:
        print('[0/4] engine_manager (8321) already running')

    # 1. llama-server (8080) + websearch_proxy (8081)
    if not port_open(8080):
        print('[1/4] starting llama-server (8080)...')
        subprocess.Popen(
            [os.path.join(LLAMACPP, 'llama-server.exe'), '-m', MODEL,
             '-ngl', '99', '-cmoe', '-c', '262144', '-ctk', 'q8_0', '-ctv', 'q8_0',
             '-fa', 'on', '-t', '14', '--host', '127.0.0.1', '--port', '8080',
             '--temp', '1.0', '--repeat-penalty', '1.1', '--presence-penalty', '0.2', '--frequency-penalty', '0.2',
             '--dynatemp-range', '0.8', '--dynatemp-exp', '1.0'],
            cwd=LLAMACPP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NEW_PROCESS_GROUP)
        for _ in range(120):
            if port_open(8080):
                print('    llama-server ready')
                break
            time.sleep(1)
    else:
        print('[1/3] llama-server (8080) already running')

    if not port_open(8081):
        print('    starting websearch_proxy (8081)...')
        subprocess.Popen([PY, PROXY], cwd=LLAMACPP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=CREATE_NEW_PROCESS_GROUP)
        for _ in range(30):
            if port_open(8081):
                print('    websearch_proxy ready')
                break
            time.sleep(1)
    else:
        print('    websearch_proxy (8081) already running')

    # 2. qq_bot adapter (8899)
    if not port_open(8899):
        print('[2/4] starting qq_bot adapter (8899)...')
        subprocess.Popen([PY, BOT], cwd=LLAMACPP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=CREATE_NEW_PROCESS_GROUP)
        for _ in range(20):
            if port_open(8899):
                print('    qq_bot ready')
                break
            time.sleep(1)
    else:
        print('[2/4] qq_bot adapter (8899) already running')

    # 3. NapCat (QQ framework)
    if not proc_exists('napcatwinbootmain'):
        print('[3/4] starting NapCat (QQ %s)...' % QQ_NUM)
        subprocess.Popen([NAPCAT, QQ_NUM], cwd=NAPCAT_CWD, creationflags=CREATE_NEW_PROCESS_GROUP)
        print('    NapCat launched')
    else:
        print('[3/4] NapCat already running')

    print('Done.')

def stop():
    print('Stopping NapCat + bot QQ...')
    kill_napcat()
    print('Stopping qq_bot adapter (8899)...')
    kill_port(8899)
    print('Stopping websearch_proxy (8081)...')
    kill_port(8081)
    print('Stopping llama-server (8080)...')
    kill_port(8080)
    print('Stopping engine_manager (8321)...')
    kill_port(8321)
    print('Done.')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'start':
        start()
    elif cmd == 'stop':
        stop()
    else:
        print('Usage: python qq_control.py start|stop')
        sys.exit(1)
