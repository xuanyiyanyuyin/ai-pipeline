# -*- coding: utf-8 -*-
"""按端口启动 V4-Flash 服务 (4 实例并行用)
   环境: V4F_HTTP_PORT=8123..8126 + CUDA_VISIBLE_DEVICES=i 分卡
"""
import os
import sys

sys.path.insert(0, '/workspace/engine')
os.chdir('/workspace/engine')
import v4flash_chunked_server as srv   # noqa: E402

port = int(os.environ.get('V4F_HTTP_PORT', '8123'))
print('[quad] 启动实例 port=%d dev=%s' % (port, os.environ.get('CUDA_VISIBLE_DEVICES', '?')), flush=True)
srv.serve(port)
