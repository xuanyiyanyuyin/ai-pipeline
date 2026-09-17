#!/bin/bash
# 4 实例并行: 每卡一个引擎进程, 端口 8123-8126
# 共享: /v4flash_expack/experts.bin + /v4flash_fp4 (系统页缓存一份)
cd /workspace
pkill -f '[v]4flash_chunked_server.py'; pkill -f '[s]erve_port'; sleep 8

for i in 0 1 2 3; do
  PORT=$((8123 + i))
  LOG=/workspace/inst_$i.log
  rm -f $LOG
  (
    export V4F_BASE=/workspace
    export V4F_MODEL_DIR=/v4flash_fp4
    export V4F_SRC_DIR=/workspace/engine/_v4flash_src
    export V4F_TORCH_FP8=1
    export V4F_TRITON_FP4=1
    export V4F_GROUPED_MOE=1
    export V4F_FUSE_EPI=0
    export V4F_RESIDENT_LAYERS=1
    export V4F_GPU_EXP_MAX=2500
    export V4F_EXP_STORE_DEV=
    export V4F_EXPACK=/v4flash_expack/experts.bin
    export V4F_PIN_LRU=0
    export V4F_LRU_ITEMS=2500
    export V4F_KEEP_GPU_CACHE=1
    export V4F_MTP=1
    export V4F_EXP_WORKERS=3
    export V4F_BUILD_WORKERS=6
    export V4F_D2H_WORKERS=4
    export V4F_HTTP_PORT=$PORT
    export CUDA_VISIBLE_DEVICES=$i
    export HIP_VISIBLE_DEVICES=$i
    setsid nohup /opt/venv/bin/python -u /workspace/serve_port.py >> $LOG 2>&1 < /dev/null &
  )
  echo "实例$i → 端口 $PORT (cuda:$i) 已提交"
  sleep 2
done

echo "--- 等 4 个实例就绪 (最多 10 分钟) ---"
for t in $(seq 1 60); do
  N=$(grep -l "HTTP 服务" /workspace/inst_*.log 2>/dev/null | wc -l)
  [ "$N" -ge 4 ] && { echo "4 个实例全部就绪 ($((t*10))s)"; break; }
  sleep 10
done
grep -h "HTTP 服务" /workspace/inst_*.log
echo "--- 显存 ---"
rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[[0-9]\]" | head -8
echo "--- 内存 ---"
free -g | head -2
