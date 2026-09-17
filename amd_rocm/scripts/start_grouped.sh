#!/bin/bash
# ═══ 分组 MoE + 融合 FP4 内核 (2026-09-15 23:0x) ═══
cd /workspace/engine
export V4F_BASE=/workspace
export V4F_MODEL_DIR=/v4flash_fp4
export V4F_SRC_DIR=/workspace/engine/_v4flash_src
export V4F_TORCH_FP8=1
export V4F_TRITON_FP4=1               # ★ 融合 MXFP4 内核 (启动后台预热)
export V4F_FUSE_EPI=0              # 融合epilogue实测更慢(75.5s vs 38.6s moe), 关闭
export V4F_GROUPED_MOE=1              # ★ 分组 MoE 前向
export V4F_RESIDENT_LAYERS=1
export V4F_GPU_EXP_MAX=2500
export V4F_EXP_STORE_DEV=
export V4F_EXP_STORE_MAX=0
export V4F_LRU_ITEMS=13000
export V4F_KEEP_GPU_CACHE=1
export V4F_MTP=1
export V4F_BUILD_WORKERS=6
export V4F_D2H_WORKERS=4
export V4F_EXPACK=/v4flash_expack/experts.bin
export V4F_PIN_LRU=0
export V4F_EXP_WORKERS=3
exec /opt/venv/bin/python -u v4flash_chunked_server.py >> /workspace/v4f_server_g.log 2>&1
