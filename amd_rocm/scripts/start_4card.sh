#!/bin/bash
# ═══ 4 卡启动脚本 (206GB 显存 → 全量专家常驻显存, 不再需要大 LRU) ═══
# 分卡布局:
#   GPU0: 43层常驻 13GB + embed/lm_head 3.2GB + 热池 2500 个(32GB) ≈ 48GB / 51.5GB
#   GPU1-3: 专家仓库 每卡 3670 个(46.8GB)  → 3×3670 = 11010 ≥ 全量 11008 ✓
#   RAM: LRU 只留 1500 作暂存 (专家都在显存里)
# 计算永远在 GPU0; 仓库卡命中时把一个专家(12.75MB)拉回来 ≈0.65ms (P2P 实测 ~20GB/s)
cd /workspace/engine
export V4F_BASE=/workspace
export V4F_MODEL_DIR=/v4flash_fp4
export V4F_SRC_DIR=/workspace/engine/_v4flash_src
export V4F_TORCH_FP8=1
export V4F_RESIDENT_LAYERS=1          # 43 层常驻 GPU0, 不再每 token 搬 13GB
export V4F_GPU_EXP_MAX=2500           # GPU0 热池 (命中免搬运)
export V4F_EXP_STORE_DEV=1,2,3        # 三张显存仓库卡 (多仓库扩展已内置)
export V4F_EXP_STORE_MAX=3670         # 每卡上限; 3 张正好放下全量 11008 个专家
export V4F_LRU_ITEMS=1500             # 显存已全覆盖, RAM LRU 只作暂存
export V4F_KEEP_GPU_CACHE=1
export V4F_MTP=1                      # 多 token 预测 (≈1.4-2x)
export V4F_EXP_WORKERS=6
export V4F_BUILD_WORKERS=6
export V4F_D2H_WORKERS=4
exec /opt/venv/bin/python v4flash_chunked_server.py
