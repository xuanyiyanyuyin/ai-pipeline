# AMD ROCm 移植 · 4×W7900D 上的 284B MoE 引擎

> 把本仓库的 DeepSeek-V4-Flash 自研引擎（三层缓存 + 43 层窗口流水）
> 从本机 **RTX 5070 12GB / CUDA** 移植到 **AMD Radeon Cloud 4 × W7900D（gfx1100 / RDNA3 / ROCm）**，
> 并把 TPOT 从 **1.45 s 压到 0.454 s/token（2.20 tok/s）**。

## 结果一览

| 指标 | 数值 |
|---|---|
| 最终解码速度（4 卡，可控基准） | **0.454 s/token = 2.20 tok/s** |
| 优化前 → 优化后 | 0.991 → 0.454 s/token（**2.2×**） |
| 首版移植 → 最终 | 1.45 → 0.454 s/token（**3.2×**） |
| 4 路并发（一卡一路，4 实例） | 单路 1.80 → 聚合 **3.83 tok/s**（2.0~2.1×） |
| 实测天花板 | **≈8 tok/s**（每 token 必搬 3.3GB ÷ RAM→GPU 26.3GB/s = 126ms） |

## 两个根因（通用性最强，值得单独看）

1. **空输出 / NaN（"同一份权重时好时坏"）** —— `to_empty()` + `load_state_dict()` 漏掉 **`persistent=False` 的非持久 buffer**（RoPE 频率），模型带着"上一手内存"上线；本机同 bug 但只表现为静默失效（`inv_freq ≈ 0`），云端那块内存含 NaN 位模式 → 约 1/2 概率空输出。
2. **专家缓存命中率恒为 0** —— 单请求工作集 ~7000 个专家 > 缓存 4000 槽 → LRU 抖动 → **每请求白读 75GB 磁盘**；把三级容量做到 ≥ 工作集后，读盘降到 2.5GB/请求。

完整证据链、性能工程、带宽/并发/显存实测、天花板核算与复现步骤见
**[AMD_ROCm移植技术详解_20260917.md](AMD_ROCm移植技术详解_20260917.md)**。

## 目录

```
amd_rocm/
├── AMD_ROCm移植技术详解_20260917.md   ★ 完整技术报告（13 章）
├── engine/                            AMD 适配版引擎（含 RoPE 修复 / 专家单文件仓库 / 分组 MoE / Triton FP4）
├── scripts/                           移植与调优脚本（补丁、专家重排、多实例服务、测速与探针）
└── results/                           实测数据（带宽、命中率、并发、吞吐）
```

## 关键改动（全部 env 可回退）

| 改动 | 机制 | 效果 |
|---|---|---|
| Triton 融合 MXFP4 内核 | 打包 int8 → 寄存器反量化 → fp32 累加（数值逐位一致） | 专家 GEMM **0.42~0.93ms → 0.063ms** |
| 分组 MoE 前向 | 一次路由摊开 → 按专家排序 gather → 逐组 GEMM → 散射 | 索引记账 **70.3 → 2.1 ms/token** |
| 专家重排（6 散块 → 1 连续记录） | 每条 13,369,344 B，`1 次 .to()` 拷整块 | 拷贝调用 **1260 → 210 次/token** |
| 拷贝线程 6 → 3 | 线程越多越慢（GIL 竞争） | 与重排合计 **0.991 → 0.454 s/token** |
| 4 进程 = 4 卡 | 绕开 GIL 与进程级驱动锁 | 每卡 0.194 ms/专家，**4.01× 线性**（对照 4 线程仅 1.1×） |
| 43 层权重常驻显存 | 免掉每 token 13GB 的层权重搬运 | 单请求 **113s → 36s** |

## 复现（新实例约 12 分钟）

```bash
# 依赖（--no-deps 保护容器自带 ROCm torch）
/opt/venv/bin/pip install -q --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==5.16.1 huggingface_hub==1.29.0 tokenizers==0.23.1 safetensors==0.8.0
# 模型（HF_HUB_DISABLE_XET=1，hf-mirror 多线程）
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
/opt/venv/bin/hf download deepseek-ai/DeepSeek-V4-Flash --local-dir /v4flash_fp4 --max-workers 8
# 专家单文件仓库（137.1GB）
cd /workspace && /opt/venv/bin/python repack_experts.py
# 4 实例，一卡一路
cd /workspace && bash start_quad.sh
```

## 注意

- **不含模型权重**；路径（`/workspace`、`/v4flash_fp4`、`/v4flash_expack`）按需调整。
- 若平台无直连外网，pip / HF 下载需走平台代理。
- 所有数字均为实例内实测（固定 prompt + 贪心 + 从日志数真实 token），脚本见 `scripts/`。
