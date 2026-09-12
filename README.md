# AI Pipeline · 自研推理与生成引擎

> 在一张 **12GB** 消费级显卡（RTX 5070）上，跑 **284B MoE 大模型**、**60GB 图像模型**和 **2K 视频生成**。
> 推理与调度代码全部自研 —— 不依赖 vLLM / diffusers / ComfyUI 的高层封装。

---

## 为什么值得看

主流做法遇到"模型比显卡大"时，答案是"换更大的卡"。这个仓库记录的是另一条路：
**把权重按块流式搬运，让只有 12GB 显存的机器跑起本该要 A100 的模型。**

| 引擎 | 模型规模 | 硬件 | 关键数字 |
|---|---|---|---|
| **FLUX.2 三层搬运** | 60GB transformer | RTX 5070 12GB | 2560×1408 出图，前 N 双块常驻 GPU + 其余流式 |
| **LTX-2.5 视频引擎** | 22B / 48 blocks | 同左 | 2K (2560×1408) 视频，显存峰值 8.4GB |
| **DeepSeek-V4-Flash 引擎** | **284B MoE / 149GB 权重** | 同左 | 自研 chunked 调度 + FP8 + CPU 侧 KV cache |
| **Qwen3.6-35B 服务** | 35B | 同左 | 本地 LLM 后端，支撑 QQ 机器人 |

---

## 核心技术

**① 三层搬运（Tiered Weight Streaming）**
单卡装不下整个模型时，把权重分层：前 N 个块常驻 GPU，其余块流式计算
（共享 GPU 槽 → copy 权重 → 前向 → 还原），全程 `no_grad`。
FLUX.2 上用这套把一个 60GB 的 transformer 压进 12GB。

**② 显存峰值的数量级压缩**
`qwen3-storage-rewrite` 重写了 Qwen3 的推理存储逻辑，GPU 按需加载，
显存峰值从 1.1GB 降到 **~25MB**。

**③ 手写分块注意力**
`v4flash_chunked_attention.py` 是纯 PyTorch 的 online chunked eager attention
（分块在线 softmax + 滑动窗口），不依赖 flash-attn / xformers。

**④ dtype 污染定位**
发现并修复了 fp32 的 `scale_shift_table` 与 bf16 激活相加导致整块计算掉进 fp32 的问题 ——
修复后 2K 视频单块前向 **46.5s → 1.1s（42×）**。

---

## 目录导览

```
├── flux*_3tier.py            FLUX / FLUX.2 三层搬运引擎
├── ltx25_*.py                LTX-2.5 自研视频引擎（e2e / VAE / 调度 / 验证）
├── v4flash_chunked_*.py      DeepSeek-V4-Flash 分块推理引擎 + 服务
├── engine_manager.py         引擎统一管理面板（8321）
├── upscale4x.py              4x-UltraSharp 超分（分块 + RGBA）
├── llamacpp/                 本地 LLM 代理 + QQ 机器人适配层
│   ├── websearch_proxy.py    人设路由 / TTS / 工具调用
│   └── qq_bot.py             OneBot v11 适配层
└── *.md                      各引擎的参数总表与调试记录
```

---

## 说明

- **不含模型权重**（数百 GB，请自行从官方仓库获取）
- **密钥已清理**：仓库内所有硬编码 API key 均已替换为占位符
- 部分脚本含本机绝对路径（`D:\ai\...`），迁移时需自行调整

---

*持续更新中 —— 当前方向：把自研引擎移植到 AMD ROCm（gfx1100）平台。*
