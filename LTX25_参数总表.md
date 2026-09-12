# LTX-2.5 完整参数总表（自研引擎）

> 生成: 2026-09-11 · 来源: `D:\ai\ltx25_*.py` 实际代码 + `engine_manager.py` 卡片

## 一、模型文件（`D:\ai\_ltx25_src\`，共 65 GB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `ltx-2.5-22b-dev-transformer-bf16_v2.safetensors` | **39.1 GB** | DiT 主干。**V2 干净**；V1 的 block6/39 ff 含 NaN |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | **24.5 GB** | Gemma4 文本编码器 + 投影层 |
| `ltx-2.5-video-vae-bf16.safetensors` | **1.4 GB** | 视频 VAE（3D，时间 8× / 空间 32× 压缩）|

## 二、DiT 结构（`ltx25_sched.py`）

| 项 | 值 |
|---|---|
| 层数 | **48 blocks** |
| 隐藏维 / 头数 / 头维 | 4096 / 32 / 128 |
| Patchify | in=**128**（latent 通道）→ out=**4096** |
| 文本条件 | cross-attention，文本维 **4096**（Gemma4 输出） |
| RoPE | positions = latent 网格 × **scale(8, 32, 32)**；temporal ÷ fps；rope_dim=64；theta=10000 |
| timestep | TIMESTEP_SCALE_MULTIPLIER = **1000** |
| 调制 | shift/scale（含 pss 修正）→ norm_out |
| 注意力后端 | **必须固定 SDPA 后端**（bf16 下不同后端差 ~4e-3，48 层串联放大） |

## 三、IO / 三层搬运（继承 0.9.8 基建）

- **权重读取全用 `pread`**（struct header + seek data_offsets）——mmap 会段错误（OS 1455 页面文件太小）
- **块权重预取**：`ThreadPoolExecutor(max_workers=2)` 后台 pread + H2D，与 block 计算重叠（隐藏磁盘 I/O）
- 三层调度 / offload / 单文件 safetensors 按块读取

## 四、采样（Euler + CFG）

| 项 | 值 |
|---|---|
| CFG | **3.0**（负提示词 = 零向量） |
| sigma 调度 | `LTX2Scheduler.execute(STEPS, lat, stretch=True, terminal=0.1)` |
| 更新式 | `v = v_pos + CFG*(v_pos - v_neg)` → Euler |
| VAE 归一化域 | 协议对齐（sigma x1=1024, x2=4096） |

## 五、两个入口

### ① 文生视频 `ltx25_e2e_1024.py`
| 项 | 值 |
|---|---|
| latent 网格 | F,H,W = **9, 16, 16** → **512×512** |
| 步数 / CFG | **30 / 3.0** |
| 文本 emb | (1, **1024**, 4096) |

### ② 图生视频 `ltx25_e2e_i2v.py`（engine_manager 卡片 `i2v`）
| 项 | 值 |
|---|---|
| 分辨率 | **2560×1408**（latent 80×44，2K 原生不超分）|
| latent 帧数 | **F = 9** |
| 步数 / CFG | **50 / 3.0** |
| 首帧机制 | VAE encoder 编码图片 → image_latent 替换第 1 帧 → **每步 Euler 后重置首帧**（denoise_mask=0）|
| CLI | `python ltx25_e2e_i2v.py [image] [prompt] [steps] [frames] [width] [height]` |

## 六、输出

- VAE decode：latent(1,128,F,H,W) → frames[f,h,w,c] → **PNG 帧 + mp4（fps=24）**
- 输出目录：`D:\ai\output\ab_test\`
- 9 latent 帧 × 时间压缩 8× → 约 **65 帧 / 2.7 秒 @24fps**（以 decode 打印为准）

## 七、文本编码 `ltx25_gemma_encode.py`

| 项 | 值 |
|---|---|
| 编码器 | **Gemma4 12B + proj**（不是 T5！）|
| 输出 | text_emb (1, S, **4096**) bf16 |
| MAXLEN | **1024**（官方 LTX2Pipeline 默认 max_sequence_length）|
| 层数 | 49 hidden states（含 embedding），官方 FeatureExtractorV2 取特征 |
| 位置 | 编码在 **CPU** 跑 |
| tokenizer | 也用 pread（mmap 会段错误）|

## 八、历史坑（已修）

| 坑 | 现象 | 修法 |
|---|---|---|
| VAE 配对错误 | 512 vs 1024 通道混用 | 对齐官方 VAE 配置 |
| RoPE 空间频率差 32× | 画面出"方块" | positions scale 改 (8,32,32) |
| Euler 步进符号反 | std 爆炸（8.04 → 1.28）| 修正步进符号 |
| V1 权重 NaN | block6/39 前馈污染 | 换 V2 权重 |
| mmap 段错误 | OS error 1455 | 全部改 pread |
| 引擎 vs 官方 | —— | 单步 forward **CORR = 0.9999** ✓ |

---
> 环境：跑 LTX-2.5 用 `D:\ai\python_ltx25\Scripts\python.exe`
