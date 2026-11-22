# Qwen3-VL-32B 文本编码器 NVFP4 量化（H3-Image / H3-Video 共用）

MiniMax-H3 系列**图像与视频共用一个 Qwen3-VL-32B 文本编码器**，且只用它第 50 层的 hidden。
本仓库给出针对该用法定制的离线 NVFP4(E2M1) 量化器与编码器：**62GB → 约 15–16GB**，RTX 5070 12GB 上 326 token 编码 **23.2s**（BF16 offload 版约 320s）。

> 本仓库只包含**代码与文档，不含权重**。权重请从官方 `MiniMaxAI/MiniMax-H3`（或 ModelScope 镜像）获取后按本 README 转换。
> 仅适用于 **NVIDIA Blackwell（sm120/sm100，RTX 50 系）**。量化格式为 per-16 block FP8 scale + per-tensor global scale 的 NVFP4。

---

## 1. 架构事实（决定了量化策略）

用 safetensors 头统计（总 33.36B 参数 ≈ 62.1GB BF16）：

| 组件 | 参数量 | 占比 | 处理 |
|---|---|---|---|
| LM 0–49 MLP（gate/up/down） | 19.66B | 58.9% | **量化** |
| LM 0–49 attn（q/k/v/o） | 4.72B | 14.1% | **量化** |
| LM 50–63（死层） | 6.83B | 20.5% | **裁掉**（读出点是第 50 层输出，后面永不执行） |
| lm_head | 0.78B | 2.3% | **丢弃**（不生成 token） |
| embed | 0.78B | 2.3% | BF16 保留（敏感） |
| 视觉塔 blocks + merger | 0.59B | 1.7% | BF16（参考图/多参考图入口，另案校准） |

- 注意力是 **GQA：64 个 Q 头 / 8 个 KV 头**（k、v 被 8× 共享放大，误差敏感）
- `q_norm`/`k_norm` 为每头 RMSNorm（128 维）；block 入口/MLP 入口有 RMSNorm；
  **没有归一化的两个接口是 `attn输出→o_proj` 和 `MLP中间→down_proj`**，也是量化误差热点
- **读出索引**：Qwen3-VL 的 `hidden_states` 不含 embedding；64 层模型的 `hidden_states[50]`
  ≡ 50 层模型（裁到 50 层后）的 **`hidden_states[49]`**。错取 `[50]` 会拿到过 final-norm 的末态，对 BF16 误报 97.5%

## 2. 量化策略

- 默认 **全 k1**（q/k/v/o/gate/up/down 全部 k1）
- 可选 `--kv2`：仅 k/v 用 k2 残差（**共用同一个全局 g**，运行时两次 GEMM 求和即可合并成一次），
  实测把注意力模块误差 13.07% → 10.19%（约等于 k/v 留 BF16 的 10.21%，但盘只 +0.13GB）
- 其余（embed、全部 norm、视觉塔、merger）BF16 原样
- 输出：`bf16/` + `fp4/`（packed uint8、scale fp8_e4m3、global g fp32）+ `fp4_manifest.json` + `config.json`（num_hidden_layers=50）

### 实测（RTX 5070 12GB）

| 指标 | BF16 | 本方案 |
|---|---|---|
| 磁盘 | 62.1GB | **约 15–16GB**（k1；含 k2 残差约 16GB） |
| 326 token 编码 | ~320s（offload） | **23.2s**（W4A16：权重 FP4，激活 BF16） |
| k1 权重往返 relL2 | — | 9.51%（E2M1 地板） |
| k2 (k/v) 权重往返 | — | 0.87% |
| 第 50 层 hidden vs BF16（k1+k/v k2） | — | **relL2 7.64%，cosine 0.9791** |
| 同 seed 出图（2048²） L1 / Pearson | — | **26.79 / 0.7953** |
| 对照：同幅随机噪声 cond（relL2 7.64%） | — | 25.61 / 0.8164 |

> 出图漂移与“同等 hidden 误差的随机扰动”在同一水位（L1 差 1.2、Pearson 差 0.02），
> 说明该量化没有引入额外结构性破坏；想更稳只能继续压低 hidden 误差（如 MLP 也上 k2）。

## 3. 用法

```bash
# 1) 转换（默认全 k1；k/v 要 k2 加 --kv2）
python qwen3vl_fp4_convert_v2.py \
  --src <官方 text_encoder 目录> --dst <输出目录> [--kv2]

# 2) 编码（只取第 49 层 hidden；输出格式与 H3 官方 cond 一致）
python h3image_encode_fp4_v2.py \
  --fp4_dir <输出目录> --src <官方 text_encoder 目录> \
  --prompt_file prompt.txt --out cond_fp4.pt
```

依赖：`torch>=2.11 (cu128)`、`transformers>=4.57`、`safetensors`。
**平台必须支持 NVFP4 张量核**（Blackwell）；40 系及更早、AMD 均不可用。

## 4. 已知限制

- 当前编码器是 **W4A16**（权重 FP4、激活 BF16）；W4A4 版本需要把 `FP4Linear` 换成 `_scaled_mm_v2` 内核，精度会再降（激活量化贡献约 9.5%/GEMM，栈内主导）
- 视觉塔未量化（多参考图时它的 token 会走 LM prefill，成本主要在 LM）
- 只验证了 t2i 链路（farm prompt）；i2i/多参考图/视频 cond 需要各自做 presentation 并复测
- 单 prompt 的出图对照不能当最终判决；正式验收需 ≥10 prompt × 2 seed + L1/Pearson

## 5. 许可

- 模型权重：Qwen3-VL 与 MiniMax-H3 的官方许可（本仓库不再分发权重）
- 代码：Apache-2.0
