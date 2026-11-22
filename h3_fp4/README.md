# H3 / H3-Image NVFP4 (W4A4) 本地流式推理 —— RTX 5070 12GB

自量化 NVFP4 k1 + 真 W4A4 FP4 张量核 + 逐层流式 + AdaLN 预计算 + 自动驻留。
权重**事先离线全量化**（运行时不量化/不反量化权重）；激活量化为 W4A4 固有在线成本。
BF16 原件保留不删。2026-11-20 实测。

## 文件（自包含，sys.path 已含本目录）
- **基础BF16引擎（依赖，已随附）**：`h3_engine.py`（视频：rms/rope/SDPA/SwiGLU/_qkv_fix/位置网格/Reader/auto_slice/VAE编解码/run_ref2va）、`h3image_engine.py`（图像：画布解析/cond加载/build_layout/VAE解码/to_png/PDD路径）。
- `h3image_fp4_convert.py` / `h3video_fp4_convert.py`：BF16→NVFP4 k1 转换器（FFN fc1/fc2 + attn qkv/out 量化；AdaLN/norm/embed/输出头/patchify 保 BF16/FP32）。
- `h3image_adaln_precompute.py` / `h3video_adaln_precompute.py`：AdaLN t 键预计算（图像/视频；视频覆盖 video shift12 / audio shift3 / 0.999）。
- `h3image_engine_fp4.py`：H3-Image FP4 引擎（NVFP4 算子 + 流式 + 自动驻留）。
- `h3_engine_fp4.py`：H3 视频 FP4 引擎（monkey-patch BF16 视频引擎；**启动按实际 steps 自动建/复用 AdaLN 缓存**；--tile/--ov 空间分块；--nodec 只测DiT）。
- `h3image_encode.py`：**纯文生图条件编码**（Qwen3-VL 取第 50 层 hidden → `cond_t2i.pt`；t2i 必须先跑它）。之前漏随附，已补。
- `h3_encode.py`：参考图/多参考图条件编码（i2i / edit / 视频 ref2va 用；`--ref` 可多张，顺序=官方 reference 顺序）。
- `h3_video_job.py`：视频任务驱动（--steps/--turbo/--frames/--size/--cond）。
- `h3_engine_server.py`：常驻 HTTP 服务（/v1/images/generations|edits、/v1/video/generations，异步 job）。
- `fused_quant.cu` + NVRTC 运行时编译：融合 NVFP4 量化 kernel（免 MSVC/nvcc）。
- `check_env.py`：**环境自检**（torch/CUDA/架构/_scaled_mm_v2/cuda-python/NVRTC/磁盘余量）。

## 环境自检（先跑这个）

```bash
python check_env.py
```

- 需要 **torch>=2.11 + CUDA 12.8 (cu128)**；CPU 版 torch 直接判 FAIL（本引擎依赖 `_scaled_mm_v2`，CPU 上没有）
- 需要 **Blackwell (sm100/sm120)**；RTX 40 系及更早、AMD 卡都会被判不可用（无 FP4 张量核）
- 磁盘：BF16 原件 66GB + FP4 37.5GB ≈ **单模态至少 110GiB 空余**；图像+视频都转则 ≈208GiB
- `cuda-python` / `nvidia-cuda-nvrtc-cu12` 缺失只 WARN（会回退慢速 Python 量化）

## 模型权重（不随仓库，需自备）
- H3-Image：`MiniMaxAI/MiniMax-H3-Image`（HF）或 ModelScope 镜像；放 `D:\ai\models\MiniMax-H3-Image`（transformer/vae/pdd_fused_heads/pdd_config/tokenizer/processor/scheduler；text_encoder 可链接 H3 的）。
- H3 视频：`MiniMax-H3/Ref2VA`（transformer/video_vae/audio_vae/text_encoder/scheduler/audio_scheduler）。
- 转换：先跑 `h3*_fp4_convert.py` 生成 FP4 分片到 `*-FP4/transformer`，再跑 `h3*_adaln_precompute.py [steps]` 建缓存。
- 路径默认 `D:\ai\...`，他机请改各文件顶部 ROOT/FP4_DIR/COND 常量。

## NVFP4 算子（已验证逐位等价）
`torch._scaled_mm_v2(a4, b4, scale_a=[swizzled_block_scale, 1/global], recipe=[BlockWise1x16, TensorWise], swizzle=[SWIZZLE_32_4_4, NO_SWIZZLE], ...)`
- block scale 经 `to_blocked()`（cuBLAS 128x4 / 32x4x4 swizzle）。
- global scale 传 **1/g**（g=(448*6)/amax 为编码侧）。
- GEMM 输出 **BF16**（不要 FP32，否则 4K 激活翻倍）。
- 激活量化分块 CH=4096 限制 int64 bucketize / fp32 临时张量。
- 残差"套娃"k 层可沿 N 合并成一次 GEMM（FP32 逐位等价、激活只量化一次）；但 H3 大矩阵已饱和，合并增益小。

## 实测（RTX 5070 12GB, Windows, torch 2.11 cu128, cuDNN 9.19）
### 量化损失（我们的方法，往返 relL2）
| | k1 | k2 套娃 | k3 套娃 |
|---|---|---|---|
| fc1/fc2/qkv/out/adaln（图像+视频同族） | 9.4-9.6% | 0.86% | 0.078% |
k1 的 9.4% 是 E2M1（1 位尾数）固有地板，全组件一致；选块救不了。

### 转换
- 图像/视频 DiT 各 66.28GB BF16 → **37.48GB**（208 FP4 + 327 保留），~100-105s。
- 37.48GB 里 26GB 是保留的 AdaLN BF16；FP4 核心仅 11.3GB。AdaLN 预计算后每图实际流 ~11.5GB。

### H3-Image（图像）
- AdaLN 预计算 58MB，逐位等价（maxdiff 0）。
- 512²：**15s**（BF16 648s ⇒ **43×**）；2048² 单步 17.1s；4096² 单步 30.6s / 整图 **178s**。
- 自动驻留：512/2048=32、4K=12；峰值 ≤9.5GB，**不溢共享**。
- 画质 k1：同场景照片级、无网格/塌灰；vs BF16 L1 9.7% / Pearson 0.84（细节漂移，结构对）。主人目视认可。

### H3 视频（Ref2VA, joint video+audio）
- 20 步 124 帧 512²：**284s**（BF16 同配置 ~30min ⇒ **~6×**）；帧清晰连贯、有运动（frame0→60 L1 0.184）。
- 速度场与 BF16 逐位级接近（|v|std 1.30 / 相邻格相关 0.40）。
- **2K/4K 单步：>240s（有界实测未完成）⇒ 12GB 卡不实用**。原因：注意力 O(L²)（2K L≈33k / 4K L≈49k）+ 激活量化 bucketize(int64) 每步 ~1800 次 + 大形状首调用 autotune。4K 满帧 L≈114k 激活 >12GB 必溢。
  优化路径：E2M1 编码改算术/融合 kernel（去 bucketize）、FP4 注意力（cuDNN sm120 现成 mxfp8/fp8，**无 fp4-input**；SageAttention3/手写 cute-DSL 为重工程）、或减帧/分块。

## 坑（实测踩过）
1. **AdaLN 缓存 steps 错配 = confetti 垃圾**（steps4 缓存跑 20 步 ⇒ searchsorted 取错 t ⇒ 调制错 ⇒ 发散）。非量化缺陷。⇒ 引擎按实际 steps 自动建缓存。
2. **pin_memory Windows 额度 ~7.5-8GB**：超了 pin 分配 OOM（非驱动坏）；32GB RAM 放 11.5GB FP4 足够，**不需要 pin**。
3. **4K 激活估算错**：fp32 GEMM 输出 + int64 bucketize 使实际 ~6GB（我估 2.1-2.5GB 错）⇒ 改 BF16 输出 + 分块量化 + 自动降驻留。
4. **驻留 sweet spot 32-36**；40+ 显存近满分配器 thrash 反慢。
5. **CPU 算不了 FP4**（NotImplementedError；FP32 也慢 93-566×）；CPU 多核预取实测 0 收益（4K 算力主导 + pageable 不真异步）。CPU 价值仅并行搬运（盘读 0.9→5.88GB/s）。
6. `run_ref2va` 未设 seed ⇒ 对照须 `torch.manual_seed`；旧 `h3_cond.pt` 缺 token_tags ⇒ 用带 tags 的 cond（如 h3image_cond_t2i.pt）。
7. 参考图按自身分辨率编码（历史苹果 512 / 冒烟 farm 512）；ref 糊 ⇒ 视频糊。

## 视频 MultiDiffusion 空间分块（已实现, h3_engine_fp4.py --tile/--ov）
- 全局 RoPE 子网格（tile 行取全局 pos 坐标）+ overlap 线性融合；ref/text/audio 全 tile 共享。
- 打包索引用**局部 arange**（全局 vid 会 index_copy 越界）；blend 权重/索引须同设备。
- 实测 2K(2048²,22帧) tile=1024：**ov0=4tiles 单步51s；ov256=9tiles 70s**（不分块>240s）；3步247s 出片连贯无接缝（糊=欠收敛非分块）。
- 4K tile1024：**ov0=16tiles 单步173s；ov256=36tiles 294s**；完整1步+解码=1239s（**4K VAE解码~945s 是大头**）。4K 20步≈58-114min⇒重；**2K 是这 box 实用上限**。
- **overlap**：ov>0 重叠区线性ramp融合⇒无缝但tiles多/慢；ov0 快但有接缝风险。
- **时序**：每tile跑全部帧（时序不分块）⇒tile内运动连贯；ref/audio全tile共享。时序分块(切帧+时序blend)未做=下一层工程。
- 每 tile 固定开销(50层 kernel+量化~3-4s)是小 L 主导；分块省注意力 O(L²) 但不省固定开销。

## 自研融合NVFP4量化算子（NVRTC官方运行时编译，免MSVC/UAC/4GB toolkit）
- `fused_quant.cu`：单kernel完成 absmax/16→FP8 scale→E2M1舍入→pack（一线程一16块）。
- NVRTC(`nvidia-cuda-nvrtc-cu12`官方pip)编sm_120 cubin + cuda.bindings driver加载，**无需nvcc/MSVC**。
- 实测 **0.091ms vs Python quant_act 14.47ms = 160×**，scale逐位一致（packed仅舍入tie差）。
- 已接入 `fp4_lin`(_quant)：砍掉4K那24%量化开销；nvrtc失败自动回退quant_act。
- 装编译器正路=CUDA toolkit+MSVC(几GB+UAC)；NVRTC是轻量官方替代（~50MB pip）。

## SM120 FP4 峰值（空闲实测，不拉频/功率/任何硬件）
- **峰值 415 TFLOPS @ 8192³**（对齐128方阵）≈ **6.7× BF16(62)**；H3 FFN形状(4096×28672×5376)=400 TFLOPS。
- 16384³ 掉到224（超显存/occupancy甜点）；长条(28672×5376)400；**8192³方阵最优**。
- **游戏负载会把峰值压到~179**（争抢SM/带宽）⇒测峰值须GPU空闲。
- `use_fast_accum=True` 对FP4不支持（torch报错）；trtllm-gen/DeepGEMM SM120不可用⇒只能CUTLASS `_scaled_mm_v2`+形状优化。
- 拉功率参考：5070默认250W、厂商标定175-280W；250→280(+12%)仅+5-8%（频率天花板边际递减）；>280W=烧风险别碰；降压(undervolt)更划算。

## k 各级别：精度 / 体积 / 速度预估（残差套娃，等价拆分不旋转）
| 级别 | bit/权重 | 往返relL2 | DiT核心体积 | 画质 | 适用 |
|---|---|---|---|---|---|
| BF16 | 16 | 0(参考) | 66.28GB | 参考 | 大显存/终稿 |
| **k1** | 4.5 | **9.4-9.6%** | **11.3GB** | 同场景照片级,细节漂移~10%(L1 9.7/Pearson0.84) | 12G/速度优先 |
| **k2** | 9 | **0.86%** | **22.5GB** | 近无损 | 24G/质量+速度平衡 |
| **k3** | 13.5 | **0.078%** | **33.75GB** | ≈BF16 | 24G+offload/质量优先 |
- k1 的 9.4% 是 E2M1(1位尾数)固有地板,全组件一致,选块救不了;k2/k3 靠残差层补尾数。
- 精度↔体积对立:k 每+1 层 +4.5bit/权重;合并 k 层沿 N 成一次 GEMM(FP32 逐位等价、激活只量化一次),不额外增体积。
- AdaLN 另计:保 BF16=26GB,或预计算 t 键缓存 58MB(图像)/~1GB(视频50步)逐位等价,推荐预计算。

## 5090 笔记本 24G 适配（群友配置）
- 24G 可**全驻留 k1(11.3GB)+4K激活(~6GB)**⇒4K 图像不用分块、不用流式,直接整图;5090 FP4 吞吐约 2×5070⇒4K 整图预计 ~60-90s。
- **k2(22.5GB)** 在 24G 全驻留(512/2K 激活小)⇒近无损+FP4 速度,是 24G 的甜点位。
- **k3(33.75GB)** >24G⇒需流式/offload,质量≈BF16 但速度降;质量优先才选。
- 4K 视频:24G  headroom 更大,1024-tile 分块(16 tiles ov0≈128s/步@5070,5090 更快)⇒4K 视频可行;VAE 解码仍是大头,建议 tiled VAE+半精度。
- 运行:同 h3_engine_fp4.py / h3image_engine_fp4.py,RESIDENT 调高(24G 可 40-50),--tile 按需。

## 结论
12GB Windows 5070：图像 FP4 512=15s/4K=178s（43×/显著），视频 512 20步=284s（6×）；视频 2K/4K 单步不实用（注意力 O(L²)+量化开销），需更大显存或 FP4 注意力/减帧。