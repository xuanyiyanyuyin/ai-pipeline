# V22.1 — 白盒多模态引擎 + 图生图生成端

V22.1 是一个从底层信号（字符 / 原始像素 / FFT）出发的白盒多模态认知引擎，以及它的图像生成端（M1 16×16 / M2 64×64）。

不使用预训练 CNN / ViT / Whisper；统一 `256D 球面 embedding + 384D 属性 = 640D` token；生成端使用冻结视觉词典 + 逐位 16 阶分类。

## 当前状态（2026-10-26）

| 模块 | 状态 |
|---|---|
| 引擎（字符/视觉/声学 → P7 → ABC → P6） | 三模态接线；需要自备文本数据；视觉/声学为实验支路 |
| M1（16×16 图生图） | 最佳考试 60.17% |
| M2（64×64 patch 化 + P7 跨 patch） | 最佳考试 65.23%（`--spatial_upgrade 1`，600 轮 b16） |
| M2 `transform_v2` | 阻塞：缺 (0,0) 自项，loss 平 ln(8) |
| M3（文生图） | 未开始 |

## 代码入口

- `train_v22_m2.py` — M2 主训练入口（当前主线）
- `train_v22_img2img.py` — M1
- `train_v22_stage1.py` — V22.1 引擎训练入口（文件头写 V19，命名遗留）

## 详细代码说明

完整的手册（目录地图、数据流、张量形状、训练/评测协议、已知缺陷、复现命令、文件指纹）：

**[V22.1_代码详细说明.md](V22.1_代码详细说明.md)**

## 快速开始

环境：Python 3.12、PyTorch、OpenCV、NumPy、pynvml；GPU 建议 ≥8GB（本机 RTX 5070 12GB）。

```bash
# M2 冒烟
python train_v22_m2.py --smoke --gen_max 600 --spatial_upgrade 1

# M2 复现当前最佳（约 3.9h）
python train_v22_m2.py --use_generic 1 --k_labeled 16 --k_generic 4 \
  --epochs 600 --batch_size 16 --lr 0.012 --regen_every 100 \
  --gen_max 600 --spatial_upgrade 1

# M1 复现 Stage B
python train_v22_img2img.py --use_generic 1 --k_labeled 16 --k_generic 4 \
  --epochs 3000 --batch_size 1024 --lr 0.012 --regen_every 100 --gen_max 300
```

数据：

- 带标签小图：`img2img_data/*.bmp`（25 张，19 训练 / 6 考试）
- 通用图库：`D:/ai/output/images`（M1 顶层 300 张；M2 递归 400 张，370 训练 / 30 考试）
- 引擎文本数据：默认 `C:/ai/data/public/public_combined.txt`（本机缺失，可改 `--data`）

## 目录

```
P1_char_word/ P2_word_char/ P3_vision/ P3_word_attr/ P5_sentence/
P6_sent_word/ P7_cross_sent/ P8_char_sent/ P9_pipeline/ P9_sent_char/
data/ img2img_data/ utils/
train_v22_stage1.py train_v22_img2img.py train_v22_m2.py
v22_vmf_modules.py probe_vram.py probe_vram2.py
```

## 已知限制

- 引擎保存字典不完整（`abcD/abcE/abc_prime/...` 部分模块未保存），按 checkpoint 不能完整复现。
- 引擎 NaN 恢复依赖可能不存在的 `V21_best.pt`。
- M2 的 `task/σ/mask` 没有进入网络；遮补/上色指标被平凡解撑高（详见手册第 8.7/11 节）。
- 当前 M2 的空间建模仍是 patch 封闭；跨 patch 像素级、内容自适应路由、多头注意力都在被阻塞的 `transform_v2` 里。

## 安全

- 仓库不包含 token、密钥或账号信息。
- `torch.load` 仅用于本机自产 checkpoint；不要加载不可信的 `.pt`（PyTorch pickle 可执行代码）。
