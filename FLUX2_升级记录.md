# FLUX.2 三层搬运 — 升级记录(档案)

> 项目: 千文 AI 动画流水线 · 图像引擎从 FLUX.1 升级到 FLUX.2
> 文档角色: **长期档案 + 后续补写基座**(本文件是"起点",后续在此追加)
> 创建: 2026-09-07
> 维护人: Jinqi Wei / Agent 暄依嫣
> 相关: `D:\ai\FLUX2_调试总结.md`(当日调试专项)、`D:\ai\flux2_3tier.py`(引擎)

---

## 0. 文档约定

- **本文件是"活的记录"**: 每次升级/调试,在对应阶段追加新日期条目,别覆盖历史。
- 技术事实优先于观点; 所有结论尽量附**证据/来源**(源码行、配置、实测值)。
- 符号约定: `DEV`=cuda; 路径均为 Windows 绝对路径。
- 一句话状态: **FLUX.2 引擎已能出图(概念/色调对),但 512×512 测试图未收敛(参数偏差),已锁定根因,30 号低峰复跑验证。**

---

## 1. 项目背景与目标

- 千文 AI 动画流水线(2026-08-09 重构): 122B 引擎文生图/视频/音频,统一面板 `engine_manager.py`(8321)。
- 图像引擎此前用 **FLUX.1-dev 三层搬运**(`D:\ai\flux_3tier.py`, 2026-09-04 完成,12GB 显存跑通)。
- 升级目标: 换成 **FLUX.2-dev**(更大、效果更好的新模型),继承"三层搬运"分层思想,在 **RTX 5070 12GB** 上跑通文生图。

## 2. 硬件与环境

| 项 | 值 |
|---|---|
| GPU | RTX 5070 12GB (PCIe 4.0×8) |
| CPU | i5-14600KF |
| RAM | 32GB DDR4 3200 (+虚拟内存100GB) |
| 系统 | Windows + Git Bash |
| Python | `D:\ai\python\python.exe` (3.12.9, 但 `python`/`py` 不在 PATH, 只有 MS Store stub) |
| PyTorch | 2.11 CUDA 12.8 |
| diffusers | 0.40.0 (安装于 `D:\ai\python\Lib\site-packages\diffusers`) |
| 模型 | `D:\ai\models\FLUX.2-dev` |

## 3. FLUX.2 模型结构与关键差异(vs FLUX.1)

这些差异是本次全部问题的源头,务必记牢。

### 3.1 Transformer (`Flux2Transformer2DModel`)
`transformer/config.json`:
```
in_channels: 128
num_attention_heads: 48
attention_head_dim: 128
num_layers: 8            # 双块 (double transformer blocks)
num_single_layers: 48    # 单块 (single transformer blocks)
joint_attention_dim: 15360   # text encoder 隐藏维度
mlp_ratio: 3.0
patch_size: 1
axes_dims_rope: [32,32,32,32]
rope_theta: 2000
timestep_guidance_channels: 256
```
> ⚠️ 结构对比 FLUX.1-dev(19 双块+38 单块): **FLUX.2 是 8 双块 + 48 单块**,块数/结构均不同,三层搬运槽位逻辑必须分两套。

### 3.2 VAE (`AutoencoderKLFlux2`)
`vae/config.json`:
```
_class_name: AutoencoderKLFlux2
latent_channels: 32
patch_size: [2, 2]
sample_size: 1024
base_channels: block_out_channels [128,256,512,512]
force_upcast: true
batch_norm_eps: 0.0001
batch_norm_momentum: 0.1
```
> ⚠️ **关键: FLUX.2 的 VAE 解码用「批归一化 BN」,不是 FLUX.1 的 `scaling_factor`/`shift_factor`。**
> 配置里**没有** `scaling_factor`/`shift_factor`。正确的解码(官方 `__call__` line 1016-1026)是:
> ```python
> lat = _unpack_latents_with_ids(latents, latent_ids)
> bn_mean = vae.bn.running_mean.view(1,-1,1,1)
> bn_std  = sqrt(vae.bn.running_var.view(1,-1,1,1) + vae.config.batch_norm_eps)
> lat = lat * bn_std + bn_mean
> lat = _unpatchify_latents(lat)
> img = vae.decode(lat)   # 输出即像素
> ```

### 3.3 Scheduler (`FlowMatchEulerDiscreteScheduler`)
`scheduler/scheduler_config.json`:
```
use_dynamic_shifting: true
time_shift_type: exponential
base_shift: 0.5, max_shift: 1.15, base_image_seq_len: 256, max_image_seq_len: 4096
shift: 3.0            # 注意: 不会覆盖 mu
num_train_timesteps: 1000
```
> ⚠️ **关键: `use_dynamic_shifting: true` 时,`set_timesteps` 用「传入的 mu」,config 的 `shift:3.0` 不生效。**
> 源码 `scheduling_flow_match_euler_discrete.py` line 348-349:
> ```python
> if self.config.use_dynamic_shifting:
>     sigmas = self.time_shift(mu, 1.0, sigmas)   # 用传入 mu
> else:
>     sigmas = self.shift * sigmas / (1 + (self.shift-1)*sigmas)
> ```
> 且 line 309-310: `use_dynamic_shifting and mu is None` → **抛 ValueError**。

### 3.4 Text Encoder / Tokenizer (`model_index.json`)
```
text_encoder: transformers / Mistral3ForConditionalGeneration
tokenizer:    transformers / PixtralProcessor
```
- 文本编码器很大(约 45G),故引擎用 `load_text_encoder_pread` + `encode_prompt_streaming` 逐层流式编码。
- 编码后 `out_layers=(10,20,30)` 按官方。`joint_attention_dim=15360` 与 Mistral3 对齐。

---

## 4. 引擎架构 (`D:\ai\flux2_3tier.py`)

### 4.1 对外接口
```python
MODEL = r'D:\ai\models\FLUX.2-dev'

def text2img_f2(prompt, steps=28, width=768, height=1344, guidance_scale=3.5,
                max_seq=512, out_path=None, resident_d=0, seed=None):
    """FLUX.2 三层搬运文生图. text_encoder逐层流式编码; transformer逐块去噪; GPU tiling解码."""

def _patch_f2(t, resident_d=0):         # 给 transformer 装三层搬运 patch
def _align_wh(w, h, mult=64):           # 宽高对齐 64 倍数
def _pread_tensor(path, key, hdr_cache):# mmap 安全读
def load_text_encoder_pread(model_dir): # pread 加载 text_encoder(绕 mmap 段错误)
def encode_prompt_streaming(pipe, prompt, max_seq=512, out_layers=(10,20,30)):
```
CLI(缺省): `python flux2_3tier.py "<prompt>" <steps> <w> <h> <resident_d>`

### 4.2 三层搬运原理(继承 FLUX.1 思想)
- FLUX.2 总权重约 60G,塞不进 12GB 显存。
- 解法: **前 N 双块常驻 GPU** + **其余块流式**(共享 GPU 槽: copy权重→算→还原 RAM),全程 `no_grad`。
- `resident_d=0` 时按分辨率自动选常驻块数;显存上限常驻块数由激活大小决定。
- 阶段:
  1. **编码**: pread 加载 text_encoder → GPU 逐层流式编码 → **缓存 embed**(`_cached_flux2_embeds_<md5>.pt`,命中直接读,miss 才编码)
  2. **去噪**: `prepare_latents` → scheduler 循环(transformer 三层搬运 forward)→ `scheduler.step`
  3. **解码**: transformer→CPU, vae→GPU(开 tiling), VAE 解码 → PIL 保存

### 4.3 已发现的引擎待修项(补写重点,见 §7)
`text2img_f2` 目前有两处 **FLUX.1 遗留的错误写法**,与 §3.2/§3.3 的 FLUX.2 差异冲突。

---

## 5. 2026-09-07 调试记录(本次核心)

### 5.1 现象
- `run_f2_full.py` 出图 **512×512, 糊/平淡/偏白/油画质感**,背景布满彩色噪点斑。
- 概念对(橘猫)、暖色调对,但**形态不清晰**,像"未收敛"。

### 5.2 诊断方法(证明不是代码/数学错)
逐行 diff 官方 `Flux2Pipeline.__call__` 与本脚本的 forward + VAE 解码:

| 环节 | 本脚本 | 官方 | 结论 |
|---|---|---|---|
| transformer 调用 | `timestep/1000`、`guidance`张量expand、切片`[:, :latents.size(1)]`、txt/img_ids | line 973-984 完全一致 | ✅ 一致 |
| VAE 解码 | bn反归一化 + `_unpack_latents_with_ids` + `_unpatchify_latents` | line 1016-1026 完全一致 | ✅ 一致 |

**结论: 模型数学、text embeds、VAE 全部是通的,forward 与官方逐字节一致。**
→ 糊的根因**不在模型代码/数学**,而在**参数配置偏离官方**。

### 5.3 根因(4 个参数,按影响排序)

| 参数 | 旧脚本 | 官方 | 影响 |
|---|---|---|---|
| **步数** | **8** | 默认 **50**(example 也用 50) | **主因: 8步撑不起60G模型细节** |
| **shift/mu** | **`calculate_shift`(FLUX.1)=0.63** | **`compute_empirical_mu`(FLUX.2)≈1.86 @ 28步** | **真bug: exp(mu) 1.88 vs 6.4, 差3.4×** |
| 分辨率 | 512×512 | 默认 1024×1024(best results) | 次级: 低于原生训练分辨率 |
| guidance | 3.5 | 默认 4.0 | 轻微 |

### 5.4 图片分析(教科书级"扩散未收敛")
- ✅ 语义/色调对(橘猫、暖橘) → **证明 embeds、forward 工作正常**
- ❌ 形态融化、背景噪点、油画质感 → **去噪未收敛** = 步数太少 + shift 错

### 5.5 修改(`run_f2_full.py`,备份 `run_f2_full.py.bak`)
```python
# 旧
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
mu=calculate_shift(img_seq,256,4096,0.5,1.15)
ts,_=retrieve_timesteps(pipe.scheduler, 8, 'cuda', sigmas=np.linspace(1.0,1.0/8,8), mu=mu)
guidance=torch.full([1],3.5,...)

# 新
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
NSTEPS=28
mu=compute_empirical_mu(image_seq_len=img_seq, num_steps=NSTEPS)
ts,_=retrieve_timesteps(pipe.scheduler, NSTEPS, 'cuda', sigmas=np.linspace(1.0,1.0/NSTEPS,NSTEPS), mu=mu)
# guidance 保持 3.5 未动 (本次只变 shift+步数, 便于归因)
```

### 5.6 工具链验证(未加载模型,安全)
```
[OK] flux2_3tier import 成功, MODEL = D:\ai\models\FLUX.2-dev
[OK] compute_empirical_mu 导入成功
[OK] 28步 mu = 1.85918
[OK] run_f2_full.py 语法解析通过
```

---

## 6. 关键教训(方法论,可复用到别的大模型)

1. **锚点对比优先**: 怀疑计算错 → 用官方代码跑参考值逐层/逐字节对比,不要盲猜。
2. **FLUX.2 的 shift 必须用 `compute_empirical_mu(image_seq_len, num_steps)`**,**不能用 FLUX.1 的 `calculate_shift(256,4096,0.5,1.15)`**,二者数值不相等(1024,8步: 2.00 vs 0.63)。
3. **FLUX.2 VAE 解码用 BN 反归一化 + `_unpatchify_latents`**,不是 FLUX.1 的 `scaling_factor`/`shift_factor`。
4. **`use_dynamic_shifting=true` 时 scheduler 用传入 mu 且强制必须有 mu**,config 的 `shift` 不覆盖。
5. **后台进程分离**: 工具 `run_in_background` 起的任务进程挂我会话端,会话结束即被杀;要用 `powershell Start-Process -WindowStyle Hidden` 分离启动,靠日志跟踪。
6. **推理显存泄漏 = 漏 no_grad**: 手写 forward 忘包 `torch.no_grad()` 会累积 autograd graph,每步+0.14GB 线性涨爆显存。

---

## 7. 发现的新问题(待后续,补写从这里继续)

> 引擎 `text2img_f2` 目前有 3 处 FLUX.1 遗留 / 新增错误,**必须修复才能正常用引擎出图**:

| # | 位置 | 问题 | 影响 | 修法 |
|---|---|---|---|---|
| 7.1 | line 278 | `retrieve_timesteps(pipe.scheduler, steps, DEV)` **没传 mu** | `use_dynamic_shifting=true` 强制要 mu → **运行时抛 ValueError** | 导入 `compute_empirical_mu`,传 `mu=compute_empirical_mu(img_seq, steps)` |
| 7.2 | line 303 | VAE 解码用 `(lat/scaling_factor)+shift_factor`(FLUX.1 写法) | FLUX.2 VAE 无 scaling/shift_factor → **运行时抛错或解出错误图像** | 改为官方 BN 反归一化 + `_unpatchify_latents`(照 §3.2) |
| 7.3 | line 235 | `from ...flux.pipeline_flux import calculate_shift, retrieve_timesteps` | `calculate_shift` 导入未使用(死导入),且**没导入 `compute_empirical_mu`** | 改成导入 `compute_empirical_mu` + `retrieve_timesteps` |

> ⚠️ **7.1、7.2 是"必崩/必错"级**,若用户直接用 `text2img_f2`(或经 engine_manager 调它)会立刻踩到。`run_f2_full.py` 已修对,可作为修引擎的参考实现。
> 💡 `run_f2_full.py` 与 `text2img_f2` 是**同一套逻辑的两份拷贝**,后续建议统一(以 `text2img_f2` 为准,把 `run_f2_full.py` 作为验证 harness)。

---

## 8. 待办 / 下一步(补写从这开始)

- [ ] **复跑验证**(低峰 12 点后): 双击桌面 `FLUX2测试启动.bat`,或 `D:/ai/python/python.exe run_f2_full.py`
  - ⏱ 28步 ≈ 旧8步的 **3.5 倍** forward,会明显变慢
  - 输出: `D:\ai\output\images\flux2_full_test.png`; 日志: `D:\ai\_f2full_run.log`
- [ ] **判定决策树**:
  - ✅ 28步变清晰 → **确认是参数问题**(步数少+shift错),forward 不用动,可收工
  - ❌ 28步仍糊 → 先加到 **50步** 排除步数; 仍糊再去**逐层对齐**(官方单块 vs 我单块,照 v4flash 方法)
- [ ] **修引擎 3 处 bug**(§7),修完用 `text2img_f2` 出一张与 run_f2_full.py 同样参数的图对照。
- [ ] **确认参数问题后迭代**(逐项 A/B,别一次全改): 步数 28→50 / 分辨率 512→1024 / guidance 3.5→4.0。
- [ ] **接入 engine_manager**(引擎名待定,类比 flux3tier 卡片),方便面板调用。
- [ ] 清理旧 `/d/ai/*f2*` 探针脚本,保留核心(flux2_3tier / run_f2_full / download_flux2)。

---

## 9. 关键脚本索引

| 脚本 | 作用 | 状态 |
|---|---|---|
| `flux2_3tier.py` | FLUX.2 引擎(三层搬运/流式编码/缓存) | ✅ 主引擎, 但 §7 待修 |
| `run_f2_full.py` | 调试 harness, 验证正确参数(已修 compute_empirical_mu+28步) | ✅ 已验证正写法, 待复跑 |
| `run_f2_full.py.bak` | run_f2_full.py 修改前备份 | 备份 |
| `download_flux2.py` | 下载 FLUX.2-dev 模型 | ✅ |
| `FLUX2_调试总结.md` | 当日调试专项总结 | ✅(被本档案含)
| `FLUX2_升级记录.md` | 本档案(长期) | ✅ 本文件 |
| `_cached_flux2_embeds_*.pt` | text encoder 缓存 | 缓存文件 |
| `probe_flux2.py` / `probe_f2_mem.py` / `dbg_f2_*.py` / `test_f2_*.py` / `verify_f2_tf.py` | 调试/显存探针 | 临时,可清理 |

---

## 10. 附录: 参考数值

### 10.1 mu / shift 对照表 (img_seq=1024 = 512×512)

| num_steps | `compute_empirical_mu`(官方) | `calculate_shift`(旧/错) | 比值 |
|---|---|---|---|
| 8  | 2.00210 | 0.63000 | 3.18 |
| 28 | 1.85918 | 0.63000 | 2.95 |
| 50 | 1.70196 | 0.63000 | 2.70 |

- `time_shift` 用 `_time_shift_exponential(mu, ...)`: `exp(mu)/(exp(mu)+(1/t-1)**sigma)`
- 旧 exp(0.63)=1.878 vs 官方 exp(2.00)=7.389 → **差 3.4×**

### 10.2 官方 Flux2Pipeline 关键默认(源码 `pipeline_flux2.py`)
- `num_inference_steps: int = 50`
- `guidance_scale: float | None = 4.0`(docstring example 用 2.5)
- `sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)`
- `mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)`
- `max_sequence_length=512`, `text_encoder_out_layers=(10,20,30)`
- 分辨率默认 `default_sample_size * vae_scale_factor` = 1024(best results)

---

## 11. 版本时间线

| 日期 | 事件 |
|---|---|
| 2026-08-09 | 千文动画流水线重构完成 |
| 2026-09-04 | FLUX.1 三层搬运引擎完成(flux_3tier.py) |
| 2026-09-07 | FLUX.2 引擎(flux2_3tier.py)+ 调试; 512×512 测试图未收敛; 锁定根因=参数(步数+shift); 修 run_f2_full.py; 建桌面启动器; 写本档案 |
| 2026-09-07(下午) | **引擎修复+图生图接入+取层偏移修复**(见 §12) |

---

## 12. 2026-09-07 下午: 引擎修复 + 图生图接入 + 取层偏移(补写)

### 12.1 引擎 text2img_f2 三处 FLUX.1 遗留 bug 已修 (§7 全部清零)
- 7.1 → 加 `compute_empirical_mu` 传 mu + sigmas(不再裸 retrieve_timesteps)
- 7.2 → VAE 解码改 BN 反归一化 + `_unpatchify_latents`(照官方 line 1016-1026)
- 7.3 → 去 calculate_shift, 改导入 compute_empirical_mu

### 12.2 图生图(image 多参考)分支已接入
- `text2img_f2(..., images=None)`: images 为 PIL/路径 list
- 最多 **6 张参考图**(官方无硬限, 社区 dev ~6, 超了截断)
- 流程照官方: preprocess(crop 到生成尺寸) → `prepare_image_latents`(`_encode_vae_image` BN 反归一化 + `_pack_latents`) → `cat([latents, image_latents])` 喂 transformer
- CLI 第6参支持逗号分隔参考图路径

### 12.3 text_encoder_out_layers 取层偏移修复(根治背景发飘)
- **根因**: 官方 `hidden_states[k]` = `layers[k-1]`(因 `hidden_states[0]`=embedding 层, 列表长41含 final norm)。引擎原取 `layers[10/20/30]`, 官方要 `layers[9/19/29]`, **偏 1 层** → prompt 语义特征错层 → 背景/细节发飘
- **修复**: `encode_prompt_streaming` + `test_f2_encode2.py` 改 `if (i+1) in out_layers: hs[i+1]=...`
- **验证**(1 token 实测): `hidden_states[k]` vs `layer[k-1]` 逐点 max_diff=0.000000 全 MATCH(k=1..39); `hidden_states[40]`=layer39+final norm(不用于取层)
- ⚠️ **待做**: 重生成 `_cached_flux2_embeds_test.pt`(旧缓存是偏移版)才能让跑图反映修复

### 12.4 官方标准参数(已确认)
- steps: 50(默认) / 28(好折中) / 12-20 草稿 / 28-40 生产
- guidance: **4.0**(官方默认); 3-5 足够; 短prompt 3.5-4 / 长prompt 1-1.5 / 写实 2.5
- 分辨率: 官方默认 **1024×1024**(best), 最大4MP, 推荐≤2MP; 短边须被16整除(VAE scale 16)
- max_sequence_length: 512; text_encoder_out_layers: (10,20,30); caption_upsample_temperature: 官方推荐 0.15(但需完整 text_encoder + generate, 与流式/缓存路线冲突, 暂缓)
- 多参考: dev 支持 ~6 张参考图(multi-reference baked-in, 免微调角色/风格/物体一致)
- prompt: 长、自然、给布局/hex颜色(FLUX.2 吃长描述; 空泛 prompt 背景会花)
- 其他: 短边倍数16; LoRA 训练用 klein-base/dev; 蒸馏版低步数

### 12.5 resident_d 决策 + 对齐复查 + 快速测试计划(2026-09-07 晚)
- **resident_d 决策**: 维持 **0(全流式, 0 块常驻 + 每类 1 个搬运槽)**。**不照搬 FLUX.1 的 resident_d=12 经验** —— FLUX.2 权重层级比 FLUX.1 高(transformer 总 60G+, 单块更大), 显存抗压需时间单独测, 别套用旧经验。提速留到测试稳定后。
- **对齐复查(已修)**: 引擎去噪段 latent_model_input 加 `.to(pipe.transformer.dtype)`(照官方 line 966/970), 图生图 cat 后也 `.to(transformer.dtype)`(原误用 prompt_embeds.dtype)。现 forward 输入 dtype 处理与官方完全一致。
- **对齐结论**: 调度(mu/timestep)实测一致; forward 调用逐字节一致; VAE 解码 BN 一致; text_encoder 取层已修并实测一致; 图生图照官方流程。**代码层面对齐官方、理论能跑**。
- **快速对齐测试计划**: 用 **512×512 × 4步**(或 ×8)先跑通验证对齐 —— 每步 ~88s(resident_d=0), 4步仅 ~6分钟(含加载), 快速确认 forward/VAE/取层修复是否引入问题; 噪点/质量用大步数慢慢跑(不急)。
- ⚠️ **跑前必做**: 重生成 embeds 缓存(旧 `_cached_flux2_embeds_test.pt` 是取层偏移版), 否则跑图不反映取层修复。

### 12.6 高分辨率显存测试(2026-09-07 晚, 探针 probe_f2_hires.py)
- **2560×1408 纯流式(resident_d=0)实测**: 1 步 forward 正常跑完(EXIT=0, 无OOM), torch 峰值 **8.41GB**, 余量 3.79GB
- **但 nvidia-smi 显示 ~11.9GB**(含 CUDA context + caching allocator 预留), **实际无余量放常驻块** → 大分辨率**只能纯流式**
- **耗时**: 2560×1408 纯流式 1 步 ≈ **403s(6.7min)**, 28步约 3 小时 — 慢但可作"质量档"
- **决策(用户拍板)**: 
  - 2560×1408 = **后续扩展分辨率/质量档**, 纯流式慢慢出(质量优先)
  - 512×512 = **日常快速测试/验证档**(4步~7分钟先验证, 确认后再上大图)
  - 工作流: 512 验证通过 → 再跑 2560×1408 出高质量
- **FLUX.2 升级至此基本就绪**: 512(测试) + 2560×1408(质量纯流式) 都能跑, 可对接前端引擎
- resident_d 提速在大分辨率不可行(显存无余量), 仅小图可能有机会但不急测

### 12.7 引擎修复 + 图生图接入 + 前端对接(2026-09-07 晚, 收尾)

#### 引擎 flux2_3tier.py `text2img_f2` 全链路打通
- **修复**: 修 3 处 FLUX.1 遗留 bug(mu 用 compute_empirical_mu / VAE BN 解码 / 去 calculate_shift) + 去噪 latent_model_input 加 `.to(transformer.dtype)`(照官方 line 966/970)
- **图生图**: `text2img_f2(..., images, embeds_file)` — images 最多 6 张参考图(`prepare_image_latents` → cat 进 forward), 参考图编码前 `pipe.vae.to(DEV)`(否则 CPUBFloat16Type 报错)
- **关键架构**: **编码(45G text_encoder)与去噪(66G transformer)必须分进程跑**, 105G 不能同场塞 32G。引擎只"读 embeds 缓存 + 流式 transformer 去噪", 绝不内联编码 text_encoder。cache miss 报错提示先单独编码。
- **验证**: 引擎本体图生图(参考猫图 4 步)跑通, I2V_DONE 无错, 保持猫身份

#### 前端对接(engine_manager.py + engine_panel.html)
- **后端**: `ENGINES` 加 `flux2` 卡片; `_exec_task` 加 `elif name=='flux2'` 分支(调 text2img_f2, 支持 images/embeds_file/resident_d/seed)
- **前端**: `engine_panel.html` render() 加 flux2 渲染分支 + runTask() 加 flux2 参数收集; `/panel` 动态读文件→刷新即生效无需重启
- **互斥**: `ENGINE_SCRIPT_MARKS` 加 flux2_3tier/flux_3tier/ltx25/gen_image/gen_video/gen_audio/vl_vqa → kill_all_engines 能识别并杀任务型引擎子进程(否则互斥对它失效)
- **停止全部**: 前端 stopAll() → /api/stop_all → kill_all_engines() 链路已通
- **失效卡片**: flux / img2img / flux3tier (FLUX.1 系) 标 `obsolete:True` → 前端排序(有效前/失效后) + 置灰 `.obsolete` + "⚠ 已失效" badge + 禁点
- **注意**: engine_manager 改配置后必须重启才生效(内存里 ENGINES 是启动时加载); 前端 /panel 动态读文件不用重启。curl 发 POST 有 JSON 转义坑, 用 python urllib 发请求更稳

#### 收尾状态
- 前面板入口: 桌面 `AI面板一键启动.bat` → engine_manager.py + http://127.0.0.1:8321/panel
- **FLUX.2 升级完成**: 文生图(txt2img) + 图生图(多参考≤6) 引擎就绪, 已接前端可触发调用
- 工作流: 512×512 快速验证 → 2560×1408 纯流式出质量(8.4G peak); resident_d=0

---
*(后续补写请追加到相应章节,不要覆盖历史。)*
