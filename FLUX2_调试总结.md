# FLUX.2 三层搬运(img2img/txt2img)调试总结

> 日期:2026-09-07
> 状态:**已定位根因,等低峰(12点)复跑验证**
> 相关脚本:`D:\ai\run_f2_full.py`(已改,备份 `run_f2_full.py.bak`)

---

## 一、问题现象

- 512×512 出图 **糊、平淡、偏白、油画质感**,背景布满彩色噪点斑
- 概念对(橘猫)、暖色调对,但**形态不清晰**,像未收敛

## 二、诊断方法(证明不是代码/数学错)

**逐行 diff 官方 `Flux2Pipeline.__call__` 与我脚本的 forward + VAE 解码:**

| 环节 | 我的脚本 | 官方 | 结论 |
|---|---|---|---|
| transformer 调用 | `timestep/1000`、`guidance` 张量 expand、切片 `[:, :latents.size(1)]`、txt_ids/img_ids | 完全一致 | ✅ 一致 |
| VAE 解码 | bn反归一化 + `_unpack_latents_with_ids` + `_unpatchify_latents` | 完全一致 | ✅ 一致 |

**结论:模型数学、text embeds、VAE 全部是通的,forward 与官方逐字节一致。**
→ 糊的根因**不在模型代码/数学**,而在**参数配置偏离官方**。

## 三、根因(4 个参数,按影响排序)

| 参数 | 我的脚本(旧) | 官方 | 影响 |
|---|---|---|---|
| **步数** | **8** | 默认 **50**(example 也用 50) | **主因:8步撑不起60G模型细节** |
| **shift/mu** | **`calculate_shift`(FLUX.1)= 0.63** | **`compute_empirical_mu`(FLUX.2)≈ 1.86 @ 28步** | **真bug:exp(mu) 1.88 vs 6.4,差 3.4×** |
| 分辨率 | 512×512 | 默认 1024×1024(best results) | 次级:低于原生训练分辨率 |
| guidance | 3.5 | 默认 4.0 | 轻微 |

**scheduler 关键点**:`FLUX.2-dev/scheduler/scheduler_config.json` 为
`FlowMatchEulerDiscreteScheduler`,配置 `use_dynamic_shifting:true`、`time_shift_type:exponential`、`shift:3.0`。
已读源码确认 `set_timesteps` 在 `use_dynamic_shifting=True` 时**使用传入的 mu**,**不被 config 的 shift=3.0 覆盖**。
所以 mu 传错是真的有影响,而且影响很大。

> ⚠️ **核心教训**:FLUX.2 的 shift 要用 `compute_empirical_mu(image_seq_len, num_steps)`,
> **不能**沿 用 FLUX.1 的 `calculate_shift(256, 4096, 0.5, 1.15)`。两个函数数值完全不相等。

## 四、已修改(run_f2_full.py)

```
旧: from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
新: from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

旧: mu=calculate_shift(img_seq,256,4096,0.5,1.15)
    ts,_ = retrieve_timesteps(pipe.scheduler, 8, 'cuda', sigmas=np.linspace(1.0,1.0/8,8), mu=mu)
新: NSTEPS=28
    mu=compute_empirical_mu(image_seq_len=img_seq, num_steps=NSTEPS)
    ts,_ = retrieve_timesteps(pipe.scheduler, NSTEPS, 'cuda', sigmas=np.linspace(1.0,1.0/NSTEPS,NSTEPS), mu=mu)
```
- guidance 保持 **3.5 未动**(这次只变 shift+步数,方便归因)
- 备份已建:`run_f2_full.py.bak`

## 五、待验证(低峰 12 点后跑)

```bash
cd /d/ai && D:/ai/python/python.exe run_f2_full.py
```
- 28 步 ≈ 之前 8 步的 **3.5 倍** forward,会明显变慢,耐心等
- 输出覆盖:`D:\ai\output\images\flux2_full_test.png`

## 六、判定决策树

- ✅ **28 步后变清晰** → **100% 确认是参数问题**(步数少 + shift错),forward 不用再动。可收工。
- ❌ **28 步仍糊** → 先加到 **50 步**排除步数因素;50 步仍糊再去**逐层对齐**(官方单块 vs 我单块,照 v4flash 方法)。

## 七、后续可迭代(确认参数问题后)

1. 步数 28 → 50(官方默认,更清晰)
2. 分辨率 512 → 1024(官方最佳)
3. guidance 3.5 → 4.0(官方默认)
4. 逐项 A/B,别一次全改,方便归因。
