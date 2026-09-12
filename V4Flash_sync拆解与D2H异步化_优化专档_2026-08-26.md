# DeepSeek-V4-Flash 优化专档：sync 拆解 → 去 pin → D2H 异步化

> 日期：2026-08-26　|　引擎：`D:\ai\v4flash_chunked_sched.py`　|　权重：`D:\ai_v4flash`（149GB / 46 shard）
> 主题：把双缓冲后的最后一个软件瓶颈（sync 1.3s）拆到底，并落地 D2H 异步化，最终 **decode -43% / MTP +57%**。

---

## 一、起点与目标

双缓冲窗口流水线（08-25）定稿后，瓶颈拆解为：

```
layer_fwd 3.8s = MoE 专家读取 2s（分布散，模型固有）+ attention 0.25s + 层内调度
sync 1.3s    = 层间同步/调度（疑是"免费"的软件瓶颈）
```

目标：**拆解 sync 1.3s**，判断是否有不必要同步可省。

---

## 二、sync 拆解探针（v4flash_sync_probe.py）

### 设计

在窗口循环的每个关键同步点插桩，拆 5 段：

```
窗口循环每轮:
  [wait_fut]     等 build_pool 线程把 .to() 提交完
  [wait_stream]  current_stream().wait_stream(copy_stream) — 等 H2D DMA 完成
  [pin]          _pin_window — pin_memory 主线程同步
  [fwd]          K 层 forward
  [d2h]          layer.cpu() 整组回 CPU（同步 D2H）
```

### decode 稳态结果（3 token 平均）

| 段 | 耗时 | 占比 | 结论 |
|---|---|---|---|
| `wait_fut` | 0.10s | 2% | 微不足道 |
| **`wait_stream`** | **0.000s** | **0%** | **双缓冲完美隐藏 H2D，GPU 从不等 DMA** |
| `pin` | 0.74s | 13% | pin_memory 主线程同步 |
| `fwd` | 3.3s | 56% | 层计算（MoE 读取为主） |
| **`d2h`** | **1.7s** | **29%** | **layer.cpu() 同步 D2H —— sync 真身最大头** |

### 关键结论

1. **附件方案 C 的"GPU 等 DMA / stream 同步过紧"假设被实测排除**：`wait_stream=0.000s`。
2. **sync 真身 = d2h（1.7s）+ pin（0.7s）**，不是 stream/event 问题。
3. d2h 是 `layer.cpu()` 整组 D2H（14.4GB/轮），带宽下限 ~8.5GB/s。

---

## 三、去 pin 优化（-8%）

### no-pin 对照

`v4flash_sync_probe.py --no-pin`（monkey-patch `_pin_window` 为 no-op）：

| 版本 | decode 稳态 |
|---|---|
| pinned | 6.0s |
| **no-pin** | **5.5s（-8%）** |

### 结论

- `wait_stream` 恒 0 → **双缓冲已把 H2D 完全隐藏，pin 不加速任何东西**。
- pin 每窗口重复 `pin_memory()`（2.4GB×6），且每次 `layer.cpu()` 后 pin 状态丢失、下轮重 pin —— **白花 0.7s**。
- **定稿：`_pin_enabled=False` 默认关**（保留函数 + 开关供回归测试）。

---

## 四、页缓存受控实验（v4flash_pgcache_probe.py）

### 动机

D2H 异步化需要全层权重 pinned 常驻（~13GB），之前**否决**理由是"会饿死页缓存，专家读盘变慢"。用户质疑该假设：专家分布散（重叠率 32%），本来就是冷读，页缓存可能无关紧要。

### 实验设计

1. baseline：跑 2 decode，统计 pread 平均耗时 / MoE 耗时 / Standby 大小
2. 顺序读 15GB shard 文件，挤掉 Standby 里的专家页
3. 再跑 2 decode，对比

### 结果

| 指标 | BASELINE | 挤压 15GB 后 | 结论 |
|---|---|---|---|
| pread 平均耗时 | 33.92ms | 31.91ms | 几乎不变 |
| decode 2 token wall | 12.63s | 12.61s | **完全不变** |
| Standby（页缓存） | **0.6GB** | 5.0GB | 引擎运行时页缓存本来就≈0 |

### 三个铁证

1. **引擎跑的时候 Standby 只有 0.6-1GB**，不是假设的 27GB —— OS 根本没缓存专家页。
2. **pread 33ms/次 ≈ 364MB/s = 磁盘随机读速度**（页缓存命中应 <1ms）→ 专家就是冷读。
3. 挤掉页缓存后 decode wall 完全不变 → **页缓存对专家读取零贡献**。

### 结论

**"D2H 异步化饿死页缓存"的担忧被实验推翻。** 专家读取是模型特性（分布散）导致的冷读为主，不依赖页缓存。D2H 异步化可行。

---

## 五、D2H 异步化（核心突破，decode -38%）

### 方案

独立 **d2h_pool（4 线程）多线程同步 D2H** 到 pinned mirror：

```python
def _offload_window(self, _g, _ge):
    """窗口层 GPU → pinned CPU（d2h_pool worker 同步 D2H，不阻塞主线程）"""
    for lid in range(_g, _ge):
        layer = self.layers[lid]
        for name, p in layer.named_parameters():
            if p.is_cuda:
                mir = getattr(p, '_cpu_pinned', None)
                if mir is None:
                    mir = torch.empty_like(p, device='cpu').pin_memory()
                    p._cpu_pinned = mir
                mir.copy_(p, non_blocking=False)   # 同步 D2H（worker 阻塞，主线程不阻塞）
                p.data = mir

# 窗口循环: D2H 阶段
self._d2h_futs[_wi] = self.d2h_pool.submit(self._offload_window, _g, _ge)

# H2D 前 / 窗口开头: 等该窗口 D2H 完成
if self._d2h_futs[_wi] is not None:
    self._d2h_futs[_wi].result()
```

### 关键设计（3 个坑，缺一不可）

| # | 坑 | 错误做法 | 正确做法 |
|---|---|---|---|
| ① | **显存复用竞态** | `mir.copy_(p, non_blocking=True)` 异步 D2H 后立即 `p.data=mir` → GPU 源释放，缓存分配器可能在拷贝完成前复用该显存 → 数据被覆盖 | **同步 copy（线程内等待）** + 线程隔离，拷贝完成才切引用 |
| ② | **流排队阻塞** | D2H 放进 `copy_stream` → D2H 1.7s 排在 H2D 前面，`wait_stream` 阻塞 | **独立 d2h_pool 线程**，D2H 与 H2D 并行 |
| ③ | **跨 forward 依赖** | 每轮重置 `_d2h_futs` → prefill 的 D2H 没人等 | **`_d2h_futs` 跨 forward 保留**（`while len < _nw: append(None)`），prefill 的 D2H 由 decode 第一轮 `result()` 等 |

### 实测结果

- **decode 5.5 → 3.4-3.5s/tok（-38%）**，d2h 段 1.7s → **0.000s**（完全隐藏）。
- 内存：pinned mirror 全量 **1515/1515 params**，显存 reserved **11.56GB < 12GB**，无 OOM。
- 正确性：贪心生成连贯无污染，MTP 正常（+4 token）。

---

## 六、MTP 基准对比（v4flash_mtp_bench.py）

100 主 token 预算（含 prefill），温度 0.7：

| 引擎 | wall | 有效 tokens | 生成阶段 | 有效 tok/s | acceptance |
|---|---|---|---|---|---|
| SYNC 备份（无 D2H 异步） | 702.6s | 187（主100+MTP87） | ~675s | 0.28 | ~87% |
| **ASYNC（D2H 异步化）** | **458.7s** | **189（主100+MTP89）** | **~431s** | **0.44（+57%）** | **89%** |

- **D2H 异步化对 MTP 净收益 +57%**（0.28→0.44 tok/s，即 3.6→2.28 s/tok）。
- **acceptance 89% 极高**（拒绝采样阈值 0.5）→ MTP 预测质量好，无需换 draft model。

---

## 七、最终成果（vs 原始引擎）

| 模式 | 原始 | 最终 | 提升 |
|---|---|---|---|
| 单 token decode | 6.0s | **3.4s** | **-43%** |
| MTP 有效 | 3.6 s/tok | **2.28 s/tok** | **-37%** |
| MTP 有效 tok/s | 0.28 | **0.44** | **+57%** |
| MTP acceptance | — | **89%** | 极高 |

**完整优化链**：层缓存 → 窗口K8 → 双缓冲 → 去pin（-8%）→ **D2H 异步化（-38% decode / +57% MTP）**

**剩余瓶颈**：`fwd 3.3s = MoE 专家读取 2s（分布散，重叠率32%，模型固有）+ attention 0.25s + 层内调度` —— **软件侧已到物理极限**（专家冷读 + D2H 带宽下限 + 149GB 磁盘 IO）。

---

## 八、工具沉淀与文件

| 文件 | 用途 |
|---|---|
| `v4flash_sync_probe.py` | sync 拆解（--no-pin 对照） |
| `v4flash_pgcache_probe.py` | 页缓存影响受控实验 |
| `v4flash_mtp_bench.py` | MTP 基准（--backup 对比同步版） |
| `v4flash_mem_check.py` | 内存 / pinned mirror / 显存检查 |
| `v4flash_chunked_sched.py` | **定稿引擎**（默认 `_pin_enabled=False` + `d2h_pool` + `_mtp_stats`） |
| `v4flash_chunked_sched_pre_d2h_async.py` | D2H 异步化前备份（同步版） |
| `v4flash_d2h_verify.py` | 正确性验证（贪心生成） |

---

## 九、教训

1. **数据用实验证伪假设**：D2H 异步化"饿死页缓存"的担忧，被受控实验（挤 15GB 页缓存 decode 不变）直接推翻。理性推理要建立在实测上。
2. **同步拷贝 + 线程隔离 优于 异步流**：异步 D2H 的 `p.data=mir` 释放 GPU 源会触发缓存分配器复用竞态；同步拷贝（放 worker 线程）简单可靠。
3. **瓶颈转移后需重新拆解，而非沿用假设**：`wait_stream=0.000s` 证明双缓冲已完美，之前以为的"stream 同步过紧"不存在，真正的大头是没被单独计时的 d2h 和 pin。
4. **32GB 内存 + 149GB 磁盘模型**：pinned 13GB 可行（专家冷读不依赖页缓存）；但 pinned 仍要控制总量（专家 LRU 14.4GB + 层 pinned 13GB ≈ 27GB，接近上限）。

---

*本文档与 `DeepSeek-V4-Flash_完整魔改记录_2026-08-16.txt`（追加章节）互为补充，后者为全量历史，本文档为 08-26 优化专档。*
