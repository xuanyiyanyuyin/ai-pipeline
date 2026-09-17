#!/bin/bash
# ═══ V4-Flash AMD(gfx1100) 适配 patch 一键脚本 ═══
# 前置: 引擎代码在 /workspace/engine/, transformers==5.16.1, 模型 FP4 版在 /v4flash_fp4
set -u
ENG=/workspace/engine
cd $ENG || exit 1
P=/opt/venv/bin/python

echo "=== P0: MXFP4 纯 PyTorch 分支 (FP4 版必需) ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_sched.py"
s = open(p, encoding="utf-8").read()
if "_dequant_mxfp4" in s:
    print("  已有, 跳过")
else:
    anchor = "def fp8_matmul("
    assert anchor in s, "fp8_matmul not found"
    add = '''_E2M1_LUT = None


def _dequant_mxfp4(x, w_i8, scale, out_dtype=torch.bfloat16):
    """纯 PyTorch MXFP4 (照官方 transformers/integrations/mxfp4.py):
    w   int8 [N, K//2]   packed E2M1 (每字节2个4bit浮点, 低nibble=偶数列)
    sc  UE8M0 [N, K//32] 解码 2^(exp-127)
    """
    global _E2M1_LUT
    if _E2M1_LUT is None or _E2M1_LUT.device != w_i8.device:
        _E2M1_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=w_i8.device)
    u8 = w_i8.view(torch.uint8)
    lo = _E2M1_LUT[(u8 & 0x0F).long()]
    hi = _E2M1_LUT[(u8 >> 4).long()]
    N, Kb = w_i8.shape
    w = torch.empty(N, Kb * 2, dtype=torch.float32, device=w_i8.device)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    exp = scale.view(torch.uint8).to(torch.int32)
    s = (exp << 23).view(torch.float32).repeat_interleave(32, dim=1)
    return x @ (w * s).to(x.dtype).t()


'''
    s = s.replace(anchor, add + anchor, 1)
    # 分派: int8 -> MXFP4
    old = '''def fp8_matmul(x, w, s, block, out_dtype=torch.bfloat16):
    """统一入口: 按 V4F_TORCH_FP8 自动选 纯PyTorch / CUDA kernel"""
    if _USE_TORCH_FP8:'''
    new = '''def fp8_matmul(x, w, s, block, out_dtype=torch.bfloat16):
    """统一入口: int8->MXFP4; 其余按 V4F_TORCH_FP8 选 纯PyTorch / CUDA kernel"""
    if w.dtype == torch.int8:
        return _dequant_mxfp4(x, w, s, out_dtype)
    if _USE_TORCH_FP8:'''
    if old in s:
        s = s.replace(old, new, 1)
        print("  MXFP4 分支 + 分派 已注入")
    else:
        print("  [warn] fp8_matmul 结构不符, 只注入函数")
    open(p, "w", encoding="utf-8").write(s)
PYEOF

echo "=== P1: 引擎参数环境变量化 ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_sched.py"
s = open(p, encoding="utf-8").read(); n = 0
reps = [
 ("self.expert_lru = ExpertLRU(self.expert_reader, max_items=1200)",
  "self.expert_lru = ExpertLRU(self.expert_reader, max_items=int(os.environ.get('V4F_LRU_ITEMS', 1200)))"),
 ("self.gpu_exp_max = 200  #", "self.gpu_exp_max = int(os.environ.get('V4F_GPU_EXP_MAX', 200))  #"),
 ("ThreadPoolExecutor(max_workers=6)   # 专家读取",
  "ThreadPoolExecutor(max_workers=int(os.environ.get('V4F_EXP_WORKERS', 6)))   # 专家读取"),
 ("ThreadPoolExecutor(max_workers=6)    # H2D预取",
  "ThreadPoolExecutor(max_workers=int(os.environ.get('V4F_BUILD_WORKERS', 6)))    # H2D预取"),
 ("ThreadPoolExecutor(max_workers=4)      # 层权重 D2H",
  "ThreadPoolExecutor(max_workers=int(os.environ.get('V4F_D2H_WORKERS', 4)))      # 层权重 D2H"),
 ("self.max_sequence_length = 1024  #", "self.max_sequence_length = int(os.environ.get('V4F_MAX_SEQ', 1024))  #"),
 ("self.chunk_size = 256           #", "self.chunk_size = int(os.environ.get('V4F_CHUNK', 256))           #"),
 ("self.kv_chunk_size = 512        #", "self.kv_chunk_size = int(os.environ.get('V4F_KV_CHUNK', 512))        #"),
 ("self.prefill_chunk_size = 1024  #", "self.prefill_chunk_size = int(os.environ.get('V4F_PREFILL_CHUNK', 1024))  #"),
]
for a, b in reps:
    if a in s and "os.environ.get" not in a:
        s = s.replace(a, b, 1); n += 1
open(p, "w", encoding="utf-8").write(s); print(f"  {n}/{len(reps)} 处")
PYEOF

echo "=== P2: finegrained_fp8 kernel -> 纯 PyTorch stub ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_sched.py"
s = open(p, encoding="utf-8").read()
if "_TorchFP8Kernel" in s:
    print("  已 patched, 跳过")
else:
    anchor = "    _ff8.FP8Linear.forward = _fp8_linear_forward_torch"
    assert anchor in s, "anchor missing"
    add = anchor + '''


if FP8_KERNEL is None:
    def _dequant_linear(x, w, scale_inv, block_size, out_dtype=None):
        N, K = w.shape
        bn, bk = int(block_size[0]), int(block_size[1])
        dt = x.dtype
        ww = w.to(dt); ss = scale_inv.to(dt)
        if N % bn == 0 and K % bk == 0:
            ww = (ww.view(N // bn, bn, K // bk, bk) * ss.view(N // bn, 1, K // bk, 1)).view(N, K)
        else:
            ww = ww * ss.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:N, :K]
        return torch.nn.functional.linear(x, ww)

    class _TorchFP8Kernel:
        @staticmethod
        def matmul(x, w, scale_inv, block_size, out_dtype=None, activation_scale=None):
            return _dequant_linear(x, w, scale_inv, block_size, out_dtype or x.dtype)
        @staticmethod
        def grouped_matmul(x, w, scale_inv, offsets=None, tokens_per_expert=None, block_size=None):
            G, N, K = w.shape
            if offsets is None:
                tok = x.shape[0] // G
                offs = [i * tok for i in range(G + 1)]
            else:
                offs = [0] + [int(o) for o in offsets]
            out = torch.empty(x.shape[0], N, dtype=x.dtype, device=x.device)
            bs = block_size if block_size is not None else (128, 128)
            for g in range(G):
                s0, s1 = offs[g], offs[g + 1]
                if s1 > s0:
                    out[s0:s1] = _dequant_linear(x[s0:s1], w[g], scale_inv[g], bs, x.dtype)
            return out
    _ff8.load_finegrained_fp8_kernel = lambda: _TorchFP8Kernel
    print("  kernel stub 已注入")
'''
    open(p, "w", encoding="utf-8").write(s.replace(anchor, add, 1))
PYEOF

echo "=== P3: FP8Linear.forward 纯 PyTorch 加速 ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_sched.py"
s = open(p, encoding="utf-8").read()
old = """        w = (self.weight.to(torch.float32)
             * self.weight_scale_inv.to(torch.float32)
               .repeat_interleave(bs_n, 0).repeat_interleave(bs_k, 1)[:N, :K])
        out = torch.nn.functional.linear(input.float(), w.to(input.dtype).t() if False else w)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input.dtype)"""
new = """        dt = input.dtype
        w = self.weight.to(dt); s = self.weight_scale_inv.to(dt)
        if N % bs_n == 0 and K % bs_k == 0:
            w = (w.view(N // bs_n, bs_n, K // bs_k, bs_k) * s.view(N // bs_n, 1, K // bs_k, 1)).view(N, K)
        else:
            w = w * s.repeat_interleave(bs_n, 0).repeat_interleave(bs_k, 1)[:N, :K]
        out = torch.nn.functional.linear(input, w, self.bias)
        return out.to(input.dtype)"""
if old in s:
    s = s.replace(old, new, 1); print("  FP8Linear 已加速")
else:
    print("  已是新版/结构不同, 跳过")
open(p, "w", encoding="utf-8").write(s)
PYEOF

echo "=== P4: 采样 NaN 防御 (ROCm multinomial 会崩 HIP 上下文) ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_sched.py"
s = open(p, encoding="utf-8").read()
old = """                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    nxt = idx.gather(-1, torch.multinomial(probs, 1))"""
new = """                    _s = probs.sum(dim=-1, keepdim=True)
                    if torch.isfinite(_s).all() and (_s > 0).all():
                        probs = probs / _s
                        nxt = idx.gather(-1, torch.multinomial(probs, 1))
                    else:
                        print("[warn] probs NaN/Inf, 回退 greedy", flush=True)
                        nxt = idx[..., :1]"""
if old in s:
    s = s.replace(old, new, 1); print("  采样防御已加")
else:
    print("  已 patched/结构不同, 跳过")
open(p, "w", encoding="utf-8").write(s)
PYEOF

echo "=== P5: Server 保留 GPU 缓存跨请求 ==="
$P - <<'PYEOF'
p = "/workspace/engine/v4flash_chunked_server.py"
s = open(p, encoding="utf-8").read()
if "V4F_KEEP_GPU_CACHE" in s:
    print("  已 patched, 跳过")
else:
    a = "_engine.gpu_experts.clear()"
    if a in s:
        s = s.replace(a, "(_engine.gpu_experts.clear() if os.environ.get('V4F_KEEP_GPU_CACHE') != '1' else None)", 1)
    a2 = "                        torch.cuda.empty_cache()\n                    except Exception:"
    if a2 in s:
        s = s.replace(a2, "                        if os.environ.get('V4F_KEEP_GPU_CACHE') != '1':\n                            torch.cuda.empty_cache()\n                    except Exception:", 1)
    open(p, "w", encoding="utf-8").write(s); print("  已改")
PYEOF

echo "=== 语法校验 ==="
$P -c "import ast; ast.parse(open('/workspace/engine/v4flash_chunked_sched.py',encoding='utf-8').read()); ast.parse(open('/workspace/engine/v4flash_chunked_server.py',encoding='utf-8').read()); print('  SYNTAX OK')"

echo "=== 启动脚本 ==="
cat > /workspace/start_engine.sh <<'SEOF'
#!/bin/bash
cd /workspace/engine
export V4F_BASE=/workspace
export V4F_MODEL_DIR=/v4flash_fp4
export V4F_SRC_DIR=/workspace/engine/_v4flash_src
export V4F_TORCH_FP8=1
export V4F_GPU_EXP_MAX=1200
export V4F_LRU_ITEMS=1500
export V4F_KEEP_GPU_CACHE=1
export V4F_EXP_WORKERS=6
export V4F_BUILD_WORKERS=6
export V4F_D2H_WORKERS=4
exec /opt/venv/bin/python v4flash_chunked_server.py
SEOF
chmod +x /workspace/start_engine.sh
echo "  /workspace/start_engine.sh 就绪 (FP4 版路径, LRU=1500/37GB)"
echo "=== ALL PATCHES DONE ==="
