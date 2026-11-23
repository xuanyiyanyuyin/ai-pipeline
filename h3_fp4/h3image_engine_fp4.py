# -*- coding: utf-8 -*-
"""H3-Image FP4 W4A4 streaming engine (NEW file; BF16 engine untouched).
Reuses h3_engine pure ops + h3image_engine layout/cond/VAE. FFN+attn via verified
_scaled_mm_v2 NVFP4 GEMM; AdaLN from bit-exact precomputed cache; pinned RAM + GPU residency.
Usage: python h3image_engine_fp4.py --run --size 512 --min_area 1024 --seed 42 --resident 40
"""
import os, sys, json, time, math, glob, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch._C import _ScalingType as ST, _SwizzleType as SW
sys.path.insert(0, r"D:\ai")
import h3_engine as he
if os.environ.get("H3_ATTN_FP4") == "1":
    import h3_attn_fp4
    h3_attn_fp4.install(he)
import h3image_engine as base

# 魔改1: fused rms_norm (1 kernel) replaces manual pow/mean/rsqrt (5 kernels), FP4 path only
def _fused_rms(x, w, eps=1e-5):
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(w.dtype)
if os.environ.get('H3_FUSED_RMS') == '1':
    he.rms = _fused_rms   # [11-24] 默认恢复 11-20 manual rms 舍入口径; fused 版需显式开启

FP4_DIR = r"D:\ai\models\MiniMax-H3-Image-FP4\transformer"
CACHE_FILE = os.path.join(FP4_DIR, "adaln_cache.safetensors")
DEV = "cuda"; DTYPE = torch.bfloat16
fp4 = torch.float4_e2m1fn_x2; fp8 = torch.float8_e4m3fn
HID = base.HID; HEADS = base.HEADS; HEAD_DIM = base.HEAD_DIM; MODE_NUM = base.MODE_NUM
LV = torch.tensor([0., 0.5, 1, 1.5, 2, 3, 4, 6], device=DEV)
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)
TARGETS = ('mlp.fc1', 'mlp.fc2', 'attn.qkv_proj', 'attn.out_proj')
_log = r"D:\ai\h3image_fp4.log"
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    try:
        with open(_log, "a", encoding="utf-8") as f: f.write(s + "\n")
    except Exception: pass

def ceil_div(a, b): return (a + b - 1) // b
def to_blocked(m):
    rows, cols = m.shape; nr = ceil_div(rows, 128); nc = ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    if (rows, cols) != (pr, pc):
        p = torch.zeros((pr, pc), device=m.device, dtype=m.dtype); p[:rows, :cols] = m
    else: p = m
    return p.view(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()

def fp4_gemm(a4, b4, sa_sw, ga, sb_sw, gb, out_dtype=torch.bfloat16):
    return torch._scaled_mm_v2(a4, b4,
        scale_a=[sa_sw, (1.0 / ga)], recipe_a=[ST.BlockWise1x16, ST.TensorWise], swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        scale_b=[sb_sw, (1.0 / gb)], recipe_b=[ST.BlockWise1x16, ST.TensorWise], swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        bias=None, out_dtype=out_dtype)

def quant_act(x2):
    # [11-24 恢复 11-20 fp32 口径] bf16 全程版有 0.16% 码翻转, 混沌放大后纹理变软; 小 M 内存可控
    M, K = x2.shape; xf = x2.to(DEV).float()
    g = (448 * 6) / xf.abs().max().clamp(min=1e-12)
    b = xf.reshape(M, K // 16, 16); am = b.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = ((am / 6) * g).to(fp8); se = s.float().clamp(min=1e-12)
    n = (b * g / se).clamp(-6, 6)
    idx = torch.bucketize(n.abs(), MID).to(torch.uint8)
    code = ((n < 0).to(torch.uint8) << 3) | idx; code = code.reshape(M, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, to_blocked(s.squeeze(-1).reshape(M, K // 16)), g.reshape(1)

def is_target(name):
    return name.endswith('.weight') and any(s in name for s in TARGETS)

def _pin(t):
    # No pin needed: 32GB RAM holds the 11.5GB FP4 weights pageable; H2D from RAM is sufficient.
    return t

class FP4Reader:
    """Load FP4 shards: targets -> pinned (packed, swizzled scale, global); norms/glue -> BF16/FP32."""
    def __init__(self):
        self.layers = {j: {} for j in range(50)}
        self.sm = {}
        self.n = 50
        shards = sorted(glob.glob(os.path.join(FP4_DIR, "model-*.safetensors")))
        for fp in shards:
            sd = load_file(fp)
            names = list(sd.keys())
            for name in names:
                if name.endswith('.fp4'):
                    bname = name[:-4]
                    packed = sd[name]
                    scale = sd[bname + '.scale']
                    g = sd[bname + '.global']
                    sw = to_blocked(scale)                      # swizzle in RAM
                    N, K = packed.shape[0], packed.shape[1] * 2
                    rec = {'packed': _pin(packed.contiguous()), 'sw': _pin(sw.contiguous()),
                           'g': g.float(), 'N': N, 'K': K}
                    if bname.startswith('blocks.'):
                        j = int(bname.split('.')[1]); rel = bname.split('.', 2)[2]
                        self.layers[j][rel] = rec
                    else:
                        self.sm[bname] = rec
                elif name.endswith('.scale') or name.endswith('.global'):
                    continue
                else:
                    if 'adaln_proj' in name or name.startswith('time_embedder') or 'video_out' in name:
                        continue                                  # cached / dropped
                    ten = sd[name]
                    if name.startswith('blocks.'):
                        j = int(name.split('.')[1]); rel = name.split('.', 2)[2]
                        self.layers[j][rel] = _pin(ten.contiguous())
                    else:
                        self.sm[name] = ten
            del sd
        log("[fp4reader] layers=%d sm=%d" % (self.n, len(self.sm)))
    def small_gpu(self):
        out = {}
        for n, v in self.sm.items():
            if isinstance(v, dict):
                out[n] = {'b4': v['packed'].to(DEV).view(fp4).t(), 'sw': v['sw'].to(DEV),
                          'g': v['g'].to(DEV), 'N': v['N']}
            else:
                out[n] = v.to(DEV)
        return out

def make_layer_fn(reader, resident):
    from concurrent.futures import ThreadPoolExecutor
    gpu = {}
    order = []
    ready = {}
    ex = ThreadPoolExecutor(max_workers=1)
    def build(i):
        st = {}
        for rel, v in reader.layers[i].items():
            if isinstance(v, dict):
                st[rel] = {'b4': v['packed'].to(DEV, non_blocking=True).view(fp4).t(),
                           'sw': v['sw'].to(DEV, non_blocking=True), 'g': v['g'].to(DEV), 'N': v['N']}
            else:
                st[rel] = v.to(DEV, non_blocking=True)
        return st
    def layer(i):
        # CPU background thread prefetches layer i+1 (H2D+prep) while GPU computes layer i
        if i + 1 < reader.n and (i + 1) not in ready and (i + 1) not in gpu:
            ready[i + 1] = ex.submit(build, i + 1)
        if i in gpu: return gpu[i]
        if i in ready:
            st = ready.pop(i).result()
        else:
            st = build(i)
        gpu[i] = st; order.append(i)
        while len(order) > resident:
            old = order.pop(0)
            if old != i: gpu.pop(old, None)
        return st
    return layer

def _quant_plain_k2(x2):
    """[11-24 主人令] 激活套层 k1+k2：A->q1; R=A-deq(q1); R->q2。fp32 数学（与 11-20 口径同）。"""
    M, K = x2.shape
    xf = x2.to(DEV).float()
    g = (448 * 6) / xf.abs().max().clamp(min=1e-12)
    b = xf.reshape(M, K // 16, 16)
    am = b.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = ((am / 6) * g).to(fp8)
    se = s.float().clamp(min=1e-12)
    n = (b * g / se).clamp(-6, 6)
    idx = torch.bucketize(n.abs(), MID).to(torch.uint8)
    code = ((n < 0).to(torch.uint8) << 3) | idx
    code = code.reshape(M, K)
    p1 = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    s1 = s.squeeze(-1)
    lv = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    d1 = (lv * se.repeat_interleave(16, 1) / g).reshape(M, K)
    r = xf - d1
    g2 = (448 * 6) / r.abs().max().clamp(min=1e-12)
    b2 = r.reshape(M, K // 16, 16)
    am2 = b2.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s2 = ((am2 / 6) * g2).to(fp8)
    se2 = s2.float().clamp(min=1e-12)
    n2 = (b2 * g2 / se2).clamp(-6, 6)
    idx2 = torch.bucketize(n2.abs(), MID).to(torch.uint8)
    code2 = ((n2 < 0).to(torch.uint8) << 3) | idx2
    code2 = code2.reshape(M, K)
    p2 = (code2[:, 0::2] | (code2[:, 1::2] << 4)).contiguous()
    return p1, to_blocked(s1), g, p2, to_blocked(s2.squeeze(-1)), g2


def fp4_lin(x, w, mod=None):
    B, L, K = x.shape
    x2 = x.reshape(B * L, K)
    M = x2.shape[0]
    CH = 4096   # bound quant temporaries (int64 bucketize / fp32 blocks) per chunk at 4K
    if M <= CH:
        if os.environ.get('H3_ACT_K2') == '1':
            p1, s1, g1, p2, s2, g2 = _quant_plain_k2(x2)
            out = fp4_gemm(p1.view(fp4), w['b4'], s1, g1, w['sw'], w['g'])
            out = out + fp4_gemm(p2.view(fp4), w['b4'], s2, g2, w['sw'], w['g'])
        else:
            xp, xs_sw, xg = quant_act(x2)
            out = fp4_gemm(xp.view(fp4), w['b4'], xs_sw, xg, w['sw'], w['g'])
    else:
        out = torch.empty(M, w['N'], device=x2.device, dtype=torch.bfloat16)
        for s in range(0, M, CH):
            mc = (mod[0], mod[1], mod[2][s:s + CH]) if mod else None
            xp, xs_sw, xg = _quant(x2[s:s + CH], mc)
            out[s:s + min(CH, M - s)] = fp4_gemm(xp.view(fp4), w['b4'], xs_sw, xg, w['sw'], w['g'])
    return out.reshape(B, L, w['N']).to(DTYPE)

# ---- self-developed fused NVFP4 activation-quant kernels (NVRTC, official, no MSVC) ----
_FUSED = None
def _init_fused():
    global _FUSED
    if _FUSED is not None: return _FUSED
    try:
        from cuda.bindings import nvrtc, driver
        cu = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fused_quant.cu')).read()
        prog = nvrtc.nvrtcCreateProgram(cu.encode(), b'fused_quant.cu', 0, [], [])[1]
        sp = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        inc = os.path.join(sp, 'nvidia', 'cuda_runtime', 'include')
        if not os.path.exists(inc):
            import nvidia.cuda_runtime  # noqa  (namespace pkg: __file__ is None, use __path__)
            inc = os.path.join(list(nvidia.cuda_runtime.__path__)[0], 'include')
        opts = [b'--gpu-architecture=sm_120', b'--std=c++17', b'-I' + inc.encode()]
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(r[0]) != 0: return None
        csz = nvrtc.nvrtcGetCUBINSize(prog)[1]; cb = bytearray(csz); nvrtc.nvrtcGetCUBIN(prog, cb)
        mod = driver.cuModuleLoadData(bytes(cb))[1]
        fns = {n: driver.cuModuleGetFunction(mod, n.encode())[1] for n in ('mod_am', 'quant_am', 'fused_quant')}
        _FUSED = (driver, fns)
        return _FUSED
    except Exception:
        return None

def _launch(driver, fn, nb, argspec):
    import ctypes
    objs = []
    for t, v in argspec:
        objs.append(ctypes.c_void_p(v) if t == 'p' else ctypes.c_float(v) if t == 'f'
                    else ctypes.c_longlong(v) if t == 'l' else ctypes.c_int(v))
    params = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
    driver.cuLaunchKernel(fn, (nb + 255) // 256, 1, 1, 256, 1, 1, 0, torch.cuda.current_stream().cuda_stream, params, 0)

def _quant(x2, mod=None):
    f = _init_fused()
    if f is None:
        if mod is not None:
            sc, sh, idx = mod
            x2 = x2 * (1.0 + sc[idx]) + sh[idx]
        return quant_act(x2)
    driver, fns = f
    M, K = x2.shape
    nb = M * (K // 16)
    packed = torch.empty(M, K // 2, dtype=torch.uint8, device=DEV)
    scale = torch.empty(nb, dtype=torch.uint8, device=DEV)
    if mod is not None:
        sc, sh, idx = mod
        amout = torch.empty(nb, dtype=torch.float32, device=DEV)
        _launch(driver, fns['mod_am'], nb, [('p', x2.data_ptr()), ('p', sc.data_ptr()), ('p', sh.data_ptr()),
                                           ('p', idx.data_ptr()), ('p', amout.data_ptr()), ('l', nb), ('i', K)])
        g = (448 * 6) / amout.max().clamp(min=1e-12)   # same am source as kernel => no fp8 saturation
        gf = float(g.item())
        _launch(driver, fns['quant_am'], nb, [('p', x2.data_ptr()), ('p', sc.data_ptr()), ('p', sh.data_ptr()),
                                             ('p', idx.data_ptr()), ('p', amout.data_ptr()), ('f', gf),
                                             ('p', packed.data_ptr()), ('p', scale.data_ptr()), ('l', nb), ('i', K)])
    else:
        g = (448 * 6) / x2.abs().amax().float().clamp(min=1e-12)
        gf = float(g.item())
        _launch(driver, fns['fused_quant'], nb, [('p', x2.data_ptr()), ('f', gf), ('p', packed.data_ptr()),
                                                ('p', scale.data_ptr()), ('l', nb), ('i', K),
                                                ('p', 0), ('p', 0), ('p', 0)])
    return packed, to_blocked(scale.view(torch.float8_e4m3fn).reshape(M, K // 16)), g

def fp4_refiner(t, sm, lora=None):
    for i in (0, 1):
        p = "token_refiner.blocks.%d." % i
        side = dict((k[1], v) for k, v in lora.items() if k[0] == i) if lora else None
        x = he.rms(t, sm[p + "norm1.weight"])
        qkv = he._add_lora(he._qkv_fix(fp4_lin(x, sm[p + "attn.qkv_proj.weight"])), x, side, "attn.qkv_proj")
        q = qkv[:, :, :7168].view(t.shape[0], -1, HEADS, HEAD_DIM)
        k = qkv[:, :, 7168:14336].view(t.shape[0], -1, HEADS, HEAD_DIM)
        v = qkv[:, :, 14336:].view(t.shape[0], -1, HEADS, HEAD_DIM)
        q = he.rms(q, sm[p + "attn.q_norm.weight"]); k = he.rms(k, sm[p + "attn.k_norm.weight"])
        o = he._sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        o = o.transpose(1, 2).reshape(t.shape[0], -1, HEADS * HEAD_DIM)
        o = he._add_lora(fp4_lin(o, sm[p + "attn.out_proj.weight"]), o, side, "attn.out_proj")
        t = t + o
        x = he.rms(t, sm[p + "norm2.weight"])
        a, g = he._add_lora(fp4_lin(x, sm[p + "mlp.fc1.weight"]), x, side, "mlp.fc1").chunk(2, dim=-1)
        hh = he._swiglu(a, g)
        t = t + he._add_lora(fp4_lin(hh, sm[p + "mlp.fc2.weight"]), hh, side, "mlp.fc2")
    return he.rms(t, sm["token_refiner.final_norm.weight"])

def fp4_block(x, st, mods, idx, cos, sin, lora=None):
    sh1, sc1, g1, sh2, sc2, g2 = mods
    sh1 = sh1.index_select(0, idx).unsqueeze(0); sc1 = sc1.index_select(0, idx).unsqueeze(0)
    g1 = g1.index_select(0, idx).unsqueeze(0)
    sh2 = sh2.index_select(0, idx).unsqueeze(0); sc2 = sc2.index_select(0, idx).unsqueeze(0)
    g2 = g2.index_select(0, idx).unsqueeze(0)
    h = he.rms(x, st["norm1.weight"]); h = h * (1.0 + sc1) + sh1
    B, Lt, HID = x.shape
    CHK = int(os.environ.get("H3_CHUNK", "4096"))
    chunked = Lt > CHK
    _lvl = os.environ.get("H3_ATTN_LEVEL", "1")
    fp4attn = os.environ.get("H3_ATTN_FP4") == "1" and _lvl in ("1", "2")
    if chunked and fp4attn:
        import h3_attn_fp4 as _A
        o = _A.attn_chunked(h, st["attn.qkv_proj.weight"], st["attn.q_norm.weight"],
                            st["attn.k_norm.weight"], cos, sin, lora, CHK, level=int(_lvl))
    else:
        qkv = he._add_lora(fp4_lin(h, st["attn.qkv_proj.weight"]), h, lora, "attn.qkv_proj")
        o = None
        if fp4attn:
            import h3_attn_fp4 as _A
            o = _A.attn_from_qkv(qkv, st["attn.q_norm.weight"], st["attn.k_norm.weight"], cos, sin)
        if o is None:
            qkv = he._qkv_fix(qkv)
            q = qkv[:, :, :7168].view(x.shape[0], -1, HEADS, HEAD_DIM)
            k = qkv[:, :, 7168:14336].view(x.shape[0], -1, HEADS, HEAD_DIM)
            v = qkv[:, :, 14336:].view(x.shape[0], -1, HEADS, HEAD_DIM)
            q = he.rms(q, st["attn.q_norm.weight"]); k = he.rms(k, st["attn.k_norm.weight"])
            q = he.rope_apply(q, cos, sin); k = he.rope_apply(k, cos, sin)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            o = he._sdpa(q, k, v)
    o = o.transpose(1, 2)          # [B, Lt, H, D]
    if chunked:
        for s in range(0, Lt, CHK):
            c = min(CHK, Lt - s)
            oc = o[:, s:s + c].reshape(B, c, HEADS * HEAD_DIM)
            oc = he._add_lora(fp4_lin(oc, st["attn.out_proj.weight"]), oc, lora, "attn.out_proj")
            x[:, s:s + c] = x[:, s:s + c] + g1[:, s:s + c] * oc
            del oc
    else:
        oc = o.reshape(B, Lt, HEADS * HEAD_DIM)
        oc = he._add_lora(fp4_lin(oc, st["attn.out_proj.weight"]), oc, lora, "attn.out_proj")
        x = x + g1 * oc
    h = he.rms(x, st["norm2.weight"]); h = h * (1.0 + sc2) + sh2
    if chunked:
        for s in range(0, Lt, CHK):
            c = min(CHK, Lt - s)
            a, gg = he._add_lora(fp4_lin(h[:, s:s + c], st["mlp.fc1.weight"]),
                                 h[:, s:s + c], lora, "mlp.fc1").chunk(2, dim=-1)
            hh = he._swiglu(a, gg)
            hc = he._add_lora(fp4_lin(hh, st["mlp.fc2.weight"]), hh, lora, "mlp.fc2")
            x[:, s:s + c] = x[:, s:s + c] + g2[:, s:s + c] * hc
            del a, gg, hh, hc
    else:
        a, gg = he._add_lora(fp4_lin(h, st["mlp.fc1.weight"]), h, lora, "mlp.fc1").chunk(2, dim=-1)
        hh = he._swiglu(a, gg)
        hc = he._add_lora(fp4_lin(hh, st["mlp.fc2.weight"]), hh, lora, "mlp.fc2")
        x = x + g2 * hc
    return x


def fp4_dit_forward(reader, ximg, cond, step_i, ti, tags, pos, image_idx, text_idx,
                    sm, layer_fn, pdd_w, pdd_b, cache):
    t0 = time.perf_counter()
    ve = he.lin(ximg.to(DEV, sm["video_patch_proj.weight"].dtype),
                sm["video_patch_proj.weight"], sm["video_patch_proj.bias"])
    te = he.lin(cond.to(DEV, DTYPE), sm["condition_proj.weight"], sm["condition_proj.bias"])
    te = fp4_refiner(te, sm, he.refiner_lora())
    L = pos.shape[0]
    h = te.new_zeros((te.shape[0], L, te.shape[-1]))
    h = h.index_copy(1, text_idx.to(DEV), te)
    h = h.index_copy(1, image_idx.to(DEV), ve.to(te.dtype))
    idx = (ti.to(DEV) * MODE_NUM + tags.to(DEV)).long()
    cos, sin = he.build_rope(pos.to(DEV), sm["rope.inv_freq"].float())
    for i in range(reader.n):
        st = layer_fn(i)
        mods = cache["s%d_l%d" % (step_i, i)].to(DEV).unbind(0)   # 6 x [3,HID]
        h = fp4_block(h, st, mods, idx, cos, sin, he.lora_side(i))
        if os.environ.get("H3_EMPTY_EVERY", "1") == "1":
            torch.cuda.empty_cache()
        if i % 10 == 0 or i == reader.n - 1:
            _card = ""
            try:
                import pynvml as _p
                _p.nvmlInit()
                _mi = _p.nvmlDeviceGetMemoryInfo(_p.nvmlDeviceGetHandleByIndex(0))
                _card = " card %.2f/%.2fGB" % (_mi.used / 2**30, _mi.total / 2**30)
            except Exception:
                pass
            log("[d] layer %d/%d %.1fs alloc %.2fGB reserved %.2fGB peak %.2fGB%s" % (
                i, reader.n, time.perf_counter() - t0,
                torch.cuda.memory_allocated() / 1024 ** 3,
                torch.cuda.memory_reserved() / 1024 ** 3,
                torch.cuda.max_memory_allocated() / 1024 ** 3, _card))
    hh = he.rms(h, sm["final_layer.norm.weight"])
    ao = cache["s%d_final" % step_i].to(DEV)
    sh, sc = ao[0], ao[1]
    _ti = ti.to(DEV)
    hh = hh * (1.0 + sc.index_select(0, _ti).unsqueeze(0)) + sh.index_select(0, _ti).unsqueeze(0)
    hh = hh.to(torch.float32)
    out = he.lin(hh, pdd_w, pdd_b).index_select(1, image_idx.to(DEV))
    return out

def run(args):
    from safetensors.torch import load_file as lf
    t_all = time.perf_counter()
    H, W = base.resolve_canvas(args)
    lh, lw = H // base.VAE_RATIO, W // base.VAE_RATIO
    cond = base.load_cond(args.cond, args.cond_mode)
    n_text = cond.shape[1]
    pos, tags, image_idx, text_idx, n_img = base.build_layout(he, n_text, lh, lw)
    log("[p] canvas %dx%d -> %d img + %d text = %d rows" % (H, W, n_img, n_text, n_img + n_text))
    pdd = lf(base.PDD_FILE)
    pdd_w = pdd["video_out.weight"].to(DEV).float(); pdd_b = pdd["video_out.bias"].to(DEV).float()
    cfg = json.load(open(base.PDD_CFG, encoding="utf-8"))
    sig0 = torch.tensor(cfg["sigmas"], dtype=torch.float64)
    ts = torch.tensor([1.0 - float(s) for s in sig0[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - ts, torch.zeros(1)]).to(torch.float64)
    cache = lf(CACHE_FILE)
    log("[m] adaln cache keys=%d" % len(cache))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    lat0 = torch.randn((1, base.LATENT_CH, 1, lh, lw), generator=g, dtype=torch.float32)
    x = lat0.reshape(1, base.LATENT_CH, 1, 1, lh // 2, 2, lw // 2, 2) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape(1, n_img, base.LATENT_CH * base.PATCH[1] * base.PATCH[2]).contiguous().to(DEV)
    reader = FP4Reader()
    sm = reader.small_gpu()
    # Auto-residency (scheme A): activations scale with L; cap resident so weights+act fit VRAM.
    L = n_img + n_text
    act_est_gb = 3.6e-4 * L + 0.1          # calibrated: 512~0.3GB, 4K~6.1GB (bf16 inter+quant+attn)
    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
    resident_eff = min(args.resident, max(8, int((free_gb - act_est_gb - 1.5) / 0.208)))
    log("[m] L=%d act_est=%.2fGB free=%.2fGB -> resident_eff=%d (cap %d)" % (
        L, act_est_gb, free_gb, resident_eff, args.resident))
    layer_fn = make_layer_fn(reader, resident_eff)
    log("[m] FP4 weights loaded, resident=%d VRAM %.2fGB" % (args.resident, torch.cuda.memory_allocated() / 1024 ** 3))
    for i in range(int(cfg["num_inference_steps"])):
        t_i = float(ts[i])
        row_t = torch.full((n_img + n_text,), t_i, dtype=torch.float32)
        uniq_t, ti = torch.unique(row_t, sorted=True, return_inverse=True)
        v = fp4_dit_forward(reader, x, cond, i, ti, tags, pos, image_idx, text_idx,
                            sm, layer_fn, pdd_w[i], pdd_b[i], cache)
        ds = float(sigmas[i] - sigmas[i + 1])
        x = x + ds * v.float()
        log("[s] step %d/6 t=%.6f |v|std=%.4f x_std=%.4f %.0fs" % (
            i + 1, t_i, float(v.float().std()), float(x.std()), time.perf_counter() - t_all))
        if not bool(torch.isfinite(x).all()):
            log("[!] non-finite, abort"); return False
    rows = x[0].float().cpu()
    y = rows.reshape(-1, lh // 2, lw // 2, base.LATENT_CH, 2, 2) \
            .permute(0, 3, 1, 4, 2, 5).reshape(1, base.LATENT_CH, lh, lw).contiguous()
    vcfg = json.load(open(base.VAE_CFG, encoding="utf-8"))
    lm = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1); ls = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1)
    y = y * ls + lm
    lat_path = args.save_latent or os.path.join(r"D:\ai\output", "h3image_fp4_%dx%d_latent.pt" % (H, W))
    torch.save(y, lat_path)
    log("[d] latent saved %s" % lat_path)
    del layer_fn, reader, sm, x, v
    import gc; gc.collect(); torch.cuda.empty_cache()
    sd, vc = base.load_vae()
    with torch.no_grad():
        image = base.vae_decode(y.to(DEV), sd, vc)
    out = args.out or os.path.join(r"D:\ai\output\images", "h3image_fp4_%dx%d_%d.png" % (H, W, args.seed))
    base.to_png(image, out)
    log("[done] FP4 full %0.fs -> %s" % (time.perf_counter() - t_all, out))
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--width", type=int, default=2048); ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--min_area", type=int, default=base.MIN_GENERATION_AREA)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond", default=base.DEFAULT_COND)
    ap.add_argument("--cond_mode", default="official")
    ap.add_argument("--resident", type=int, default=32)
    ap.add_argument("--out", default=None); ap.add_argument("--save_latent", default=None)
    a = ap.parse_args()
    if a.size is not None: a.width = a.height = a.size
    log("=" * 60); log("[fp4] start %dx%d min_area=%d seed=%d resident=%d" % (a.width, a.height, a.min_area, a.seed, a.resident))
    ok = run(a) if a.run else False
    log("[fp4] RESULT: %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())