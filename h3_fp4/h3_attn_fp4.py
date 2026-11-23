# -*- coding: utf-8 -*-
"""H3 引擎的 FP4 flash-attention 接入层（monkey-patch he._sdpa）。
开关：H3_ATTN_FP4=1；H3_ATTN_LEVEL=1|2（默认1）；H3_ATTN_DEBUG=1 打印每层 L/耗时（前3层）。
"""
import math, os, sys, time
sys.path.insert(0, r"D:\ai\output\attn_kernel")
sys.path.insert(0, r"D:\ai\ve_triton")
os.environ.setdefault("CC", r"D:\ai\ve_triton\triton\runtime\tcc\tcc.exe")
import torch

_LEVEL = int(os.environ.get("H3_ATTN_LEVEL", "1"))
_DEBUG = os.environ.get("H3_ATTN_DEBUG") == "1"
_calls = [0]
DEV = "cuda"

import flash_fp4 as _k
try:
    import pynvml as _nv
    _nv.nvmlInit()
    _H = _nv.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _nv = None


def fp4_sdpa(q, k, v):
    t0 = time.perf_counter()
    outs = [_k.flash_attn_masked(q[b], k[b], v[b], _LEVEL) for b in range(q.shape[0])]
    o = outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs)
    _calls[0] += 1
    if _DEBUG and (_calls[0] <= 3 or _calls[0] % 10 == 0):
        try:
            mi = _nv.nvmlDeviceGetMemoryInfo(_H)
            vram = "card %.2f/%.2fGB" % (mi.used / 2**30, mi.total / 2**30)
        except Exception:
            vram = "card n/a"
        print("[attn-fp4] #%d k%d L=%d %.1fms  torch alloc %.2fGB reserved %.2fGB peak %.2fGB  %s"
              % (_calls[0], _LEVEL, q.shape[2], (time.perf_counter() - t0) * 1e3,
                 torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30,
                 torch.cuda.max_memory_allocated() / 2**30, vram), flush=True)
    return o


def attn_from_qkv(qkv, wq, wk, cos, sin):
    """直接量化路径：qkv[1,L,21504] -> rms+rope+FP4 融合 -> flash 注意力，返回 [1,H,L,D] bf16。
    仅 level=1；level=2 返回 None 由调用方走旧路径。"""
    if _LEVEL != 1:
        return None
    D = 128
    Hn = qkv.shape[2] // (3 * D)
    L = qkv.shape[1]
    Lp = ((L + 63) // 64) * 64
    b = _k._fast_bufs(Hn, Lp, D, 8)
    o = b['o']
    cosb = cos.to(torch.bfloat16).contiguous()
    sinb = sin.to(torch.bfloat16).contiguous()
    from quant_fused import qkv_rope_quant
    NM = Lp // 64
    for h0, c in qkv_rope_quant(qkv[0], wq, wk, cosb, sinb, Hn, L, Lp, 8, b):
        _k.flash_fp4[(NM, c)](
            b['qp'][:c * Lp].view(torch.uint8), b['qs'][:c * Lp],
            b['qp'][:c * Lp].view(torch.uint8), b['qs'][:c * Lp],
            b['kp'][:c * Lp].view(torch.uint8), b['ks'][:c * Lp],
            b['vp'][:c * Lp].view(torch.uint8), b['vs'][:c],
            b['kp'][:c * Lp].view(torch.uint8), b['ks'][:c * Lp],
            b['vp'][:c * Lp].view(torch.uint8), b['vs'][:c],
            o[h0:h0 + c], Lp, D, 64, 64, NM, False, 1.0 / math.sqrt(D), L, True,
            num_warps=4, num_stages=2)
    return o[:, :L].unsqueeze(0)


def attn_chunked(h, w_qkv, wq, wk, cos, sin, lora, CHK=4096, level=1):
    """qkv GEMM 按 4096 行分块；rms/rope 走 he 已验证算子，量化走 stack_parts_fused（level1/2）。
    返回 [1,H,L,D] bf16（O 为常驻缓冲视图）。"""
    import h3_engine as he
    import h3image_engine_fp4 as F4
    from quant_fused import stack_parts_fused
    D = 128
    Hn = 56
    Lt = h.shape[1]
    Lp = ((Lt + 63) // 64) * 64
    u8, fp8, bf = torch.uint8, torch.float8_e4m3fn, torch.bfloat16
    QP = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); QS = torch.zeros(Hn, Lp, D // 16, device=DEV, dtype=fp8)
    KP = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); KS = torch.zeros(Hn, Lp, D // 16, device=DEV, dtype=fp8)
    VP = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); VS = torch.zeros(Hn, D, Lp // 16, device=DEV, dtype=fp8)
    QP2 = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); QS2 = torch.zeros(Hn, Lp, D // 16, device=DEV, dtype=fp8)
    KP2 = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); KS2 = torch.zeros(Hn, Lp, D // 16, device=DEV, dtype=fp8)
    VP2 = torch.zeros(Hn, Lp, D // 2, device=DEV, dtype=u8); VS2 = torch.zeros(Hn, D, Lp // 16, device=DEV, dtype=fp8)
    O = torch.empty(Hn, Lp, D, device=DEV, dtype=bf)
    for s in range(0, Lt, CHK):
        c = min(CHK, Lt - s)
        Lpc = ((c + 63) // 64) * 64
        y = F4.fp4_lin(h[:, s:s + c], w_qkv)
        y = he._add_lora(y, h[:, s:s + c], lora, "attn.qkv_proj")
        y = he._qkv_fix(y)[0]
        q = y[:, :7168].view(c, Hn, D); k = y[:, 7168:14336].view(c, Hn, D); v = y[:, 14336:].view(c, Hn, D)
        q = he.rms(q, wq); k = he.rms(k, wk)
        q = he.rope_apply(q.unsqueeze(0), cos[s:s + c], sin[s:s + c])[0]
        k = he.rope_apply(k.unsqueeze(0), cos[s:s + c], sin[s:s + c])[0]
        q = q.transpose(0, 1).contiguous(); k = k.transpose(0, 1).contiguous(); v = v.transpose(0, 1).contiguous()
        if Lpc != c:
            qp0 = torch.zeros(Hn, Lpc, D, device=DEV, dtype=bf); qp0[:, :c] = q; q = qp0
            kp0 = torch.zeros(Hn, Lpc, D, device=DEV, dtype=bf); kp0[:, :c] = k; k = kp0
            vp0 = torch.zeros(Hn, Lpc, D, device=DEV, dtype=bf); vp0[:, :c] = v; v = vp0
        a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2 = stack_parts_fused(q, k, v, level)
        QP[:, s:s + Lpc].copy_(a1); QS[:, s:s + Lpc].copy_(b1)
        KP[:, s:s + Lpc].copy_(c1); KS[:, s:s + Lpc].copy_(d1)
        VP[:, s:s + Lpc].copy_(e1); VS[:, :, s // 16:(s + Lpc) // 16].copy_(f1)
        QP2[:, s:s + Lpc].copy_(a2); QS2[:, s:s + Lpc].copy_(b2)
        KP2[:, s:s + Lpc].copy_(c2); KS2[:, s:s + Lpc].copy_(d2)
        VP2[:, s:s + Lpc].copy_(e2); VS2[:, :, s // 16:(s + Lpc) // 16].copy_(f2)
        del y, q, k, v, a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2
    NM = Lp // 64
    for h0 in range(0, Hn, 8):
        c = min(8, Hn - h0)
        _k.flash_fp4[(NM, c)](
            QP[h0:h0 + c].reshape(c * Lp, D // 2), QS[h0:h0 + c].reshape(c * Lp, D // 16),
            QP2[h0:h0 + c].reshape(c * Lp, D // 2), QS2[h0:h0 + c].reshape(c * Lp, D // 16),
            KP[h0:h0 + c].reshape(c * Lp, D // 2), KS[h0:h0 + c].reshape(c * Lp, D // 16),
            VP[h0:h0 + c].reshape(c * Lp, D // 2), VS[h0:h0 + c],
            KP2[h0:h0 + c].reshape(c * Lp, D // 2), KS2[h0:h0 + c].reshape(c * Lp, D // 16),
            VP2[h0:h0 + c].reshape(c * Lp, D // 2), VS2[h0:h0 + c],
            O[h0:h0 + c], Lp, D, 64, 64, NM, level == 2, 1.0 / math.sqrt(D), Lt, True,
            num_warps=4, num_stages=2)
    return O[:, :Lt].unsqueeze(0)


def install(he):
    he._sdpa = fp4_sdpa
    print("[attn-fp4] installed level=%d" % _LEVEL, flush=True)
