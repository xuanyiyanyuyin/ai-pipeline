# -*- coding: utf-8 -*-
"""FP4 flash-attention（NVFP4: e2m1 nibble + per-16 e4m3 block scale），Triton，sm_120。
LEVEL=1: 单 pass；LEVEL=2: 残差套层，S/O 均为 (1+2) 全 4 项。

★sm_120 已知缺陷（本文件绕开）：tl.dot_scaled 的 e2m1 操作数若来自寄存器计算（如 flash 里的
  P=exp(S)），打包 nibble 会被 lowering 误读（实测 relL2 1.40 vs 0.012）。Q/K/V 从内存加载不受
  影响。修复：对打包字节与尺度张量各做一次 identity tl.gather（内部经共享内存做布局往返），
  dot_scaled 即恢复正常（0.0125）。全局 scratch+barrier 方案不可用（与异步搬运竞态）。

布局（由 semantic.dot_scaled 约定）:
  Q  lhs [L,128] 沿 K 打包 → [L,64] u8, scales [L,8] fp8
  K  rhs k_pack=True → [64,L] u8（量化后转置）, scales [L,8]
  P  核内量化 [BM,BN] → [BM,BN/2] u8, scales [BM,BN/16]
  V  rhs_k_pack=False → [L,64] u8（沿 head_dim 配对）, scales [128,L/16]（按 key 16 分组）
"""
import os, sys, math, time
sys.path.insert(0, r'D:\ai\ve_triton')
import torch
import torch.nn.functional as Fn
import triton
import triton.language as tl

DEV = 'cuda'
FP8 = torch.float8_e4m3fn
LV = torch.tensor([0., .5, 1, 1.5, 2, 3, 4, 6], device=DEV)
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)


# ---------------- host side quant ----------------
def _codes(x):
    M, K = x.shape
    b = x.reshape(M, K // 16, 16)
    am = b.abs().amax(-1).clamp(min=1e-12)
    s = (am / 6.0).to(FP8)
    n = (b / s.float().clamp(min=1e-12).unsqueeze(-1)).clamp(-6, 6).to(torch.bfloat16)
    idx = torch.bucketize(n.abs().float(), MID).to(torch.uint8)
    code = (idx | ((n < 0).to(torch.uint8) << 3)).reshape(M, K)
    return code, s


def _decode(code, s):
    lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    return lvl * s.float().repeat_interleave(16, dim=1)


def _pack(code):
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()


def quant_q(x):
    code, s = _codes(x.float())
    return _pack(code), s.contiguous(), _decode(code, s)


def quant_k(x):
    p, s, d = quant_q(x)
    return p.t().contiguous(), s, d


def quant_v(x):
    L, D = x.shape
    xf = x.float()
    b = xf.reshape(L // 16, 16, D)
    am = b.abs().amax(1)
    s = (am / 6.0).to(FP8)
    n = (b / s.float().clamp(min=1e-12).unsqueeze(1)).clamp(-6, 6).to(torch.bfloat16)
    idx = torch.bucketize(n.abs().float(), MID).to(torch.uint8)
    code = (idx | ((n < 0).to(torch.uint8) << 3)).reshape(L, D)
    lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    dq = lvl * s.float().repeat_interleave(16, dim=0)
    return _pack(code), s.t().contiguous(), dq


def stack_parts(q, k, v, level):
    qp1, qs1, qd1 = quant_q(q)
    kp1, ks1, kd1 = quant_k(k)
    vp1, vs1, vd1 = quant_v(v)
    if level == 1:
        return qp1, qs1, kp1, ks1, vp1, vs1, qp1, qs1, kp1, ks1, vp1, vs1
    qp2, qs2, _ = quant_q(q.float() - qd1)
    kp2, ks2, _ = quant_k(k.float() - kd1)
    vp2, vs2, _ = quant_v(v.float() - vd1)
    return qp1, qs1, kp1, ks1, vp1, vs1, qp2, qs2, kp2, ks2, vp2, vs2


# ---------------- batched host quant over H heads（单次核启动） ----------------
def _codes_flat(x):
    """x [N,K] fp32 -> code u8 [N,K], scales fp8 [N,K/16]（比较式分桶，不用 bucketize）"""
    N, K = x.shape
    b = x.reshape(N, K // 16, 16)
    am = b.abs().amax(-1).clamp(min=1e-12)
    s = (am / 6.0).to(FP8)
    n = (b / s.float().clamp(min=1e-12).unsqueeze(-1)).clamp(-6, 6).to(torch.bfloat16)
    a = n.abs()
    idx = ((a > 0.25).to(torch.uint8) + (a > 0.75).to(torch.uint8) + (a > 1.25).to(torch.uint8) +
           (a > 1.75).to(torch.uint8) + (a > 2.5).to(torch.uint8) + (a > 3.5).to(torch.uint8) +
           (a > 5.0).to(torch.uint8))
    code = (idx | ((n < 0).to(torch.uint8) << 3)).reshape(N, K)
    return code, s


def _decode_flat(code, s):
    lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    return lvl * s.float().repeat_interleave(16, dim=1)


def quant_q_batch(x, need_dq=False):
    """q [H,L,D] bf16 -> packed [H,L,D/2], scales [H,L,D/16], dq [H*L,D] or None"""
    H, L, D = x.shape
    code, s = _codes_flat(x.reshape(H * L, D).float())
    dq = _decode_flat(code, s) if need_dq else None
    return _pack(code).view(H, L, D // 2), s.view(H, L, D // 16), dq


def quant_k_batch(x, need_dq=False):
    H, L, D = x.shape
    p, s, d = quant_q_batch(x, need_dq)
    return p.transpose(1, 2).contiguous(), s, d


def quant_v_batch(x, need_dq=False):
    """v [H,L,D] -> packed [H,L,D/2]（沿 head_dim 配对）, scales [H,D,L/16], dq [H*L,D] or None"""
    H, L, D = x.shape
    xf = x.float()
    b = xf.reshape(H, L // 16, 16, D)
    am = b.abs().amax(2)
    s = (am / 6.0).to(FP8)                      # [H, L/16, D]
    n = (b / s.float().clamp(min=1e-12).unsqueeze(2)).clamp(-6, 6).to(torch.bfloat16)
    a = n.abs()
    idx = ((a > 0.25).to(torch.uint8) + (a > 0.75).to(torch.uint8) + (a > 1.25).to(torch.uint8) +
           (a > 1.75).to(torch.uint8) + (a > 2.5).to(torch.uint8) + (a > 3.5).to(torch.uint8) +
           (a > 5.0).to(torch.uint8))
    code = (idx | ((n < 0).to(torch.uint8) << 3)).reshape(H * L, D)
    dq = None
    if need_dq:
        lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
        dq = lvl * s.float().repeat_interleave(16, dim=1).reshape(H * L, D)
    return _pack(code).view(H, L, D // 2), s.transpose(1, 2).contiguous(), dq


def stack_parts_batch(q, k, v, level):
    H, L, D = q.shape
    n2 = level == 2
    qp1, qs1, qd1 = quant_q_batch(q, n2)
    kp1, ks1, kd1 = quant_k_batch(k, n2)
    vp1, vs1, vd1 = quant_v_batch(v, n2)
    if level == 1:
        return qp1, qs1, kp1, ks1, vp1, vs1, qp1, qs1, kp1, ks1, vp1, vs1
    qp2, qs2, _ = quant_q_batch((q.reshape(H * L, D).float() - qd1).view(H, L, D).to(q.dtype))
    kp2, ks2, _ = quant_k_batch((k.reshape(H * L, D).float() - kd1).view(H, L, D).to(k.dtype))
    vp2, vs2, _ = quant_v_batch((v.reshape(H * L, D).float() - vd1).view(H, L, D).to(v.dtype))
    return qp1, qs1, kp1, ks1, vp1, vs1, qp2, qs2, kp2, ks2, vp2, vs2


# ---------------- kernel ----------------
@triton.jit
def _lvl7(idx):
    f = idx.to(tl.float32)
    return tl.where(f == 0., 0., tl.where(f == 1., .5, tl.where(f == 2., 1., tl.where(
        f == 3., 1.5, tl.where(f == 4., 2., tl.where(f == 5., 3., tl.where(f == 6., 4., 6.)))))))


@triton.jit
def _quant_p(x, BM: tl.constexpr, BN: tl.constexpr):
    xb = tl.reshape(x, (BM, BN // 16, 16))
    am = tl.max(tl.abs(xb), 2)
    s = (am * 0.16666666666666666).to(tl.float8e4nv)
    se = tl.maximum(s.to(tl.float32), 1e-12)
    n = xb / se[:, :, None]
    a = tl.abs(n)
    idx = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) +
           (a > 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) +
           (a > 5.0).to(tl.uint8))
    neg = (n < 0).to(tl.uint8)
    code = idx | (neg << 3)
    d = _lvl7(idx) * tl.where(neg > 0, -1.0, 1.0) * se[:, :, None]
    d = tl.reshape(d, (BM, BN))
    lo, hi = tl.split(tl.reshape(code, (BM, BN // 2, 2)))
    packed = lo | (hi << 4)
    return packed, s, d


@triton.jit
def flash_fp4(Q, QS, Q2, QS2, K1, K1S, V1, V1S, K2, K2S, V2, V2S, O,
              L: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              NM, L2: tl.constexpr, SCALE: tl.constexpr, LT, OUT16: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    prog = pid_m + NM * h
    D2: tl.constexpr = D // 2
    D16: tl.constexpr = D // 16
    L16: tl.constexpr = L // 16
    NB16: tl.constexpr = BN // 16
    NB2: tl.constexpr = BN // 2
    rm = pid_m * BM + tl.arange(0, BM)
    rdk = tl.arange(0, D2)
    rds = tl.arange(0, D16)
    rn = tl.arange(0, BN)
    rp = tl.arange(0, NB2)
    rs = tl.arange(0, NB16)
    rd = tl.arange(0, D)

    qo = (h.to(tl.int64) * L + rm[:, None]) * D2
    q1 = tl.load(Q + qo + rdk[None, :])
    qs1 = tl.load(QS + (h.to(tl.int64) * L + rm[:, None]) * D16 + rds[None, :])
    if L2:
        q2 = tl.load(Q2 + qo + rdk[None, :])
        qs2 = tl.load(QS2 + (h.to(tl.int64) * L + rm[:, None]) * D16 + rds[None, :])
    ipack = rp[None, :] + tl.zeros((BM, 1), tl.int32)
    iscale = rs[None, :] + tl.zeros((BM, 1), tl.int32)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    for j in range(0, L, BN):
        ko = h.to(tl.int64) * L * D2 + (j + rn)[None, :] * D2 + rdk[:, None]
        kp1 = tl.load(K1 + ko)
        ks1 = tl.load(K1S + (h.to(tl.int64) * L + (j + rn))[:, None] * D16 + rds[None, :])
        s = tl.dot_scaled(q1, qs1, "e2m1", kp1, ks1, "e2m1", out_dtype=tl.float32)
        if L2:
            kp2 = tl.load(K2 + ko)
            ks2 = tl.load(K2S + (h.to(tl.int64) * L + (j + rn))[:, None] * D16 + rds[None, :])
            s = tl.dot_scaled(q2, qs2, "e2m1", kp1, ks1, "e2m1", acc=s, out_dtype=tl.float32)
            s = tl.dot_scaled(q1, qs1, "e2m1", kp2, ks2, "e2m1", acc=s, out_dtype=tl.float32)
            s = tl.dot_scaled(q2, qs2, "e2m1", kp2, ks2, "e2m1", acc=s, out_dtype=tl.float32)
        s = s * SCALE
        if LT < L:
            s = tl.where((j + rn)[None, :] < LT, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        m_i = m_new

        pp1, ps1, pd1 = _quant_p(p, BM, BN)
        pp1 = tl.gather(pp1, ipack, 1)
        ps1 = tl.gather(ps1, iscale, 1)
        if L2:
            pp2, ps2, _ = _quant_p(p - pd1, BM, BN)
            pp2 = tl.gather(pp2, ipack, 1)
            ps2 = tl.gather(ps2, iscale, 1)

        vo = h.to(tl.int64) * L * D2 + (j + rn)[:, None] * D2 + rdk[None, :]
        vp1 = tl.load(V1 + vo)
        vs1 = tl.load(V1S + h.to(tl.int64) * D * L16 + rd[:, None] * L16 + (j // 16 + tl.arange(0, NB16))[None, :])
        acc = tl.dot_scaled(pp1, ps1, "e2m1", vp1, vs1, "e2m1", acc=acc, rhs_k_pack=False, out_dtype=tl.float32)
        if L2:
            vp2 = tl.load(V2 + vo)
            vs2 = tl.load(V2S + h.to(tl.int64) * D * L16 + rd[:, None] * L16 + (j // 16 + tl.arange(0, NB16))[None, :])
            acc = tl.dot_scaled(pp2, ps2, "e2m1", vp1, vs1, "e2m1", acc=acc, rhs_k_pack=False, out_dtype=tl.float32)
            acc = tl.dot_scaled(pp1, ps1, "e2m1", vp2, vs2, "e2m1", acc=acc, rhs_k_pack=False, out_dtype=tl.float32)
            acc = tl.dot_scaled(pp2, ps2, "e2m1", vp2, vs2, "e2m1", acc=acc, rhs_k_pack=False, out_dtype=tl.float32)

    o = acc / l_i[:, None]
    if OUT16:
        tl.store(O + (h.to(tl.int64) * L + rm[:, None]) * D + rd[None, :], o.to(tl.bfloat16),
                 mask=rm[:, None] < LT)
    else:
        tl.store(O + (h.to(tl.int64) * L + rm[:, None]) * D + rd[None, :], o, mask=rm[:, None] < LT)


def flash_attn(q, k, v, level=1, BM=64, BN=64, num_warps=4, num_stages=2):
    """q,k,v [H,L,128] bf16 -> [H,L,128] fp32（批量量化 + 单次 grid 启动）"""
    H, L, D = q.shape
    assert L % BM == 0 and L % BN == 0 and D == 128
    NM = L // BM
    qp1, qs1, kp1, ks1, vp1, vs1, qp2, qs2, kp2, ks2, vp2, vs2 = stack_parts_batch(q, k, v, level)
    o = torch.empty(H, L, D, device=DEV, dtype=torch.float32)
    flash_fp4[(NM, H)](
        qp1.view(torch.uint8), qs1, qp2.view(torch.uint8), qs2,
        kp1.view(torch.uint8), ks1, vp1.view(torch.uint8), vs1,
        kp2.view(torch.uint8), ks2, vp2.view(torch.uint8), vs2, o,
        L, D, BM, BN, NM, level == 2, 1.0 / math.sqrt(D), L, False,
        num_warps=num_warps, num_stages=num_stages)
    return o


_FC = {}


def _fast_bufs(H, Lp, D, CH):
    """按头分块 CH 的常驻缓冲；O 保持全量 [H,Lp,D]。"""
    key = (H, Lp, D, CH)
    b = _FC.get(key)
    if b is None:
        u8, bf = torch.uint8, torch.bfloat16
        b = {'qp': torch.zeros(CH * Lp, D // 2, device=DEV, dtype=u8),
             'qs': torch.zeros(CH * Lp, D // 16, device=DEV, dtype=FP8),
             'kp': torch.zeros(CH * Lp, D // 2, device=DEV, dtype=u8),
             'ks': torch.zeros(CH * Lp, D // 16, device=DEV, dtype=FP8),
             'vp': torch.zeros(CH * Lp, D // 2, device=DEV, dtype=u8),
             'sct': torch.zeros(CH * (Lp // 16), D, device=DEV, dtype=FP8),
             'vs': torch.zeros(CH, D, Lp // 16, device=DEV, dtype=FP8),
             'code': torch.zeros(CH * Lp, D, device=DEV, dtype=u8),
             'o': torch.empty(H, Lp, D, device=DEV, dtype=bf)}
        while len(_FC) >= 3:
            _FC.pop(next(iter(_FC)))
        _FC[key] = b
    return b


def flash_attn_masked(q, k, v, level=1, BM=64, BN=64, num_warps=4, num_stages=2, CH=8):
    """引擎用：q,k,v [H,L,128] bf16，任意 L、任意 strides；level1 零复制+按头分块常驻缓冲。"""
    H, L, D = q.shape
    mult = BM * BN // math.gcd(BM, BN)
    Lp = ((L + mult - 1) // mult) * mult
    NM = Lp // BM
    if level == 1:
        from quant_fused import quant_qk_m, quant_v_m
        b = _fast_bufs(H, Lp, D, CH)
        o = b['o']
        for h0 in range(0, H, CH):
            h1 = min(h0 + CH, H)
            c = h1 - h0
            qs_ = q[h0:h1]
            ks_ = k[h0:h1]
            vv_ = v[h0:h1]
            qp, qsc = quant_qk_m(qs_, Lp, out=(b['qp'][:c * Lp], b['qs'][:c * Lp]))
            kp, ksc = quant_qk_m(ks_, Lp, out=(b['kp'][:c * Lp], b['ks'][:c * Lp]))
            vp, vsc = quant_v_m(vv_, Lp, CH=c, out=(b['vp'][:c * Lp], b['sct'][:c * (Lp // 16)],
                                                    b['code'][:c * Lp], b['vs'][:c]))
            flash_fp4[(NM, c)](
                qp.view(torch.uint8), qsc, qp.view(torch.uint8), qsc,
                kp.view(torch.uint8), ksc, vp.view(torch.uint8), vsc,
                kp.view(torch.uint8), ksc, vp.view(torch.uint8), vsc, o[h0:h1],
                Lp, D, BM, BN, NM, False, 1.0 / math.sqrt(D), L, True,
                num_warps=num_warps, num_stages=num_stages)
        return o[:, :L]
    pad = Lp - L
    q2 = torch.nn.functional.pad(q, (0, 0, 0, pad)).contiguous()
    k2_ = torch.nn.functional.pad(k, (0, 0, 0, pad)).contiguous()
    v2_ = torch.nn.functional.pad(v, (0, 0, 0, pad)).contiguous()
    from quant_fused import stack_parts_fused
    qp1, qs1, kp1, ks1, vp1, vs1, qp2, qs2, kp2, ks2, vp2, vs2 = stack_parts_fused(q2, k2_, v2_, level)
    o = torch.empty(H, Lp, D, device=DEV, dtype=torch.bfloat16)
    flash_fp4[(NM, H)](
        qp1.view(torch.uint8), qs1, qp2.view(torch.uint8), qs2,
        kp1.view(torch.uint8), ks1, vp1.view(torch.uint8), vs1,
        kp2.view(torch.uint8), ks2, vp2.view(torch.uint8), vs2, o,
        Lp, D, BM, BN, NM, level == 2, 1.0 / math.sqrt(D), L, True,
        num_warps=num_warps, num_stages=num_stages)
    return o[:, :L]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if mode == 'test':
        torch.manual_seed(0)
        H, L, D = 4, 1024, 128
        q = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
        ref = Fn.scaled_dot_product_attention(q.float(), k.float(), v.float())
        bf = Fn.scaled_dot_product_attention(q, k, v).float()
        print('bf16-sdpa vs fp32 ref  relL2=%.4f' % ((bf - ref).norm() / ref.norm()).item(), flush=True)
        for lv in (1, 2):
            t0 = time.perf_counter()
            o = flash_attn(q, k, v, lv)
            torch.cuda.synchronize()
            print('kernel k%d vs fp32  relL2=%.4f   vs bf16-sdpa relL2=%.4f   (%.1fs)'
                  % (lv, (o - ref).norm() / ref.norm(), (o - bf).norm() / bf.norm(),
                     time.perf_counter() - t0), flush=True)
    elif mode == 'bench':
        H, D = 56, 128
        for L in [int(x) for x in sys.argv[2:]] or [1024, 4096]:
            q = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
            k = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
            v = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
            def bench(fn, n=8):
                for _ in range(2): fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n): fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n
            t_sdpa = bench(lambda: Fn.scaled_dot_product_attention(q, k, v))
            t_end1 = bench(lambda: flash_attn(q, k, v, 1))
            t_end2 = bench(lambda: flash_attn(q, k, v, 2))
            # 纯核（预算化 + 单次启动，纳秒级 host 开销）
            NM = L // 64
            o = torch.empty(H, L, D, device=DEV, dtype=torch.float32)
            for lv in (1, 2):
                qp1, qs1, kp1, ks1, vp1, vs1, qp2, qs2, kp2, ks2, vp2, vs2 = stack_parts_batch(q, k, v, lv)
                def raw():
                    flash_fp4[(NM, H)](qp1.view(torch.uint8), qs1, qp2.view(torch.uint8), qs2,
                                       kp1.view(torch.uint8), ks1, vp1.view(torch.uint8), vs1,
                                       kp2.view(torch.uint8), ks2, vp2.view(torch.uint8), vs2,
                                       o, L, D, 64, 64, NM, lv == 2, 1.0 / math.sqrt(D), L, False,
                                       num_warps=4, num_stages=2)
                    return o
                t = bench(raw)
                print('H%d L%d: sdpa_bf16=%.1fms | fp4-k%d core=%.1fms  end2end=%.1fms  (core %.2fx)'
                      % (H, L, t_sdpa * 1e3, lv, t * 1e3, (t_end1 if lv == 1 else t_end2) * 1e3, t_sdpa / t), flush=True)


if __name__ == '__main__':
    main()
