# -*- coding: utf-8 -*-
"""融合 host 量化（level-1 路径）：一条 kernel 完成 分块尺度→码→nibble 打包。
数值与 flash_fp4 的 torch 版逐位一致（同 amax/同 ÷6/同 bf16 舍入/同比较分桶）。
Q/K: X [H,L,128] bf16 -> packed [H,L,64] u8, scales [H,L,8] fp8（K 由 host 一次 permute 转置）
V:   X [H,L,128] bf16 -> packed [H,L,64], scales [H,128,L/16]（按 key 16 分组；host 一次 permute）
"""
import sys
sys.path.insert(0, r'D:\ai\ve_triton')
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
DEV = 'cuda'


@triton.jit
def _codes16(xb, BM: tl.constexpr, K: tl.constexpr):
    am = tl.maximum(tl.max(tl.abs(xb), 2), 1e-12)
    s = (am / 6.0).to(tl.float8e4nv)
    se = tl.maximum(s.to(tl.float32), 1e-12)
    n = xb / se[:, :, None]
    n = tl.minimum(tl.maximum(n, -6.0), 6.0).to(tl.bfloat16)
    a = tl.abs(n.to(tl.float32))
    idx = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) +
           (a > 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) +
           (a > 5.0).to(tl.uint8))
    neg = (n < 0).to(tl.uint8)
    code = idx | (neg << 3)
    lo, hi = tl.split(tl.reshape(code, (BM, K // 2, 2)))
    return (lo | (hi << 4)), s


@triton.jit
def quant_qk_kernel(X, PACK, SC, N, K: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    msk = rm < N
    rk = tl.arange(0, K)
    x = tl.load(X + rm[:, None].to(tl.int64) * K + rk[None, :], mask=msk[:, None], other=0.0)
    xb = tl.reshape(x.to(tl.float32), (BM, K // 16, 16))
    packed, s = _codes16(xb, BM, K)
    tl.store(PACK + rm[:, None].to(tl.int64) * (K // 2) + tl.arange(0, K // 2)[None, :], packed, mask=msk[:, None])
    tl.store(SC + rm[:, None].to(tl.int64) * (K // 16) + tl.arange(0, K // 16)[None, :], s, mask=msk[:, None])


@triton.jit
def quant_v_scales_kernel(X, SCT, L, SH, SL, LP: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    h = tl.program_id(0); pid = tl.program_id(1)
    G: tl.constexpr = BLK // 16
    rm = pid * BLK + tl.arange(0, BLK)
    msk = rm < L
    rd = tl.arange(0, D)
    x = tl.load(X + h.to(tl.int64) * SH + rm[:, None].to(tl.int64) * SL + rd[None, :],
                mask=msk[:, None], other=0.0)
    xb = tl.reshape(x.to(tl.float32), (G, 16, D))
    am = tl.maximum(tl.max(tl.abs(xb), 1), 1e-12)
    s = (am / 6.0).to(tl.float8e4nv)
    rg = tl.arange(0, G)
    gvalid = ((pid * G + rg) * 16 < L)[:, None]
    tl.store(SCT + (h.to(tl.int64) * (LP // 16) + pid * G + rg)[:, None] * D + rd[None, :], s, mask=gvalid)


@triton.jit
def quant_v_codes_kernel(X, CODE, L, SH, SL, LP: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    h = tl.program_id(0); pid = tl.program_id(1)
    G: tl.constexpr = BLK // 16
    rm = pid * BLK + tl.arange(0, BLK)
    msk = rm < L
    rd = tl.arange(0, D)
    x = tl.load(X + h.to(tl.int64) * SH + rm[:, None].to(tl.int64) * SL + rd[None, :],
                mask=msk[:, None], other=0.0)
    xb = tl.reshape(x.to(tl.float32), (G, 16, D))
    am = tl.maximum(tl.max(tl.abs(xb), 1), 1e-12)
    se = tl.maximum((am / 6.0).to(tl.float8e4nv).to(tl.float32), 1e-12)
    n = xb / se[:, None, :]
    n = tl.minimum(tl.maximum(n, -6.0), 6.0).to(tl.bfloat16)
    a = tl.abs(n.to(tl.float32))
    idx = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) +
           (a > 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) +
           (a > 5.0).to(tl.uint8))
    neg = (n < 0).to(tl.uint8)
    code = tl.reshape(idx | (neg << 3), (BLK, D))
    tl.store(CODE + (h.to(tl.int64) * LP + rm[:, None].to(tl.int64)) * D + rd[None, :], code,
             mask=msk[:, None])


def quant_qk(x):
    """x [H,L,D] bf16 -> packed [H,L,D/2], scales [H,L,D/16]"""
    H, L, D = x.shape
    N = H * L
    packed = torch.empty(N, D // 2, device=x.device, dtype=torch.uint8)
    sc = torch.empty(N, D // 16, device=x.device, dtype=FP8)
    quant_qk_kernel[(triton.cdiv(N, 64),)](x.reshape(N, D).contiguous(), packed, sc, N, D, 64, num_warps=4)
    return packed.view(H, L, D // 2), sc.view(H, L, D // 16)


def quant_v(x):
    """x [H,L,D] bf16 -> packed [H,L,D/2], scales [H,D,L/16]"""
    H, L, D = x.shape
    code = torch.empty(H * L, D, device=x.device, dtype=torch.uint8)
    sct = torch.empty(H * (L // 16), D, device=x.device, dtype=FP8)
    xc = x.contiguous()
    quant_v_scales_kernel[(H, triton.cdiv(L, 64))](xc, sct, L, xc.stride(0), xc.stride(1), L, D, 64, num_warps=4)
    quant_v_codes_kernel[(H, triton.cdiv(L, 64))](xc, code, L, xc.stride(0), xc.stride(1), L, D, 64, num_warps=4)
    c = code.view(H, L, D)
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()
    return packed, sct.view(H, L // 16, D).transpose(1, 2).contiguous()


def stack_parts_fused(q, k, v, level=1):
    """level1/2 均走融合核；level2 的残差 = x - deq1（GPU 内核反量化）。"""
    H, L, D = q.shape
    qp1, qs1 = quant_qk(q)
    kp1t, ks1 = quant_qk(k)
    vp1, vs1 = quant_v(v)
    if level == 1:
        return qp1, qs1, kp1t, ks1, vp1, vs1, qp1, qs1, kp1t, ks1, vp1, vs1
    qd1, kd1, vd1 = deq_parts_fused(qp1, qs1, kp1t, ks1, vp1, vs1, H, L, D)
    qr = (q.reshape(H * L, D).float() - qd1).view(H, L, D).to(q.dtype)
    kr = (k.reshape(H * L, D).float() - kd1).view(H, L, D).to(k.dtype)
    vr = (v.reshape(H * L, D).float() - vd1).view(H, L, D).to(v.dtype)
    qp2, qs2 = quant_qk(qr)
    kp2t, ks2 = quant_qk(kr)
    vp2, vs2 = quant_v(vr)
    return qp1, qs1, kp1t, ks1, vp1, vs1, qp2, qs2, kp2t, ks2, vp2, vs2

@triton.jit
def _deq_level(code_u8):
    idx = code_u8 & 7
    f = idx.to(tl.float32)
    v = tl.where(f == 0., 0., tl.where(f == 1., .5, tl.where(f == 2., 1., tl.where(
        f == 3., 1.5, tl.where(f == 4., 2., tl.where(f == 5., 3., tl.where(f == 6., 4., 6.)))))))
    return v * tl.where((code_u8 & 8) > 0, -1.0, 1.0)


@triton.jit
def deq_qk_kernel(PACK, SC, OUT, N, K: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    msk = rm < N
    rc = tl.arange(0, K // 2)
    p = tl.load(PACK + rm[:, None].to(tl.int64) * (K // 2) + rc[None, :], mask=msk[:, None], other=0)
    lo = p & 15
    hi = p >> 4
    code = tl.interleave(lo, hi)                    # [BM, K]
    rd = tl.arange(0, K)
    rs = tl.arange(0, K // 16)
    s = tl.load(SC + rm[:, None].to(tl.int64) * (K // 16) + rs[None, :], mask=msk[:, None], other=1.0)
    se = s.to(tl.float32).repeat_interleave(16, 1) if False else tl.reshape(
        tl.broadcast_to(s[:, :, None].to(tl.float32), (BM, K // 16, 16)), (BM, K))
    tl.store(OUT + rm[:, None].to(tl.int64) * K + rd[None, :], _deq_level(code) * se, mask=msk[:, None])


@triton.jit
def deq_v_kernel(PACK, SC, OUT, L: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    h = tl.program_id(0); pid = tl.program_id(1)
    rm = pid * BLK + tl.arange(0, BLK)
    rc = tl.arange(0, D // 2)
    p = tl.load(PACK + (h.to(tl.int64) * L + rm[:, None].to(tl.int64)) * (D // 2) + rc[None, :])
    code = tl.interleave(p & 15, p >> 4)
    rd = tl.arange(0, D)
    rg = (pid * BLK + tl.arange(0, BLK)) // 16
    s = tl.load(SC + (h.to(tl.int64) * (L // 16) + (pid * BLK // 16 + tl.arange(0, BLK // 16)))[:, None] * D + rd[None, :])
    se = tl.reshape(tl.broadcast_to(s[:, None, :], (BLK // 16, 16, D)), (BLK, D))
    tl.store(OUT + (h.to(tl.int64) * L + rm[:, None].to(tl.int64)) * D + rd[None, :], _deq_level(code) * se)


def deq_parts_fused(qp, qs, kp, ks, vp, vs, H, L, D):
    """从 fused 量化产物反量化出 dq（fp32），供 k2 残差。kp 为自然布局 [H,L,D/2]。"""
    N = H * L
    qd = torch.empty(N, D, device='cuda', dtype=torch.float32)
    deq_qk_kernel[(triton.cdiv(N, 64),)](qp.reshape(N, D // 2), qs.reshape(N, D // 16), qd, N, D, 64, num_warps=4)
    kd = torch.empty(N, D, device='cuda', dtype=torch.float32)
    kp = kp.reshape(N, D // 2).contiguous()
    deq_qk_kernel[(triton.cdiv(N, 64),)](kp, ks.reshape(N, D // 16), kd, N, D, 64, num_warps=4)
    vd = torch.empty(N, D, device='cuda', dtype=torch.float32)
    sct = vs.transpose(1, 2).contiguous().reshape(H * (L // 16), D)
    deq_v_kernel[(H, L // 64)](vp.reshape(N, D // 2), sct, vd, L, D, 64, num_warps=4)
    return qd, kd, vd

@triton.jit
def quant_qk_kernel_m(X, PACK, SC, L, SH, SL, LP: tl.constexpr, K: tl.constexpr, BM: tl.constexpr):
    h = tl.program_id(0); pid = tl.program_id(1)
    rm = pid * BM + tl.arange(0, BM)
    msk = rm < L
    rk = tl.arange(0, K)
    x = tl.load(X + h.to(tl.int64) * SH + rm[:, None].to(tl.int64) * SL + rk[None, :],
                mask=msk[:, None], other=0.0)
    xb = tl.reshape(x.to(tl.float32), (BM, K // 16, 16))
    packed, s = _codes16(xb, BM, K)
    tl.store(PACK + (h.to(tl.int64) * LP + rm[:, None].to(tl.int64)) * (K // 2) + tl.arange(0, K // 2)[None, :],
             packed, mask=msk[:, None])
    tl.store(SC + (h.to(tl.int64) * LP + rm[:, None].to(tl.int64)) * (K // 16) + tl.arange(0, K // 16)[None, :],
             s, mask=msk[:, None])


def quant_qk_m(x, Lp, out=None):
    """x [H,L,D] bf16（任意 L；支持非连续视图，按 strides 读）-> packed [H,Lp,D/2], scales [H,Lp,D/16]。"""
    H, L, D = x.shape
    assert x.stride(2) == 1
    if out is None:
        packed = torch.zeros(H * Lp, D // 2, device=x.device, dtype=torch.uint8)
        sc = torch.zeros(H * Lp, D // 16, device=x.device, dtype=FP8)
    else:
        packed, sc = out
    quant_qk_kernel_m[(H, triton.cdiv(L, 64))](x, packed, sc, L, x.stride(0), x.stride(1),
                                               Lp, D, 64, num_warps=4)
    return packed.view(H, Lp, D // 2), sc.view(H, Lp, D // 16)


def quant_v_m(x, Lp, CH=8, out=None):
    """V：任意 L，不复制、不补零，按 CH 头分块（码缓冲 H/CH 份）。packed [H,Lp,D/2], scales [H,D,Lp/16]。"""
    H, L, D = x.shape
    if out is None:
        packed = torch.zeros(H * Lp, D // 2, device=x.device, dtype=torch.uint8)
        sct = torch.zeros(H * (Lp // 16), D, device=x.device, dtype=FP8)
        code = torch.zeros(CH * Lp, D, device=x.device, dtype=torch.uint8)
        vs_t = None
    else:
        packed, sct, code, vs_t = out
    pvh = packed.view(H, Lp, D // 2)
    G = triton.cdiv(L, 64)
    for h0 in range(0, H, CH):
        h1 = min(h0 + CH, H)
        xs = x[h0:h1]
        xv = xs.reshape(-1, D)
        quant_v_scales_kernel[(h1 - h0, G)](xs, sct[h0 * (Lp // 16):h1 * (Lp // 16)], L,
                                            xs.stride(0), xs.stride(1), Lp, D, 64, num_warps=4)
        quant_v_codes_kernel[(h1 - h0, G)](xs, code, L, xs.stride(0), xs.stride(1), Lp, D, 64, num_warps=4)
        c = code.view(h1 - h0, Lp, D)
        del xv
        pvh[h0:h1].copy_(c[..., 0::2] | (c[..., 1::2] << 4))
        del c
    if vs_t is None:
        vs = sct.view(H, Lp // 16, D).transpose(1, 2).contiguous()
    else:
        vs_t.copy_(sct.view(H, Lp // 16, D).transpose(1, 2))
        vs = vs_t
    return packed.view(H, Lp, D // 2), vs

@triton.jit
def qkv_rope_quant_kernel(X, WQ, WK, COS, SIN, QP, QS, KP, KS, VP, VS,
                          L, H0, ROFF, LP: tl.constexpr, XK: tl.constexpr, D: tl.constexpr,
                          RD: tl.constexpr, BM: tl.constexpr):
    """qkv[L,XK] bf16(interleaved [h][q,k,v][d]) -> 直接产出 FP4 打包 q/k/v。
    Q/K 尺度按行(16组沿D)；V 尺度按 key 16 分组存 [H*D, Lp/16] 转置布局（与注意力核一致）。
    逐位仿 he.rms + he.rope_apply 的 bf16 舍入链。"""
    h = tl.program_id(0)
    pid = tl.program_id(1)
    habs = H0 + h
    rm = pid * BM + tl.arange(0, BM)
    msk = rm < L
    rd = tl.arange(0, D)
    xo = rm[:, None].to(tl.int64) * XK + habs.to(tl.int64) * (3 * D) + rd[None, :]
    xq = tl.load(X + xo, mask=msk[:, None], other=0.0).to(tl.float32)
    xk = tl.load(X + xo + D, mask=msk[:, None], other=0.0).to(tl.float32)
    xv = tl.load(X + xo + 2 * D, mask=msk[:, None], other=0.0)
    rq = 1.0 / tl.sqrt(tl.sum(xq * xq, 1) / D + 1e-5)
    rk = 1.0 / tl.sqrt(tl.sum(xk * xk, 1) / D + 1e-5)
    wq = tl.load(WQ + rd).to(tl.float32)
    wk = tl.load(WK + rd).to(tl.float32)
    yq = ((xq * rq[:, None]).to(tl.bfloat16).to(tl.float32) * wq[None, :]).to(tl.bfloat16)
    yk = ((xk * rk[:, None]).to(tl.bfloat16).to(tl.float32) * wk[None, :]).to(tl.bfloat16)
    inrd = rd[None, :] < RD
    sc = (rd + RD // 2) % RD
    xs_ = rm[:, None].to(tl.int64) * XK + habs.to(tl.int64) * (3 * D) + sc[None, :]
    xq2 = tl.load(X + xs_, mask=msk[:, None] & inrd, other=0.0).to(tl.float32)
    xk2 = tl.load(X + xs_ + D, mask=msk[:, None] & inrd, other=0.0).to(tl.float32)
    wq2 = tl.load(WQ + sc, mask=inrd, other=0.0).to(tl.float32)
    wk2 = tl.load(WK + sc, mask=inrd, other=0.0).to(tl.float32)
    yq2 = ((xq2 * rq[:, None]).to(tl.bfloat16).to(tl.float32) * wq2[None, :]).to(tl.bfloat16)
    yk2 = ((xk2 * rk[:, None]).to(tl.bfloat16).to(tl.float32) * wk2[None, :]).to(tl.bfloat16)
    co = tl.load(COS + rm[:, None].to(tl.int64) * RD + rd[None, :], mask=msk[:, None] & inrd, other=0.0).to(tl.bfloat16)
    so = tl.load(SIN + rm[:, None].to(tl.int64) * RD + rd[None, :], mask=msk[:, None] & inrd, other=0.0).to(tl.bfloat16)
    hrot = rd[None, :] < RD // 2
    rotq = tl.where(hrot, -yq2, yq2)
    rotk = tl.where(hrot, -yk2, yk2)
    tq = ((yq.to(tl.float32) * co.to(tl.float32)).to(tl.bfloat16).to(tl.float32) +
          (rotq.to(tl.float32) * so.to(tl.float32)).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tk = ((yk.to(tl.float32) * co.to(tl.float32)).to(tl.bfloat16).to(tl.float32) +
          (rotk.to(tl.float32) * so.to(tl.float32)).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    oq = tl.where(inrd, tq, yq)
    ok = tl.where(inrd, tk, yk)
    row = h.to(tl.int64) * LP + (ROFF + rm[:, None]).to(tl.int64)
    qb, qsc = _codes16(tl.reshape(oq.to(tl.float32), (BM, D // 16, 16)), BM, D)
    kb, ksc = _codes16(tl.reshape(ok.to(tl.float32), (BM, D // 16, 16)), BM, D)
    tl.store(QP + row * (D // 2) + tl.arange(0, D // 2)[None, :], qb, mask=msk[:, None])
    tl.store(QS + row * (D // 16) + tl.arange(0, D // 16)[None, :], qsc, mask=msk[:, None])
    tl.store(KP + row * (D // 2) + tl.arange(0, D // 2)[None, :], kb, mask=msk[:, None])
    tl.store(KS + row * (D // 16) + tl.arange(0, D // 16)[None, :], ksc, mask=msk[:, None])
    # V：按 key 16 分组（与注意力核一致）
    G: tl.constexpr = BM // 16
    xvb = tl.reshape(xv.to(tl.float32), (G, 16, D))
    amv = tl.maximum(tl.max(tl.abs(xvb), 1), 1e-12)
    sv = (amv / 6.0).to(tl.float8e4nv)
    sev = tl.maximum(sv.to(tl.float32), 1e-12)
    nv = xvb / sev[:, None, :]
    nv = tl.minimum(tl.maximum(nv, -6.0), 6.0).to(tl.bfloat16)
    av = tl.abs(nv.to(tl.float32))
    idxv = ((av > 0.25).to(tl.uint8) + (av > 0.75).to(tl.uint8) + (av > 1.25).to(tl.uint8) +
            (av > 1.75).to(tl.uint8) + (av > 2.5).to(tl.uint8) + (av > 3.5).to(tl.uint8) +
            (av > 5.0).to(tl.uint8))
    negv = (nv < 0).to(tl.uint8)
    codev = tl.reshape(idxv | (negv << 3), (BM, D))
    lov, hiv = tl.split(tl.reshape(codev, (BM, D // 2, 2)))
    vp = lov | (hiv << 4)
    tl.store(VP + row * (D // 2) + tl.arange(0, D // 2)[None, :], vp, mask=msk[:, None])
    rg = tl.arange(0, G)
    gid = (ROFF // 16) + pid * G + rg
    # gid 是全局组号，必须与全局 padded 长度 LP 比；若用块长 L 会把 ROFF>0 的块全部掩掉
    tl.store(VS + h.to(tl.int64) * (D * (LP // 16)) + rd[:, None].to(tl.int64) * (LP // 16) + gid[None, :],
             tl.trans(sv), mask=(gid * 16 < LP)[None, :])


def qkv_rope_quant(qkv, wq, wk, cosb, sinb, H, L, Lp, CH, out, ROFF=0, torch_out=None):
    """按头分块跑融合核；out 需含 qp/qs/kp/ks/vp/vs 缓冲（CH 头）。ROFF = 该片在整段里的起始行。"""
    for h0 in range(0, H, CH):
        c = min(CH, H - h0)
        qkv_rope_quant_kernel[(c, triton.cdiv(L, 64))](
            qkv, wq, wk, cosb, sinb,
            out['qp'][:c * Lp], out['qs'][:c * Lp], out['kp'][:c * Lp], out['ks'][:c * Lp],
            out['vp'][:c * Lp], out['vs'][:c],
            L, h0, ROFF, Lp, qkv.stride(0), 128, 96, 64, num_warps=4)
        yield h0, c
