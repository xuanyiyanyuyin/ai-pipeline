# -*- coding: utf-8 -*-
import sys, time, torch
import torch.nn.functional as Fn
sys.path.insert(0, r'D:\ai')
import h3image_engine_fp4 as F
DEV = 'cuda'; fp8 = torch.float8_e4m3fn
LV = torch.tensor([0., .5, 1, 1.5, 2, 3, 4, 6], device=DEV)
MID = F.MID
def quant_raw(x2):
    M, K = x2.shape
    g = (448 * 6) / x2.abs().amax().float().clamp(min=1e-12)
    b = x2.reshape(M, K // 16, 16)
    am = b.abs().amax(-1, keepdim=True).float().clamp(min=1e-12)
    s = ((am / 6) * g).to(fp8); se = s.float().clamp(min=1e-12)
    n = (b * (g / se).to(x2.dtype)).clamp(-6, 6)
    idx = torch.bucketize(n.abs(), MID.to(n.dtype)).to(torch.uint8)
    code = ((n < 0).to(torch.uint8) << 3) | idx
    code = code.reshape(M, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, code, s.squeeze(-1), g
def dequant(code, s, g):
    lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    se = s.float().clamp(min=1e-12).repeat_interleave(16, dim=1)
    return (lvl * se / g).reshape(code.shape)
def parts(x, k):
    out = []; r = x.float()
    for _ in range(k):
        p, c, s, g = quant_raw(r.to(torch.bfloat16))
        d = dequant(c, s, g)
        out.append((p, F.to_blocked(s), g, d)); r = r - d
    return out
def gemm(a, b):
    return F.fp4_gemm(a[0].view(F.fp4), b[0].view(F.fp4).t(), a[1], a[2], b[1], b[2]).float()
torch.manual_seed(0)
H, L, D = 4, 256, 128
q = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
k = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
v = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
rels = []
t0 = time.perf_counter()
for h in range(H):
    qp = parts(q[h], 2); kp = parts(k[h], 2); vp = parts(v[h].t().contiguous(), 2)
    s = torch.zeros(L, L, device=DEV)
    for a in qp:
        for b in kp:
            s += gemm(a, b)
    p = torch.softmax(s / (D ** 0.5), -1).to(torch.bfloat16)
    pp = parts(p, 2)
    o = torch.zeros(L, D, device=DEV)
    for a in pp:
        for b in vp:
            o += gemm(a, b)
    ref = Fn.scaled_dot_product_attention(q[h:h+1], k[h:h+1], v[h:h+1])[0].float()
    rels.append(((o - ref).norm() / ref.norm()).item())
torch.cuda.synchronize()
print('fp4-k2 attention out relL2 per-head:', ['%.4f' % r for r in rels], 'mean=%.4f' % (sum(rels) / H))
print('prototype time %.2fs (H%d L%d, correctness only)' % (time.perf_counter() - t0, H, L))
print('OK')