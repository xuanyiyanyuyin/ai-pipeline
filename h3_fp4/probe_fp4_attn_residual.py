# -*- coding: utf-8 -*-
import sys, torch
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
    return code.reshape(M, K), s.squeeze(-1), g
def dequant(code, s, g):
    lvl = LV[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    se = s.float().clamp(min=1e-12).repeat_interleave(16, dim=1)
    return (lvl * se / g).reshape(code.shape)
def residual_stack(x, k):
    parts = []; r = x.float()
    for _ in range(k):
        c, s, g = quant_raw(r.to(torch.bfloat16))
        d = dequant(c, s, g)
        parts.append(d); r = r - d
    return parts
torch.manual_seed(0)
H, L, D = 8, 1024, 128
q = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
k = torch.randn(H, L, D, device=DEV, dtype=torch.bfloat16)
scale = 1.0 / (D ** 0.5)
s_ref = (q.float() @ k.float().transpose(-1, -2)) * scale
p_ref = torch.softmax(s_ref, -1)
for kk in (1, 2, 3):
    qp = [residual_stack(q[h], kk) for h in range(H)]
    kp = [residual_stack(k[h], kk) for h in range(H)]
    s_q = torch.zeros_like(s_ref)
    for h in range(H):
        for a in qp[h]:
            for b in kp[h]:
                s_q[h] += (a @ b.transpose(-1, -2)) * scale
    rel = (s_q - s_ref).norm() / s_ref.norm()
    p_q = torch.softmax(s_q, -1)
    l1 = (p_q - p_ref).abs().sum(-1).mean()
    top = (p_ref.argmax(-1) == p_q.argmax(-1)).float().mean()
    gemms = kk * kk
    print('k%d gemms=%2d score_relL2=%.4f softmax_L1=%.4f argmax_agree=%.4f  fp4_gemm_time_vs_bf16=%.2fx' % (
        kk, gemms, rel, l1, top, gemms / 6.7))
print('OK')