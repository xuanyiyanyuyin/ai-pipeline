# -*- coding: utf-8 -*-
"""H3-Image BF16 -> NVFP4 k1 converter. FFN+attn quantized; AdaLN/glue kept BF16/FP32.
Self-tests E2M1 packing + round-trip before full conversion. BF16 source untouched."""
import os, glob, json, time, shutil
import torch
from safetensors.torch import load_file, save_file
from torch._C import _ScalingType as ST, _SwizzleType as SW

def ceil_div(a, b): return (a + b - 1) // b
def to_blocked(m):
    """NVFP4 scale swizzle (128x4 tile, 32x4x4 internal) per cuBLAS block-scaling layout."""
    rows, cols = m.shape; nr = ceil_div(rows, 128); nc = ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    if (rows, cols) != (pr, pc):
        p = torch.zeros((pr, pc), device=m.device, dtype=m.dtype); p[:rows, :cols] = m
    else:
        p = m
    return p.view(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()

def fp4_gemm(a4, b4, sa_sw, ga, sb_sw, gb, out_dtype=torch.float32):
    """Verified NVFP4 W4A4 GEMM. ga/gb are ENCODE globals; kernel needs 1/g. b4=(K//2,N)."""
    return torch._scaled_mm_v2(a4, b4,
        scale_a=[sa_sw, (1.0 / ga)], recipe_a=[ST.BlockWise1x16, ST.TensorWise], swizzle_a=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        scale_b=[sb_sw, (1.0 / gb)], recipe_b=[ST.BlockWise1x16, ST.TensorWise], swizzle_b=[SW.SWIZZLE_32_4_4, SW.NO_SWIZZLE],
        bias=None, out_dtype=out_dtype)

SRC = r'D:\ai\models\MiniMax-H3-Image\transformer'
DST = r'D:\ai\models\MiniMax-H3-Image-FP4\transformer'
os.makedirs(DST, exist_ok=True)
fp4 = torch.float4_e2m1fn_x2
fp8 = torch.float8_e4m3fn
DEV = 'cuda'
LV = torch.tensor([0., 0.5, 1, 1.5, 2, 3, 4, 6], device=DEV)
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)
E2M1_MAX = 6.0
FP8_MAX = 448.0

# quantize targets: per-token matmul core (FFN + attention projections)
def is_target(name):
    if not name.endswith('.weight'):
        return False
    return any(s in name for s in ('mlp.fc1', 'mlp.fc2', 'attn.qkv_proj', 'attn.out_proj'))

def quant_nvfp4(W):
    """W (N,K) -> packed uint8 (N,K//2), scale fp8 (N,K//16), global fp32 (1,)"""
    N, K = W.shape
    Wf = W.to(DEV).float()
    amax_t = Wf.abs().max().clamp(min=1e-12)
    g = (FP8_MAX * E2M1_MAX) / amax_t                      # per-tensor global scale
    blocks = Wf.reshape(N, K // 16, 16)
    amax_b = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale_fp8 = ((amax_b / E2M1_MAX) * g).to(fp8)          # block scale in fp8
    scale_eff = scale_fp8.float().clamp(min=1e-12)
    norm = (blocks * g / scale_eff).clamp(-E2M1_MAX, E2M1_MAX)
    idx = torch.bucketize(norm.abs(), MID).to(torch.uint8)  # 0..7 magnitude code
    sign = (norm < 0).to(torch.uint8) << 3
    code = (sign | idx).reshape(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed.cpu(), scale_fp8.squeeze(-1).reshape(N, K // 16).cpu(), g.reshape(1).cpu()

def dequant_nvfp4(packed, scale_fp8, g, N, K):
    p = packed.to(DEV)
    lo = p & 0x0F; hi = (p >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K)
    idx = (codes & 0x07).long()
    sign = torch.where((codes & 0x08) > 0, -1.0, 1.0)
    vals = LV[idx] * sign
    se = scale_fp8.to(DEV).float().reshape(N, K // 16, 1)
    return (vals.reshape(N, K // 16, 16) * se / g.to(DEV)).reshape(N, K)

def selftest():
    fp = sorted(glob.glob(os.path.join(SRC, '*.safetensors')))[0]
    sd = load_file(fp)
    key = next(k for k in sd if is_target(k))
    W = sd[key][:256].float()
    N, K = W.shape
    Wp, Ws, Wg = quant_nvfp4(W)
    Wq = dequant_nvfp4(Wp, Ws, Wg, N, K)
    relw = ((Wq - W.to(DEV)).norm() / W.to(DEV).norm()).item() * 100
    # end-to-end W4A4 via verified _scaled_mm_v2 path
    M = 128
    X = torch.randn(M, K)
    Xp, Xs, Xg = quant_nvfp4(X)
    a4 = Xp.to(DEV).view(fp4); b4 = Wp.to(DEV).view(fp4).t()
    out = fp4_gemm(a4, b4, to_blocked(Xs.to(DEV)), Xg.to(DEV), to_blocked(Ws.to(DEV)), Wg.to(DEV))
    ref = X.to(DEV) @ W.to(DEV).t()
    rele = ((out - ref).norm() / ref.norm()).item() * 100
    print(f'[selftest] {key} {tuple(W.shape)} weight_roundtrip={relw:.3f}%  W4A4_GEMM_end2end={rele:.2f}%', flush=True)
    assert relw < 15.0, f'weight roundtrip too high: {relw}'
    assert rele < 25.0, f'W4A4 GEMM end2end too high (packing/scale wrong?): {rele}'
    return relw

def main():
    t0 = time.time()
    rel = selftest()
    shards = sorted(glob.glob(os.path.join(SRC, '*.safetensors')))
    print(f'[convert] {len(shards)} shards, target=FFN+attn FP4 k1, rest BF16/FP32 copy', flush=True)
    tot_in = tot_out = nq = nk = 0
    manifest = {'format': 'nvfp4_k1', 'e2m1_max': 6.0, 'fp8_max': 448.0, 'block': 16,
                'target_suffixes': ['mlp.fc1', 'mlp.fc2', 'attn.qkv_proj', 'attn.out_proj'],
                'keys': 'name.fp4=uint8(N,K//2) name.scale=fp8(N,K//16) name.global=fp32(1)',
                'selftest_relL2_pct': rel, 'shards': []}
    for si, fp in enumerate(shards):
        ts = time.time()
        sd = load_file(fp)
        out = {}
        sb = 0
        for name, ten in sd.items():
            sb += ten.numel() * ten.element_size()
            if is_target(name) and ten.dim() == 2 and ten.shape[1] % 16 == 0:
                packed, scale, g = quant_nvfp4(ten)
                out[name + '.fp4'] = packed
                out[name + '.scale'] = scale
                out[name + '.global'] = g
                nq += 1
            else:
                out[name] = ten.contiguous()
                nk += 1
        base = os.path.basename(fp)
        save_file(out, os.path.join(DST, base))
        ob = sum(t.numel() * t.element_size() for t in out.values())
        tot_in += sb; tot_out += ob
        manifest['shards'].append({'file': base, 'in_bytes': sb, 'out_bytes': ob})
        print(f'[shard {si+1}/{len(shards)}] {base} in {sb/1e9:.2f}GB -> out {ob/1e9:.2f}GB  {time.time()-ts:.1f}s  (quant {nq} keep {nk})', flush=True)
        del sd, out
        torch.cuda.empty_cache()
    manifest['total_in_bytes'] = tot_in; manifest['total_out_bytes'] = tot_out
    manifest['elapsed_s'] = time.time() - t0
    json.dump(manifest, open(os.path.join(DST, 'fp4_manifest.json'), 'w'), indent=2)
    print(f'RESULT: DONE quant_tensors={nq} kept={nk} in={tot_in/1e9:.2f}GB out={tot_out/1e9:.2f}GB '
          f'ratio={tot_in/tot_out:.2f}x elapsed={time.time()-t0:.1f}s selftest_relL2={rel:.3f}%', flush=True)

if __name__ == '__main__':
    main()