# -*- coding: utf-8 -*-
"""H3 VIDEO AdaLN precompute, t-keyed (separable per timestep). Covers video(shift12)+audio(shift3)+0.999 ref.
Uses h3_engine.time_embed/lin verbatim => bit-exact. ADDS cache file only; original BF16 AdaLN kept."""
import os, sys, json, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
sys.path.insert(0, r'D:\ai')
import h3_engine as he

SRC = r'D:\ai\models\MiniMax-H3\Ref2VA\transformer'
DST = r'D:\ai\models\MiniMax-H3-FP4\transformer'
DEV = 'cuda'; HID = 5376
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def schedules(steps):
    u = torch.linspace(1.0, 0.0, steps)
    sig = torch.unique_consecutive(12.0 * u / (1 + 11.0 * u))
    asig = torch.unique_consecutive(3.0 * u / (1 + 2.0 * u))
    tvid = 1.0 - sig; taux = 1.0 - asig
    T = torch.unique(torch.cat([tvid, taux, torch.tensor([0.999])]))
    return T.float()

def main():
    t0 = time.time()
    T = schedules(STEPS).to(DEV)
    n_t = T.shape[0]
    print(f'[video-adaln] steps={STEPS} n_t={n_t} t={[round(float(x),4) for x in T]}', flush=True)
    need = ['time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
            'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias',
            'final_layer.adaln_proj.linear.weight', 'final_layer.adaln_proj.linear.bias']
    need += [f'blocks.{j}.adaln_proj.linear.{w}' for j in range(50) for w in ('weight', 'bias')]
    W = {}
    import glob
    for fp in sorted(glob.glob(os.path.join(SRC, '*.safetensors'))):
        sd = load_file(fp)
        for k in need:
            if k in sd: W[k] = sd[k]
        del sd
        if len(W) == len(need): break
    assert len(W) == len(need)
    temb = he.time_embed(T)
    temb = he.lin(temb, W['time_embedder.proj_in.weight'].to(DEV), W['time_embedder.proj_in.bias'].to(DEV))
    temb = he.lin(F.silu(temb), W['time_embedder.proj_out.weight'].to(DEV), W['time_embedder.proj_out.bias'].to(DEV))
    st = F.silu(temb.float())
    cache = {'t_list': T.cpu()}
    for j in range(50):
        aw = W[f'blocks.{j}.adaln_proj.linear.weight'].to(DEV)
        ab = W[f'blocks.{j}.adaln_proj.linear.bias'].to(DEV)
        a = he.lin(st.to(aw.dtype), aw, ab)                 # [n_t, 96768]
        mods = a.reshape(n_t, 3, 6, HID).permute(2, 0, 1, 3).contiguous()  # [6, n_t, 3, HID]
        cache[f'l{j}'] = mods.to(torch.bfloat16).cpu()
    fw = W['final_layer.adaln_proj.linear.weight'].to(DEV)
    fb = W['final_layer.adaln_proj.linear.bias'].to(DEV)
    ao = he.lin(st.to(fw.dtype), fw, fb)                    # [n_t, 10752]
    cache['final'] = ao.reshape(n_t, 2, HID).to(torch.bfloat16).cpu()
    out = os.path.join(DST, 'adaln_cache.safetensors')
    save_file(cache, out)
    # bit-exact verify: recompute layer0 at T[1] via engine path, compare to cache gather
    tt = T[1:2]
    tb = he.time_embed(tt)
    tb = he.lin(tb, W['time_embedder.proj_in.weight'].to(DEV), W['time_embedder.proj_in.bias'].to(DEV))
    tb = he.lin(F.silu(tb), W['time_embedder.proj_out.weight'].to(DEV), W['time_embedder.proj_out.bias'].to(DEV))
    aw = W['blocks.0.adaln_proj.linear.weight'].to(DEV)
    a = he.lin(F.silu(tb.float()).to(aw.dtype), aw, W['blocks.0.adaln_proj.linear.bias'].to(DEV))
    ref = a.reshape(1, 3, 6, HID).permute(2, 0, 1, 3).to(torch.bfloat16).cpu()
    got = cache['l0'][:, 1:2, :, :]
    md = (ref.float() - got.float()).abs().max().item()
    sz = os.path.getsize(out)
    print(f'RESULT: DONE cache={sz/1e6:.1f}MB keys={len(cache)} verify_maxdiff={md:.6f} elapsed={time.time()-t0:.1f}s (original BF16 kept)', flush=True)
    assert md == 0.0

if __name__ == '__main__':
    main()