# -*- coding: utf-8 -*-
"""AdaLN modulation precompute: 6 PDD steps x 50 layers x 6 mods x 3 modalities -> ~58MB cache.
Uses h3_engine.time_embed/lin verbatim => bit-exact vs h3image_engine.py:204-229. BF16 source read-only."""
import os, sys, glob, json, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'D:\ai')
import h3_engine as he

SRC = r'D:\ai\models\MiniMax-H3-Image\transformer'
DST = r'D:\ai\models\MiniMax-H3-Image-FP4\transformer'
PDD_CFG = r'D:\ai\models\MiniMax-H3-Image\pdd_config.json'
DEV = 'cuda'; HID = 5376; MODE_NUM = 3
os.makedirs(DST, exist_ok=True)

def main():
    t0 = time.time()
    cfg = json.load(open(PDD_CFG, encoding='utf-8'))
    sig0 = torch.tensor(cfg['sigmas'], dtype=torch.float64)
    ts = torch.tensor([1.0 - float(s) for s in sig0[:-1]], dtype=torch.float32)  # EXACT as run():486
    nsteps = int(cfg['num_inference_steps'])
    print(f'[adaln] {nsteps} steps t_i={[round(float(x),6) for x in ts]}', flush=True)

    need = ['time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
            'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias',
            'final_layer.adaln_proj.linear.weight', 'final_layer.adaln_proj.linear.bias']
    need += [f'blocks.{j}.adaln_proj.linear.{w}' for j in range(50) for w in ('weight', 'bias')]
    W = {}
    for fp in sorted(glob.glob(os.path.join(SRC, '*.safetensors'))):
        sd = load_file(fp)
        for k in need:
            if k in sd:
                W[k] = sd[k]
        del sd
        if len(W) == len(need):
            break
    print(f'[adaln] loaded {len(W)}/{len(need)} weight tensors', flush=True)
    assert len(W) == len(need), f'missing keys: {set(need)-set(W)}'

    cache = {}
    for i in range(nsteps):
        uniq_t = torch.tensor([float(ts[i])], dtype=torch.float32)
        temb = he.time_embed(uniq_t.to(DEV))
        temb = he.lin(temb, W['time_embedder.proj_in.weight'].to(DEV), W['time_embedder.proj_in.bias'].to(DEV))
        temb = he.lin(F.silu(temb), W['time_embedder.proj_out.weight'].to(DEV), W['time_embedder.proj_out.bias'].to(DEV))
        st = F.silu(temb.float())
        for j in range(50):
            aw = W[f'blocks.{j}.adaln_proj.linear.weight'].to(DEV)
            ab = W[f'blocks.{j}.adaln_proj.linear.bias'].to(DEV)
            a = he.lin(st.to(aw.dtype), aw, ab).view(-1, 6 * HID)   # [3, 6*HID]
            mods = torch.stack(a.chunk(6, dim=-1), dim=0)            # [6, 3, HID]
            cache[f's{i}_l{j}'] = mods.to(torch.bfloat16).cpu()
        fw = W['final_layer.adaln_proj.linear.weight'].to(DEV)
        fb = W['final_layer.adaln_proj.linear.bias'].to(DEV)
        ao = he.lin(st.to(fw.dtype), fw, fb).view(-1, 2 * HID)
        cache[f's{i}_final'] = torch.stack(ao.chunk(2, dim=-1), dim=0).to(torch.bfloat16).cpu()
        print(f'[adaln] step {i+1}/{nsteps} done  {time.time()-t0:.1f}s', flush=True)

    out = os.path.join(DST, 'adaln_cache.safetensors')
    save_file(cache, out)
    sz = os.path.getsize(out)
    # verify bit-exact: recompute step0 layer0 via same path, compare
    uniq_t = torch.tensor([float(ts[0])], dtype=torch.float32)
    temb = he.time_embed(uniq_t.to(DEV))
    temb = he.lin(temb, W['time_embedder.proj_in.weight'].to(DEV), W['time_embedder.proj_in.bias'].to(DEV))
    temb = he.lin(F.silu(temb), W['time_embedder.proj_out.weight'].to(DEV), W['time_embedder.proj_out.bias'].to(DEV))
    aw = W['blocks.0.adaln_proj.linear.weight'].to(DEV)
    a = he.lin(F.silu(temb.float()).to(aw.dtype), aw, W['blocks.0.adaln_proj.linear.bias'].to(DEV)).view(-1, 6*HID)
    ref = torch.stack(a.chunk(6, dim=-1), 0).to(torch.bfloat16).cpu()
    md = (ref.float() - cache['s0_l0'].float()).abs().max().item()
    print(f'RESULT: DONE cache={sz/1e6:.2f}MB keys={len(cache)} verify_maxdiff={md:.6f} elapsed={time.time()-t0:.1f}s', flush=True)
    assert md == 0.0, 'precompute not bit-exact!'

if __name__ == '__main__':
    main()