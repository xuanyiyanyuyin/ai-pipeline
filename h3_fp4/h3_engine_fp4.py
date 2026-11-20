# -*- coding: utf-8 -*-
"""H3 VIDEO FP4 W4A4 engine: monkey-patch h3_engine (BF16 video) Reader/auto_slice/dit_forward.
Reuses video ref2va conditioning / 3-axis rope / dual scheduler / video+audio VAE unchanged.
FFN+attn via verified NVFP4 _scaled_mm_v2; AdaLN from t-keyed precomputed cache (original BF16 kept).
Usage: python h3_engine_fp4.py --ref <img> --out <mp4> --steps 4 --frames 124 --size 512
"""
import os, sys, glob, json, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from safetensors.torch import load_file
sys.path.insert(0, r"D:\ai")
import h3_engine as he
import h3image_engine_fp4 as F4

FP4_DIR = r"D:\ai\models\MiniMax-H3-FP4\transformer"
CACHE = os.path.join(FP4_DIR, "adaln_cache.safetensors")
DEV = he.DEV; HID = he.HID; MODE_NUM = he.MODE_NUM
fp4 = F4.fp4
RESIDENT = int(os.environ.get("H3_FP4_RESIDENT", "12"))
_cache = None

def _load_cache():
    global _cache
    if _cache is None:
        _cache = load_file(CACHE)
    return _cache

def _schedules(steps):
    u = torch.linspace(1.0, 0.0, steps)
    sig = torch.unique_consecutive(12.0 * u / (1 + 11.0 * u))
    asig = torch.unique_consecutive(3.0 * u / (1 + 2.0 * u))
    return torch.unique(torch.cat([1.0 - sig, 1.0 - asig, torch.tensor([0.999])])).float()

def build_cache(steps):
    """Auto-build (or reuse on-disk) AdaLN t-keyed cache for the ACTUAL steps => no steps mismatch."""
    import glob as _g
    path = os.path.join(FP4_DIR, 'adaln_cache_s%d.safetensors' % steps)
    if os.path.exists(path):
        print('[video-fp4] reuse AdaLN cache steps=%d' % steps, flush=True)
        return load_file(path)
    print('[video-fp4] building AdaLN cache steps=%d ...' % steps, flush=True)
    t0 = time.time()
    SRC = r'D:\ai\models\MiniMax-H3\Ref2VA\transformer'
    T = _schedules(steps).to(DEV); n_t = T.shape[0]
    need = ['time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
            'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias',
            'final_layer.adaln_proj.linear.weight', 'final_layer.adaln_proj.linear.bias']
    need += ['blocks.%d.adaln_proj.linear.%s' % (j, w) for j in range(50) for w in ('weight', 'bias')]
    W = {}
    for fp in sorted(_g.glob(os.path.join(SRC, '*.safetensors'))):
        sd = load_file(fp)
        for k in need:
            if k in sd: W[k] = sd[k]
        del sd
        if len(W) == len(need): break
    temb = he.time_embed(T)
    temb = he.lin(temb, W['time_embedder.proj_in.weight'].to(DEV), W['time_embedder.proj_in.bias'].to(DEV))
    temb = he.lin(torch.nn.functional.silu(temb), W['time_embedder.proj_out.weight'].to(DEV), W['time_embedder.proj_out.bias'].to(DEV))
    st = torch.nn.functional.silu(temb.float())
    cache = {'t_list': T.cpu()}
    for j in range(50):
        aw = W['blocks.%d.adaln_proj.linear.weight' % j].to(DEV)
        a = he.lin(st.to(aw.dtype), aw, W['blocks.%d.adaln_proj.linear.bias' % j].to(DEV))
        cache['l%d' % j] = a.reshape(n_t, 3, 6, HID).permute(2, 0, 1, 3).contiguous().to(torch.bfloat16).cpu()
    fw = W['final_layer.adaln_proj.linear.weight'].to(DEV)
    ao = he.lin(st.to(fw.dtype), fw, W['final_layer.adaln_proj.linear.bias'].to(DEV))
    cache['final'] = ao.reshape(n_t, 2, HID).to(torch.bfloat16).cpu()
    from safetensors.torch import save_file
    save_file(cache, path)
    print('[video-fp4] AdaLN cache steps=%d n_t=%d built %.1fs' % (steps, n_t, time.time() - t0), flush=True)
    return cache

class VideoFP4Reader:
    def __init__(self):
        self.layers = {j: {} for j in range(50)}
        self.sm = {}
        self.n = 50
        for fp in sorted(glob.glob(os.path.join(FP4_DIR, "model-*.safetensors"))):
            sd = load_file(fp)
            for name in list(sd.keys()):
                if name.endswith('.fp4'):
                    bn = name[:-4]
                    N, K = sd[name].shape[0], sd[name].shape[1] * 2
                    rec = {'packed': sd[name].contiguous(), 'sw': F4.to_blocked(sd[bn + '.scale']).contiguous(),
                           'g': sd[bn + '.global'].float(), 'N': N, 'K': K}
                    if bn.startswith('blocks.'):
                        j = int(bn.split('.')[1]); self.layers[j][bn.split('.', 2)[2]] = rec
                    else:
                        self.sm[bn] = rec
                elif name.endswith('.scale') or name.endswith('.global'):
                    continue
                else:
                    if 'adaln_proj' in name or name.startswith('time_embedder'):
                        continue  # from t-keyed cache
                    ten = sd[name]
                    if name.startswith('blocks.'):
                        j = int(name.split('.')[1]); self.layers[j][name.split('.', 2)[2]] = ten.contiguous()
                    else:
                        self.sm[name] = ten
            del sd
    def small(self, dev=DEV):
        out = {}
        for n, v in self.sm.items():
            if isinstance(v, dict):
                out[n] = {'b4': v['packed'].to(dev).view(fp4).t(), 'sw': v['sw'].to(dev), 'g': v['g'].to(dev), 'N': v['N']}
            else:
                out[n] = v.to(dev)
        return out

def _auto_slice(r, resident=3, cache=6, prefetch=True):
    return F4.make_layer_fn(r, RESIDENT)

def fp4_dit_forward(reader, x_video, x_audio, cond, timestep, timestep_indices, token_tags, position_ids,
                    video_indices, audio_indices, text_indices, sm=None, layer_fn=None, verbose=False):
    t0 = time.perf_counter()
    sm = sm or reader.small()
    layer_fn = layer_fn or _auto_slice(reader)
    c = _load_cache()
    inv = sm["rope.inv_freq"].to(torch.float32)
    ve = he.lin(x_video.to(DEV, sm["video_patch_proj.weight"].dtype), sm["video_patch_proj.weight"], sm["video_patch_proj.bias"])
    ae = he.lin(x_audio.to(DEV, sm["audio_patch_proj.weight"].dtype), sm["audio_patch_proj.weight"], sm["audio_patch_proj.bias"])
    te = he.lin(cond.to(DEV, he.DTYPE), sm["condition_proj.weight"], sm["condition_proj.bias"])
    te = F4.fp4_refiner(te, sm)
    L = position_ids.shape[0]
    h = te.new_zeros((te.shape[0], L, te.shape[-1]))
    h = h.index_copy(1, text_indices.to(DEV), te)
    h = h.index_copy(1, video_indices.to(DEV), ve.to(te.dtype))
    h = h.index_copy(1, audio_indices.to(DEV), ae.to(te.dtype))
    idx = (timestep_indices.to(DEV) * MODE_NUM + token_tags.to(DEV)).to(torch.long)
    cos, sin = he.build_rope(position_ids.to(DEV), inv)
    tmap = torch.searchsorted(c['t_list'], timestep.cpu())   # CPU index into CPU cache
    n_t = tmap.shape[0]
    for i in range(reader.n):
        st = layer_fn(i)
        mods = c['l%d' % i][:, tmap, :, :].to(DEV).reshape(6, n_t * MODE_NUM, HID).unbind(0)
        h = F4.fp4_block(h, st, mods, idx, cos, sin)
        if verbose and (i % 10 == 0 or i == reader.n - 1):
            print("[e] layer %d/%d %.1fs VRAM %.2fGB" % (i, reader.n, time.perf_counter() - t0,
                  torch.cuda.memory_allocated() / 1024 ** 3), flush=True)
    hh = he.rms(h, sm["final_layer.norm.weight"])
    fc = c['final'][tmap].to(DEV)                      # [n_t,2,HID]
    sh, sc = fc[:, 0, :], fc[:, 1, :]
    _ti = timestep_indices.to(DEV)
    hh = hh * (1.0 + sc.index_select(0, _ti).unsqueeze(0)) + sh.index_select(0, _ti).unsqueeze(0)
    hh = hh.to(torch.float32)
    vo = he.lin(hh, sm["final_layer.video_out.weight"], sm["final_layer.video_out.bias"]).index_select(1, video_indices.to(DEV))
    ao = he.lin(hh, sm["final_layer.audio_out.weight"], sm["final_layer.audio_out.bias"]).index_select(1, audio_indices.to(DEV))
    return vo, ao

# monkey-patch the BF16 video engine's compute path
he.Reader = VideoFP4Reader
he.auto_slice = _auto_slice
he.dit_forward = fp4_dit_forward

def run_ref2va_tiled(ref, out, steps=20, frames=124, H=2048, W=2048, tile=1024, ov=256, prompt="", seed=42):
    """MultiDiffusion spatial tiling for video DiT: global-RoPE subgrid + overlap blend."""
    import numpy as np, imageio.v3 as iio
    torch.manual_seed(seed)
    r = VideoFP4Reader(); sm = r.small(); lf = _auto_slice(r, RESIDENT)
    v, c = he._vae(); lat = he.encode_ref(v, c, ref); xv_ref = he.patchify(lat); Nr = xv_ref.shape[1]
    n_lat = (frames - 5) // 17 * 5 + 2
    xv = torch.cat([xv_ref, he.patchify(torch.randn(1, 24, n_lat, H // 16, W // 16, device=DEV))], 1)
    Nv = xv.shape[1]
    _cpt = os.environ.get("H3_COND_PT", "D:/ai/h3image_cond_t2i.pt")
    if os.path.exists(_cpt):
        _cd = torch.load(_cpt, map_location="cpu", weights_only=True)
        _hs = _cd["hidden_states"].float(); cond = _hs.unsqueeze(0) if _hs.dim() == 2 else _hs
        Nt = cond.shape[1]; cond_tags = _cd["token_tags"].reshape(-1).to(torch.long)
    else:
        Nt = 16; cond = torch.zeros(1, Nt, 5120); cond_tags = torch.ones(Nt, dtype=torch.long)
    cond = cond.to(DEV)
    na = int(round(frames / 24 * 40)) * 2; xa = torch.randn(1, na, 32, device=DEV)
    vid = torch.arange(0, Nv); txt = torch.arange(Nv, Nv + Nt); aud = torch.arange(Nv + Nt, Nv + Nt + na)
    L = Nv + Nt + na
    tags = torch.cat([torch.zeros(Nv, dtype=torch.long), cond_tags, torch.full((na,), 2, dtype=torch.long)])
    pos = torch.zeros(L, 3, dtype=torch.float64)
    pos[Nv:Nv + Nt, 0] = torch.arange(Nt, dtype=torch.float64)
    rt = float(Nt); rg, _ = he._frame_position_grid(int(lat.shape[-2]), int(lat.shape[-1]), 2, 2)
    pos[0:Nr, 0] = rt; pos[0:Nr, 1:] = rg; rt += 1.0
    tg, tw = he._frame_position_grid(H // 16, W // 16, 2, 2); rpf = tg.shape[0]; nt = (Nv - Nr) // rpf
    pos[Nr:Nv, 0] = he._temporal_position_grid(nt, rt).repeat_interleave(rpf); pos[Nr:Nv, 1:] = tg.repeat(nt, 1)
    he._fill_audio_positions(pos, slice(Nv + Nt, L), na // 2, rt, tw, 2)
    grid_h, grid_w = H // 32, W // 32
    tile_pg, ov_pg = tile // 32, ov // 32; stride = tile_pg - ov_pg
    rows = torch.arange(Nv - Nr); hh = (rows % rpf) // grid_w; ww = (rows % rpf) % grid_w
    ys = list(range(0, grid_h, stride)); xs = list(range(0, grid_w, stride))
    if ys[-1] + tile_pg < grid_h: ys.append(grid_h - tile_pg)
    if xs[-1] + tile_pg < grid_w: xs.append(grid_w - tile_pg)
    tiles = [(y0, x0, min(y0 + tile_pg, grid_h), min(x0 + tile_pg, grid_w)) for y0 in ys for x0 in xs]
    print("[tiled] %dx%d grid=%dx%d tiles=%d tile_pg=%d ov_pg=%d" % (H, W, grid_h, grid_w, len(tiles), tile_pg, ov_pg), flush=True)
    u = torch.linspace(1.0, 0.0, steps)
    sig = torch.unique_consecutive(12.0 * u / (1 + 11.0 * u)); asig = torch.unique_consecutive(3.0 * u / (1 + 2.0 * u))
    ne = int(sig.shape[0]) - 1
    xt = xv.clone(); xa = xa.clone(); t0 = time.time()
    for i in range(ne):
        _t = float(1.0 - sig[i]); _at = float(1.0 - asig[i])
        row_t = torch.full((L,), _t, dtype=torch.float32); row_t[:Nr] = max(_t, 0.999); row_t[Nv + Nt:] = _at
        uniq_t, ti = torch.unique(row_t, sorted=True, return_inverse=True)
        vo_acc = torch.zeros(1, Nv - Nr, 96, device=DEV); w_acc = torch.zeros(1, Nv - Nr, 1, device=DEV)
        ao_acc = torch.zeros(1, na, 32, device=DEV)
        for (y0, x0, y1, x1) in tiles:
            sel = ((hh >= y0) & (hh < y1) & (ww >= x0) & (ww < x1)).nonzero().flatten()
            if sel.numel() == 0: continue
            vidx_t = torch.cat([vid[:Nr], vid[Nr:][sel]]); nvt = vidx_t.numel()
            pos_t = torch.cat([pos[vid[:Nr]], pos[vid[Nr:][sel]], pos[txt], pos[aud]])
            tags_t = torch.cat([tags[vid[:Nr]], tags[vid[Nr:][sel]], tags[txt], tags[aud]])
            ti_t = torch.cat([ti[vid[:Nr]], ti[vid[Nr:][sel]], ti[txt], ti[aud]])
            xv_t = torch.cat([xv[:, :Nr], xv[:, Nr:][:, sel]], 1)
            # packing indices must be LOCAL to the tile sequence (pos/tags/ti already in local order)
            vid_l = torch.arange(nvt); txt_l = torch.arange(nvt, nvt + Nt); aud_l = torch.arange(nvt + Nt, nvt + Nt + na)
            vo_t, ao_t = fp4_dit_forward(r, xv_t, xa, cond, uniq_t, ti_t, tags_t, pos_t, vid_l, aud_l, txt_l, sm=sm, layer_fn=lf)
            hl = (hh[sel] - y0).float(); wl = (ww[sel] - x0).float()
            wy = torch.clamp(torch.minimum(hl + 1, (y1 - y0) - hl) / ov_pg, max=1.0)
            wx = torch.clamp(torch.minimum(wl + 1, (x1 - x0) - wl) / ov_pg, max=1.0)
            w = (wy * wx).clamp(min=1e-3).view(1, -1, 1).to(DEV)
            vo_acc[:, sel.to(DEV)] += vo_t[:, Nr:] * w; w_acc[:, sel.to(DEV)] += w; ao_acc += ao_t
        vo = torch.cat([torch.zeros(1, Nr, 96, device=DEV), vo_acc / w_acc], 1)
        ao = ao_acc / len(tiles)
        xt = xt + (sig[i] - sig[i + 1]) * vo; xt[:, :Nr] = xv_ref
        xa = xa + (asig[i] - asig[i + 1]) * ao
        print("[tiled] step %d/%d %.0fs" % (i + 1, ne, time.time() - t0), flush=True)
    _tgt = xt[:, Nr:]; B, N2, D = _tgt.shape
    Hh, Ww = H // 16, W // 16; T = N2 // ((Hh // 2) * (Ww // 2))
    y = _tgt.reshape(B, T, Hh // 2, Ww // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, 24, T, Hh, Ww)
    lm = torch.tensor(c["latents_mean"]).view(1, -1, 1, 1, 1).to(y.device); ls = torch.tensor(c["latents_std"]).view(1, -1, 1, 1, 1).to(y.device)
    y = y * ls + lm
    del lf, r, sm; import gc; gc.collect(); torch.cuda.empty_cache()
    v = v.to(DEV); dec = he._decode_safe(v, y); fr = dec[0] if isinstance(dec, (tuple, list)) else dec
    fr = fr[0].permute(1, 2, 3, 0).float().cpu().numpy()
    fr = fr * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
    fr = np.clip(fr, 0, 1); os.makedirs(os.path.dirname(out), exist_ok=True)
    iio.imwrite(out, (fr * 255).astype("uint8"), fps=24)
    print("[tiled] RESULT OK %s frames=%s total=%.0fs" % (out, fr.shape, time.time() - t0), flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=r"D:\ai\output\h3video_fp4.mp4")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tile", type=int, default=0, help="MultiDiffusion spatial tile canvas px (0=off)")
    ap.add_argument("--ov", type=int, default=256, help="tile overlap px")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    global _cache
    _cache = build_cache(a.steps)
    print("[video-fp4] resident=%d steps=%d frames=%d size=%d tile=%d" % (RESIDENT, a.steps, a.frames, a.size, a.tile), flush=True)
    if a.tile > 0:
        run_ref2va_tiled(a.ref, a.out, steps=a.steps, frames=a.frames, H=a.size, W=a.size, tile=a.tile, ov=a.ov, prompt=a.prompt, seed=a.seed)
    else:
        he.run_ref2va(a.ref, a.out, steps=a.steps, frames=a.frames, H=a.size, W=a.size, prompt=a.prompt, verbose=True)
    print("[video-fp4] RESULT: OK -> %s" % a.out, flush=True)

if __name__ == "__main__":
    main()