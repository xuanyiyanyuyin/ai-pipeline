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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=r"D:\ai\output\h3video_fp4.mp4")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="")
    a = ap.parse_args()
    global _cache
    _cache = build_cache(a.steps)
    print("[video-fp4] resident=%d steps=%d frames=%d size=%d" % (RESIDENT, a.steps, a.frames, a.size), flush=True)
    he.run_ref2va(a.ref, a.out, steps=a.steps, frames=a.frames, H=a.size, W=a.size, prompt=a.prompt, verbose=True)
    print("[video-fp4] RESULT: OK -> %s" % a.out, flush=True)

if __name__ == "__main__":
    main()