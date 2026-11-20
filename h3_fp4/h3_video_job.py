# -*- coding: utf-8 -*-
import os, sys, argparse
os.environ.setdefault('H3_COND_PT', r'D:\ai\h3_cond_apple.pt')
sys.path.insert(0, r'D:\ai')
import h3_engine_fp4 as V
import h3_engine as he
ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=8); ap.add_argument('--turbo', type=int, default=8)
ap.add_argument('--frames', type=int, default=124); ap.add_argument('--size', type=int, default=512)
ap.add_argument('--cond', default=r'D:\ai\h3_cond_apple.pt')
a = ap.parse_args()
os.environ['H3_COND_PT'] = a.cond
V._cache = V.build_cache(a.steps)
he.run_ref2va(ref=a.ref, out=a.out, steps=a.steps, frames=a.frames, H=a.size, W=a.size,
              turbo=(str(a.turbo) if a.turbo else "none"))
print('[h3video] RESULT: OK', flush=True)