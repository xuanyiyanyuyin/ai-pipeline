import sys, time, torch
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
import faulthandler; faulthandler.enable()
import numpy as np
t0 = time.perf_counter()

# 读现成 embeds (prompt: a cute orange cat — 与参考图语义一致)
ck = torch.load(r'D:\ai\_cached_flux2_embeds_test.pt', map_location='cpu', weights_only=True)
prompt_embeds = ck['embeds'].to('cuda').bfloat16()
print(f'embeds {tuple(prompt_embeds.shape)}', flush=True)

from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    low_cpu_mem_usage=True, text_encoder=None)
print(f'pipe加载 {time.perf_counter()-t0:.0f}s', flush=True)
r = f2._patch_f2(pipe.transformer, resident_d=0)
print(f'patch 常驻{r}', flush=True)

text_ids = pipe._prepare_text_ids(prompt_embeds).to('cuda')
nch = pipe.transformer.config.in_channels // 4
W, H = 512, 512
latents, latent_ids = pipe.prepare_latents(1, nch, W, H, prompt_embeds.dtype, 'cuda',
    torch.Generator('cuda').manual_seed(1), None)

# ── 图生图: 编码参考图 (VAE 需在 GPU) ──
pipe.vae.to('cuda')
from PIL import Image
ref = Image.open(r'D:\ai\output\images\flux2_full_test.png').convert('RGB')
im = pipe.image_processor.preprocess(ref, height=H, width=W, resize_mode="crop")
print(f'参考图 preprocess: {tuple(im.shape)}', flush=True)
img_latents, img_latent_ids = pipe.prepare_image_latents(
    [im], batch_size=1, generator=None, device='cuda', dtype=pipe.vae.dtype)
print(f'图生图 img_latents {tuple(img_latents.shape)}  ids {tuple(img_latent_ids.shape)}', flush=True)

# scheduler
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
img_seq = (W//16)*(H//16)
NSTEPS = 4
mu = compute_empirical_mu(image_seq_len=img_seq, num_steps=NSTEPS)
ts, _ = retrieve_timesteps(pipe.scheduler, NSTEPS, 'cuda',
    sigmas=np.linspace(1.0, 1.0/NSTEPS, NSTEPS), mu=mu)
guidance = torch.full([1], 4.0, device='cuda', dtype=torch.float32).expand(latents.shape[0])

for i, t in enumerate(ts):
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    latent_model_input = torch.cat([latents, img_latents], dim=1).to(pipe.transformer.dtype)
    latent_image_ids = torch.cat([latent_ids, img_latent_ids], dim=1)
    npred = pipe.transformer(hidden_states=latent_model_input, timestep=timestep/1000, guidance=guidance,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids,
        joint_attention_kwargs=None, return_dict=False)[0]
    npred = npred[:, :latents.size(1)]
    latents = pipe.scheduler.step(npred, t, latents, return_dict=False)[0]
    del npred, latent_model_input
    torch.cuda.empty_cache()
    print(f'step{i} {time.perf_counter()-t0:.0f}s', flush=True)

# decode
pipe.transformer.to('cpu'); torch.cuda.empty_cache()
pipe.vae.to('cuda')
if hasattr(pipe.vae, 'enable_tiling'): pipe.vae.enable_tiling()
with torch.no_grad():
    lat_u = pipe._unpack_latents_with_ids(latents, latent_ids)
    bn_mean = pipe.vae.bn.running_mean.view(1,-1,1,1).to(lat_u.device, lat_u.dtype)
    bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1,-1,1,1)+pipe.vae.config.batch_norm_eps).to(lat_u.device, lat_u.dtype)
    lat_u = lat_u * bn_std + bn_mean
    lat_u = pipe._unpatchify_latents(lat_u)
    img = pipe.vae.decode(lat_u, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type='pil')[0]
fn = r'D:\ai\output\images\flux2_i2v_test.png'
img.save(fn)
print(f'I2V_DONE {time.perf_counter()-t0:.0f}s -> {fn}', flush=True)
