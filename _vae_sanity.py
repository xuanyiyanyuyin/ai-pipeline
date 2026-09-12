import sys, os, torch
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE']='1'
VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'
latent = torch.load(r'D:\ai\output\ab_test\lat_1024_512x512_f9.pt', weights_only=True)
print(f'latent: {tuple(latent.shape)} dtype={latent.dtype} std={latent.float().std().item():.4f} mean={latent.float().mean().item():.4f}')
print(f'min={latent.float().min().item():.3f} max={latent.float().max().item():.3f}')

from ltx_core.loader import SingleGPUModelBuilder
from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, video_decoder_sd_ops_for_checkpoint
from ltx_core.model.video_vae.diffusion_tiling import DiffVAEMode
from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op
builder = SingleGPUModelBuilder(
    model_class_configurator=VideoDecoderConfigurator,
    model_path=VAE_FILE,
    model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE_FILE),
    module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER),),
)
decoder = builder.build(dtype=torch.bfloat16)
# 检查 per_channel_statistics (VAE 的 latent 归一化协议)
for name, buf in decoder.named_buffers():
    if 'per_channel' in name or 'mean' in name or 'statistics' in name:
        print(f'{name}: shape={tuple(buf.shape)} val={buf.flatten()[:4].tolist()}')
# 单帧 decode
lat0 = latent[:, :, 0:1, :, :].to('cuda').bfloat16()
with torch.inference_mode():
    frames = next(decoder.decode_video(lat0, tiling_config=None))
print(f'单帧 decode: {tuple(frames.shape)} dtype={frames.dtype} range=[{frames.float().min():.3f},{frames.float().max():.3f}]')
# 保存单帧 png
import numpy as np
from PIL import Image
np_img = (frames[0].float().cpu().numpy()*255).clip(0,255).astype('uint8')
Image.fromarray(np_img).save(r'D:\ai\output\ab_test\sanity_frame0.png')
print('单帧已存 sanity_frame0.png')
