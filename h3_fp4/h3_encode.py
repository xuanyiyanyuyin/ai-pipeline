# -*- coding: utf-8 -*-
"""H3 文本/参考图编码器（独立子进程）—— Qwen3-VL(62GB) 不进主进程内存。
呈现方式逐行照官方 ref2va: py_h3/diffusers/modular_pipelines/minimax_h3/encoders.py:543 _build_presentation
用法: python h3_encode.py --ref <图> --prompt "<文本>" --out <pt>
"""
import argparse, os, sys, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import torch

H3_ROOT = r"D:\ai\models\MiniMax-H3\Ref2VA"
TEXT_TAG, VIDEO_TAG = 1, 0                 # modular_pipeline.py:24-26（video=0 text=1 audio=2）

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, nargs="+", help="参考图路径，可多张（顺序=官方 edit 的 reference 顺序）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=50)
    ap.add_argument("--offload", default=r"D:\ai\models\MiniMax-H3\_offload")
    a = ap.parse_args()
    os.makedirs(a.offload, exist_ok=True)

    sys.path.insert(0, r"D:\ai\py_h3")          # 隔离的 diffusers 0.41.0
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    t0 = time.time()
    print("[h3_encode] 加载 processor …", flush=True)
    proc = AutoProcessor.from_pretrained(os.path.join(H3_ROOT, "processor"), trust_remote_code=True)
    tk = proc.tokenizer
    print("[h3_encode] 加载 Qwen3-VL（device_map=auto + 磁盘 offload；62GB 权重不整份进内存）…", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(H3_ROOT, "text_encoder"),
        dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=a.offload,          # accelerate 把放不下的权重落到磁盘
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print("[h3_encode] 加载耗时 %.1f s" % (time.time() - t0), flush=True)

    # 1. 参考图的视觉特征（官方 image_processor，与 VAE 看到的是同一张图）；多张按序 concat
    images = [Image.open(r).convert("RGB") for r in a.ref]
    vision = proc.image_processor(images=images, return_tensors="pt")
    merge = proc.image_processor.merge_size ** 2
    n_imgs = [int(g.prod()) // merge for g in vision["image_grid_thw"]]

    # 2. [FIX-1016 PRESENTATION] 官方呈现 = "<Picture i>: " + 视觉块（逐张有序）+ 提示词逐字，
    #    encoders.py:561 文档串明写 "no chat template and no special tokens"。
    #    视觉 token 的 id 直接取 text_encoder/config.json（词表 pad 叫 image_pad(151655)）。
    tcfg = json.load(open(os.path.join(H3_ROOT, "text_encoder", "config.json"), encoding="utf-8"))
    pids = tk(a.prompt, add_special_tokens=False)["input_ids"]
    token_ids, token_tags = [], []
    for i, n_img in enumerate(n_imgs):
        label = tk("<Picture %d>: " % (i + 1), add_special_tokens=False)["input_ids"]
        block = ([tcfg["vision_start_token_id"]]
                 + [tcfg["image_token_id"]] * n_img
                 + [tcfg["vision_end_token_id"]])
        token_ids += list(label) + block
        token_tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
    token_ids += list(pids)
    token_tags += [TEXT_TAG] * len(pids)
    print("[h3_encode] 呈现 %d 行 = %d 张参考视觉块 + 提示词 %d（视觉块标 tag=0）" % (
        len(token_ids), len(n_imgs), len(pids)), flush=True)

    ids = torch.tensor([token_ids], dtype=torch.long)
    mm = torch.tensor(proc.create_mm_token_type_ids([token_ids]), dtype=torch.long)
    dev = model.device
    vkw = {}
    for name, value in vision.items():
        vkw[name] = value.to(dev, model.dtype) if name.startswith("pixel_") else value.to(dev)

    # 3. [FIX-1016 COND-LAYER] 官方读 text_encoder.model 的 hidden_states[50]，不走 lm_head
    #    （encoders.py:63-69 专门警告末层是 post-norm、不是 H3 要的条件；旧写法 [-2] 取到第 63 层）。
    #    直接调子模块会绕过 accelerate 的卸载钩子 ⇒ 先把钩子手工点一次（encoders.py:88-90）。
    hook = getattr(model, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(model)
    with torch.no_grad():
        out = model.model(
            input_ids=ids.to(dev),
            attention_mask=torch.ones_like(ids).to(dev),
            mm_token_type_ids=mm.to(dev),
            use_cache=False,
            output_hidden_states=True,
            **vkw,
        )
    if hook is not None and hasattr(hook, "post_forward"):
        try:
            hook.post_forward(model, {})
        except Exception:
            pass
    hs = out.hidden_states[a.layer].detach().to(torch.bfloat16).cpu()
    torch.save({"hidden_states": hs, "input_ids": ids,
                "token_tags": torch.tensor(token_tags, dtype=torch.long)}, a.out)
    print("[h3_encode] 写出 %s  hidden=%s  层 %d  %.1fs" % (a.out, tuple(hs.shape), a.layer, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
