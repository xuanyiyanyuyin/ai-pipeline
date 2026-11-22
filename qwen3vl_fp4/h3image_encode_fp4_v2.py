# -*- coding: utf-8 -*-
"""Qwen3-VL 编码器（定制 FP4 v2：LM 0-49，q/o/gate/up/down k1，k/v k2，视觉塔 BF16）。

用法:
  python h3image_encode_fp4_v2.py --prompt "..." --out cond_fp4.pt [--fp4_dir DIR] [--layer 50]
只读取 hidden_states[layer] 写盘，格式与 h3image_encode.py 一致（hidden_states/input_ids/token_tags/prompt）。
"""
import argparse, glob, json, os, time
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from safetensors.torch import load_file

fp4 = torch.float4_e2m1fn_x2
FP8 = 448.0
E2M1 = 6.0
LV_CPU = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])


def deq(packed, se, g, N, K, dev):
    p = packed.to(dev)
    codes = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).reshape(N, K)
    lv = LV_CPU.to(dev)
    vals = lv[(codes & 0x07).long()] * torch.where((codes & 0x08) > 0, -1.0, 1.0)
    return (vals.reshape(N, K // 16, 16) * se.to(dev).float().reshape(N, K // 16, 1) / g.to(dev).float()).reshape(N, K)


class FP4Linear(torch.nn.Module):
    """权重常驻 CPU；forward 时按需反量化（k2 = 两次 GEMM 权重求和）。"""
    def __init__(self, packed, scale, g, N, K, res=None):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("g", g, persistent=False)
        self.N, self.K = N, K
        self.res = None
        if res is not None:
            self.register_buffer("r_packed", res[0], persistent=False)
            self.register_buffer("r_scale", res[1], persistent=False)
            self.res = True

    def forward(self, x):
        dev = x.device
        W = deq(self.packed, self.scale, self.g, self.N, self.K, dev)
        if self.res:
            W = W + deq(self.r_packed, self.r_scale, self.g, self.N, self.K, dev)
        return torch.nn.functional.linear(x, W.to(x.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp4_dir", default=r"D:\ai\models\Qwen3VL-FP4-v2")
    ap.add_argument("--src", default=r"D:\ai\models\MiniMax-H3\Ref2VA\text_encoder")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--out", default=r"D:\ai\output\cond_fp4_v2.pt")
    ap.add_argument("--layer", type=int, default=50)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.dev)
    t0 = time.time()

    cfg = AutoConfig.from_pretrained(a.fp4_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(cfg)
    # BF16 部分（embed/norm/视觉塔/merger）
    bd = {}
    for f in glob.glob(os.path.join(a.fp4_dir, "bf16", "*.safetensors")):
        bd.update(load_file(f))
    missing, unexpected = model.load_state_dict(bd, strict=False, assign=True)
    print("[enc] bf16 loaded %d tensors; missing=%d unexpected=%d" % (len(bd), len(missing), len(unexpected)), flush=True)
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    model.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(cfg.text_config)
    # FP4 目标 Linear
    fp = {}
    for f in glob.glob(os.path.join(a.fp4_dir, "fp4", "*.safetensors")):
        fp.update(load_file(f))
    man = json.load(open(os.path.join(a.fp4_dir, "fp4_manifest.json")))["tensors"]
    n_rep = 0
    for key, info in man.items():
        mod = model.get_submodule(key[:-len(".weight")])
        N, K = info["shape"]
        res = (fp[key + ".r_packed"], fp[key + ".r_scale"]) if info["mode"] == "k2" else None
        lin = FP4Linear(fp[key + ".packed"], fp[key + ".scale"], fp[key + ".g"], N, K, res)
        parent = model.get_submodule(key[:-len(".weight")].rsplit(".", 1)[0])
        setattr(parent, key[:-len(".weight")].rsplit(".", 1)[1], lin)
        n_rep += 1
    print("[enc] replaced %d Linear with FP4Linear" % n_rep, flush=True)

    # 逐层上卡/下卡
    layers = model.model.language_model.layers
    def pre_hook(mod, args):
        mod.to(dev)
    def post_hook(mod, args, out):
        mod.to("cpu")
        torch.cuda.empty_cache()
        return out
    for lyr in layers:
        lyr.register_forward_pre_hook(pre_hook)
        lyr.register_forward_hook(post_hook)

    proc = AutoProcessor.from_pretrained(a.src, trust_remote_code=True)
    tk = proc.tokenizer
    prompt = a.prompt
    if a.prompt_file:
        prompt = open(a.prompt_file, encoding="utf-8").read().strip()
    if prompt is None:
        prompt = "The image is an overhead, top-down photograph of a rustic farm-to-table feast."
    ids = tk(prompt, add_special_tokens=False)["input_ids"][:1024]
    inp = torch.tensor([ids], device=dev)
    token_tags = torch.ones(inp.shape, dtype=torch.long, device=dev)  # text=1
    model.eval()
    model.model.language_model.embed_tokens.to(dev)
    model.model.language_model.norm.to(dev)
    print("[enc] starting forward, tokens=%d" % len(ids), flush=True)
    with torch.no_grad():
        out = model.model(input_ids=inp, use_cache=False, output_hidden_states=True)
    hs50 = out.hidden_states[a.layer].detach().to(torch.bfloat16).cpu()
    hs49 = out.hidden_states[a.layer-1].detach().to(torch.bfloat16).cpu()
    hs = hs50
    torch.save({"hidden_states": hs50, "hidden_states_49": hs49, "input_ids": inp.cpu(), "token_tags": token_tags.cpu(), "prompt": prompt}, a.out)
    print("[enc] wrote %s hidden=%s  %.1fs" % (a.out, tuple(hs.shape), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
