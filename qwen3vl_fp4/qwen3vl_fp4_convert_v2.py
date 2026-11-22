# -*- coding: utf-8 -*-
"""Qwen3-VL 文本编码器 -> 定制 FP4（k1 + k/v k2），面向“只读第 50 层”的 H3 条件编码。

策略（实测定稿，2026-11-22）：
- 只量化 LM layers 0..49；51..63 丢弃（读出点 hidden_states[50]，后面永不执行）
- lm_head 丢弃（不生成 token）
- q_proj/o_proj/gate/up/down: k1
- k_proj/v_proj: k2（残差共用同一个全局 g，运行时两次 GEMM 求和；8:64 GQA 共享，误差放大敏感）
- 其余（embed、q/k norm、各项 norm、视觉塔 visual.*、merger）：BF16 原样拷贝
输出目录结构：
  out/bf16/model-XXXXX.safetensors      # 非量化张量（BF16 原样）
  out/fp4/model-XXXXX.safetensors       # 量化张量：<key>.packed(uint8) .scale(fp8) .g(fp32) [.r_packed .r_scale]
  out/fp4_manifest.json                 # 策略 + 形状 + 分片映射
  out/config.json                       # num_hidden_layers=50
用法:
  python qwen3vl_fp4_convert_v2.py --src <text_encoder> --dst <out> [--smoke]
"""
import argparse, json, os, re, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

LV = None
MID = None
DEV = "cuda"
E2M1, FP8 = 6.0, 448.0

TARGET_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$"
)
KV = {"k_proj", "v_proj"}


def init_dev():
    global LV, MID
    LV = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=DEV)
    MID = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=DEV)


def quant(W, g=None):
    """W (N,K) float -> packed uint8(N,K//2), scale fp8(N,K//16), g"""
    N, K = W.shape
    Wf = W.to(DEV).float()
    if g is None:
        g = (FP8 * E2M1) / Wf.abs().max().clamp(min=1e-12)
    else:
        g = g.to(DEV)
    b = Wf.reshape(N, K // 16, 16)
    se8 = ((b.abs().amax(-1, keepdim=True).clamp(min=1e-12) / E2M1) * g).to(torch.float8_e4m3fn)
    se = se8.float().clamp(min=1e-12)
    norm = (b * g / se).clamp(-E2M1, E2M1)
    idx = torch.bucketize(norm.abs(), MID).to(torch.uint8)
    sign = (norm < 0).to(torch.uint8) << 3
    code = (sign | idx).reshape(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed.cpu(), se8.squeeze(-1).reshape(N, K // 16).cpu(), g.reshape(1).float().cpu()


def dequant(packed, se8, g, N, K, dev="cuda"):
    p = packed.to(dev)
    codes = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1).reshape(N, K)
    lv = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
    vals = lv[(codes & 0x07).long()] * torch.where((codes & 0x08) > 0, -1.0, 1.0)
    se = se8.to(dev).float().reshape(N, K // 16, 1)
    return (vals.reshape(N, K // 16, 16) * se / g.to(dev)).reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\ai\models\MiniMax-H3\Ref2VA\text_encoder")
    ap.add_argument("--dst", default=r"D:\ai\models\Qwen3VL-FP4-v2")
    ap.add_argument("--smoke", action="store_true", help="只处理 layer 0 做格式冒烟")
    ap.add_argument("--kv2", action="store_true", help="k/v 用 k2 残差（默认全 k1）")
    a = ap.parse_args()
    init_dev()
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg.setdefault("text_config", {})["num_hidden_layers"] = 50
    os.makedirs(os.path.join(a.dst, "bf16"), exist_ok=True)
    os.makedirs(os.path.join(a.dst, "fp4"), exist_ok=True)

    bf16_buf, fp4_buf = {}, {}
    shard = {"b": 0, "f": 0}
    def flush_if(force=False):
        for buf, key, sub in ((bf16_buf, "b", "bf16"), (fp4_buf, "f", "fp4")):
            tot = sum(x.numel() * x.element_size() for x in buf.values())
            if buf and (force or tot > 1.2 * 2**30):
                shard[key] += 1
                save_file(dict(buf), os.path.join(a.dst, sub, "model-%05d.safetensors" % shard[key]))
                buf.clear()
    manifest = {"format": "nvfp4_custom_v2", "policy": {"k1": ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                                         "k2": ["k_proj", "v_proj"]},
                "layers": [0, 49], "dropped": ["lm_head", "layers 50..63"], "tensors": {}}
    copied = quantized = dropped = 0
    t0 = time.time()
    cur = None; st = None
    keys = sorted(wm)
    for k in keys:
        m = TARGET_RE.match(k)
        lm = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
        li = int(lm.group(1)) if lm else None
        if a.smoke and (li is None or li != 0) and "embed_tokens" not in k:
            continue
        if k == "lm_head.weight" or (li is not None and li > 49) or (k.endswith("lm_head.weight")):
            dropped += 1
            continue
        keep_layer = (li is None) or (li <= 49)
        f = wm[k]
        if f != cur:
            if st is not None: st.__exit__(None, None, None)
            st = safe_open(os.path.join(a.src, f), framework="pt", device="cpu"); cur = f
        t = st.get_tensor(k).clone()
        if m and keep_layer:
            name = m.group(2).split(".")[-1]  # q_proj 等
            N, K = t.shape
            if a.kv2 and name in ("k_proj", "v_proj"):
                p1, s1, g1 = quant(t)
                W1 = dequant(p1, s1, g1, N, K)
                p2, s2, _ = quant((t.to(DEV) - W1).cpu(), g=g1)
                fp4_buf[k + ".packed"] = p1
                fp4_buf[k + ".scale"] = s1
                fp4_buf[k + ".g"] = g1
                fp4_buf[k + ".r_packed"] = p2
                fp4_buf[k + ".r_scale"] = s2
                manifest["tensors"][k] = {"mode": "k2", "shape": [N, K]}
            else:
                p1, s1, g1 = quant(t)
                fp4_buf[k + ".packed"] = p1
                fp4_buf[k + ".scale"] = s1
                fp4_buf[k + ".g"] = g1
                manifest["tensors"][k] = {"mode": "k1", "shape": [N, K]}
            quantized += 1
        else:
            bf16_buf[k] = t
            copied += 1
        del t
        flush_if()
        if (quantized + copied) % 10 == 0:
            torch.cuda.empty_cache()
        if (quantized + copied) % 20 == 0:
            print("[conv] q=%d copy=%d drop=%d layer=%s %.0fs gpu=%.1fGB" % (quantized, copied, dropped, li, time.time() - t0, torch.cuda.memory_allocated()/2**30), flush=True)
    if st is not None: st.__exit__(None, None, None)

    flush_if(force=True)
    json.dump(manifest, open(os.path.join(a.dst, "fp4_manifest.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
    print("[conv] done quantized=%d copied=%d dropped=%d  %.1fs" % (quantized, copied, dropped, time.time() - t0))


if __name__ == "__main__":
    main()
