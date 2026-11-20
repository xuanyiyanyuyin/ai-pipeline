# -*- coding: utf-8 -*-
"""H3 常驻引擎 API 服务（api化/引擎化）：FP4 图像权重驻留 + cond 缓存 + 串行 job 队列。
端点（async job）:
  POST /v1/images/generations {prompt,size,seed}
  POST /v1/images/edits       {prompt,images:[base64...],size,seed}   # 多参考图有序
  POST /v1/video/generations  {prompt,image:base64,frames,steps,turbo,size}
  GET  /v1/jobs/<id>
图像在进程内用驻留 reader 渲染（免每job加载权重）；cond 缓存键=sha(refs+prompt) 免重复 VL 冷编码；
video 走子进程 h3_video_job（独立权重，避免与图像同驻留超显存）。单 GPU 串行队列防争抢。
启动: D:\\ai\\python\\python.exe h3_engine_server.py  (port 8340)
"""
import os, sys, json, time, uuid, base64, hashlib, threading, subprocess, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, r"D:\ai")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import h3image_engine_fp4 as F4
import h3_engine as he
from safetensors.torch import load_file

AI = r"D:\ai"
PY = sys.executable
PORT = int(os.environ.get("H3_ENGINE_PORT", "8340"))
CACHE_DIR = os.path.join(AI, "cache", "h3cond")
OUT_DIR = os.path.join(AI, "output", "h3api")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("[h3engine] loading resident FP4 image weights …", flush=True)
READER = F4.FP4Reader()
SM = READER.small_gpu()
CFG = json.load(open(F4.base.PDD_CFG, encoding="utf-8"))
SIG0 = torch.tensor(CFG["sigmas"], dtype=torch.float64)
TS = torch.tensor([1.0 - float(s) for s in SIG0[:-1]], dtype=torch.float32)
SIGMAS = torch.cat([1.0 - TS, torch.zeros(1)]).to(torch.float64)
CACHE = load_file(F4.CACHE_FILE)
PDD = load_file(F4.base.PDD_FILE)
PDD_W = PDD["video_out.weight"].to(F4.DEV).float()
PDD_B = PDD["video_out.bias"].to(F4.DEV).float()
_LF = {}
def layer_fn(res):
    if res not in _LF:
        _LF[res] = F4.make_layer_fn(READER, res)
    return _LF[res]
print("[h3engine] resident ready", flush=True)

JOBS = {}
QUEUE = []
LOCK = threading.Lock()


def render_image(cond_path, size, seed, out):
    H = W = size
    lh, lw = H // F4.base.VAE_RATIO, W // F4.base.VAE_RATIO
    cond = F4.base.load_cond(cond_path, "official")
    n_text = cond.shape[1]
    pos, tags, image_idx, text_idx, n_img = F4.base.build_layout(he, n_text, lh, lw)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat0 = torch.randn((1, F4.base.LATENT_CH, 1, lh, lw), generator=g, dtype=torch.float32)
    x = lat0.reshape(1, F4.base.LATENT_CH, 1, 1, lh // 2, 2, lw // 2, 2) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape(1, n_img, F4.base.LATENT_CH * F4.base.PATCH[1] * F4.base.PATCH[2]).contiguous().to(F4.DEV)
    L = n_img + n_text
    act = 3.6e-4 * L + 0.1
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    res = min(32, max(8, int((free - act - 1.5) / 0.208)))
    lf = layer_fn(res)
    for i in range(int(CFG["num_inference_steps"])):
        row_t = torch.full((L,), float(TS[i]), dtype=torch.float32)
        uniq_t, ti = torch.unique(row_t, sorted=True, return_inverse=True)
        v = F4.fp4_dit_forward(READER, x, cond, i, ti, tags, pos, image_idx, text_idx,
                               SM, lf, PDD_W[i], PDD_B[i], CACHE)
        x = x + float(SIGMAS[i] - SIGMAS[i + 1]) * v.float()
        if not bool(torch.isfinite(x).all()):
            raise RuntimeError("non-finite")
    y = x[0].float().cpu().reshape(-1, lh // 2, lw // 2, F4.base.LATENT_CH, 2, 2) \
            .permute(0, 3, 1, 4, 2, 5).reshape(1, F4.base.LATENT_CH, lh, lw).contiguous()
    vcfg = json.load(open(F4.base.VAE_CFG, encoding="utf-8"))
    lm = torch.tensor(vcfg["latents_mean"]).view(1, -1, 1, 1)
    ls = torch.tensor(vcfg["latents_std"]).view(1, -1, 1, 1)
    y = y * ls + lm
    sd, vc = F4.base.load_vae()
    with torch.no_grad():
        image = F4.base.vae_decode(y.to(F4.DEV), sd, vc)
    F4.base.to_png(image, out)
    del sd, vc
    torch.cuda.empty_cache()


def _encode(refs, prompt, tag):
    h = hashlib.sha256()
    h.update((tag + prompt).encode())
    for r in refs:
        h.update(open(r, "rb").read())   # 按内容哈希，跨job同图同prompt才命中
    key = h.hexdigest()[:16]
    cp = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.isfile(cp):
        return cp, True
    if refs:
        cmd = [PY, os.path.join(AI, "h3_encode.py"), "--ref", *refs, "--prompt", prompt, "--out", cp]
    else:
        cmd = [PY, os.path.join(AI, "h3image_encode.py"), "--prompt", prompt, "--out", cp]
    r = subprocess.run(cmd, capture_output=True, cwd=AI)
    if r.returncode != 0 or not os.path.isfile(cp):
        raise RuntimeError("encode failed: " + r.stdout.decode("utf-8", errors="replace")[-800:])
    return cp, False


def _worker():
    while True:
        with LOCK:
            jid = QUEUE.pop(0) if QUEUE else None
        if not jid:
            time.sleep(0.3)
            continue
        j = JOBS[jid]
        j["status"] = "running"
        t0 = time.time()
        try:
            sp = j["spec"]
            if sp["kind"] == "video":
                ref = os.path.join(OUT_DIR, jid + "_ref.png")
                open(ref, "wb").write(base64.b64decode(sp["image"]))
                out = os.path.join(OUT_DIR, jid + ".mp4")
                r = subprocess.run([PY, os.path.join(AI, "h3_video_job.py"), "--ref", ref, "--out", out,
                                    "--steps", str(sp.get("steps", 8)), "--turbo", str(sp.get("turbo", 8)),
                                    "--frames", str(sp.get("frames", 124)), "--size", str(sp.get("size", 512))],
                                   capture_output=True, cwd=AI)
                ok = r.returncode == 0 and os.path.isfile(out)
                j.update(status="done" if ok else "error", out="/out/" + jid + ".mp4" if ok else None,
                         error=None if ok else r.stdout.decode("utf-8", errors="replace")[-800:])
            else:
                refs = []
                for i, b in enumerate(sp.get("images") or []):
                    rp = os.path.join(OUT_DIR, jid + "_ref%d.png" % i)
                    open(rp, "wb").write(base64.b64decode(b))
                    refs.append(rp)
                cp, hit = _encode(refs, sp["prompt"], "edit" if refs else "t2i")
                j["cond_cache_hit"] = hit
                out = os.path.join(OUT_DIR, jid + ".png")
                render_image(cp, int(sp.get("size", 512)), int(sp.get("seed", 42)), out)
                j.update(status="done", out="/out/" + jid + ".png")
        except Exception as e:
            j.update(status="error", error=str(e))
        j["elapsed"] = round(time.time() - t0, 1)


threading.Thread(target=_worker, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def _json(self, d, code=200):
        b = json.dumps(d).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/jobs/"):
            j = JOBS.get(self.path.rsplit("/", 1)[-1])
            return self._json(j or {"error": "unknown job"}, 200 if j else 404)
        if self.path.startswith("/out/"):
            fn = self.path.split("/out/", 1)[1]
            p = os.path.join(OUT_DIR, fn)
            if not os.path.isfile(p):
                return self._json({"error": "not found"}, 404)
            ct = "image/png" if fn.endswith(".png") else "video/mp4"
            b = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        return self._json({"ok": True, "service": "h3-engine", "queue": len(QUEUE)})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 64 * 1024 * 1024:
            return self._json({"error": "too large"}, 413)
        sp = json.loads(self.rfile.read(n).decode() or "{}")
        if self.path == "/v1/images/generations":
            sp["kind"] = "image"
        elif self.path == "/v1/images/edits":
            sp["kind"] = "image"
        elif self.path == "/v1/video/generations":
            sp["kind"] = "video"
        else:
            return self._json({"error": "unknown endpoint"}, 404)
        jid = "e" + uuid.uuid4().hex[:10]
        JOBS[jid] = {"id": jid, "status": "queued", "kind": sp["kind"], "spec": sp, "out": None, "error": None}
        with LOCK:
            QUEUE.append(jid)
        return self._json({"ok": True, "job_id": jid})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("[h3engine] listening :%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()