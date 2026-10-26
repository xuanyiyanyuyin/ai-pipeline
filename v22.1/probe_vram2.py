"""
显存对账探针 v2 (20261024) — 三方口径对账 + 中位数计时
回答: 12GB卡为何 batch1536(峰值分配仅6.94GB)就报0空闲? 计时方法是否可靠?
口径:
  A. torch: max_memory_allocated / memory_reserved (分配器视角)
  B. nvidia-smi: memory.used/free (驱动视角, 探针持张量时快照)
  C. Windows计数器: 本pid Dedicated Usage (系统每进程视角)
变量: PYTORCH_CUDA_ALLOC_CONF 环境变量决定 expandable_segments 开/关
"""
import os, sys, time, subprocess
import torch, torch.nn as nn, torch.nn.functional as F

IMG, N_POS, BINS, EMB = 16, 768, 16, 256
PID = os.getpid()
MODE = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<default>") or "<default>"

class VisualEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, EMB, bias=False)
        nn.init.orthogonal_(self.proj.weight)
    def forward(self, img):
        return F.normalize(self.proj(img), dim=-1)

class I2IGen(nn.Module):
    def __init__(self):
        super().__init__()
        be = torch.empty(BINS, EMB); nn.init.orthogonal_(be)
        self.register_buffer("bin_embed", be)
        self.register_buffer("avg3", self._neighbor_avg(IMG, 1))
        self.register_buffer("avg5", self._neighbor_avg(IMG, 2))
        self.mix_in = nn.Linear(EMB, EMB, bias=False)
        self.mix_c3 = nn.Linear(EMB, EMB, bias=False)
        self.mix_c5 = nn.Linear(EMB, EMB, bias=False)
        for m in (self.mix_in, self.mix_c3, self.mix_c5):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        self.input_proj = nn.Linear(512, EMB, bias=False)
        self.encoder = nn.Sequential(nn.Linear(EMB, EMB), nn.GELU(), nn.Linear(EMB, EMB), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(N_POS, EMB) * 0.1)
        self.gate = nn.Linear(EMB, EMB, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
    @staticmethod
    def _neighbor_avg(img, r):
        n = img * img * 3
        M = torch.zeros(n, n)
        for y in range(img):
            for x in range(img):
                for y2 in range(max(0, y - r), min(img, y + r + 1)):
                    for x2 in range(max(0, x - r), min(img, x + r + 1)):
                        for c in range(3):
                            M[(y * img + x) * 3 + c, (y2 * img + x2) * 3 + c] = 1.0
        return M / M.sum(dim=1, keepdim=True)
    def forward(self, deg_bins, img_emb, vis_attr):
        q_in = self.bin_embed[deg_bins]
        ctx3 = self.avg3 @ q_in
        ctx5 = self.avg5 @ q_in
        feat = self.mix_in(q_in) + self.mix_c3(ctx3) + self.mix_c5(ctx5)
        x = self.input_proj(torch.cat([img_emb, vis_attr], dim=-1))
        h = self.encoder(x).unsqueeze(1)
        hp = h * torch.tanh(self.gate(self.pos_embed)).unsqueeze(0)
        q = feat + hp + self.pos_embed.unsqueeze(0)
        return q @ self.bin_embed.t()

def nvidia_snapshot():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        u, f = [int(x) for x in r.stdout.strip().split(",")]
        return u, f
    except Exception:
        return -1, -1

def win_dedicated_mb(pid):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        f"(Get-Counter '\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
        ".CounterSamples | ForEach-Object { [int]($_.CookedValue/1MB) }"],
        capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split() if x.strip().isdigit())
    except Exception:
        return -1

dev = "cuda"
free_b, tot_b = torch.cuda.mem_get_info()
print(f"[环境] ALLOC_CONF={MODE}")
print(f"[环境] torch口径空闲 {free_b/2**30:.2f}GB / 总 {tot_b/2**30:.2f}GB | pid={PID}")
u, f = nvidia_snapshot()
print(f"[环境] nvidia-smi: used {u}MiB / free {f}MiB")
ve, gen = VisualEmbed().to(dev), I2IGen().to(dev)
params = list(ve.parameters()) + list(gen.parameters())
opt = torch.optim.Adam(params, lr=1e-4)

hdr = f"{'batch':>6} | {'A峰值分配':>9} | {'A保留':>9} | {'B驱动used':>9} | {'B驱动free':>9} | {'C本进程':>8} | {'步时中位':>8}"
print(hdr)
for b in [256, 512, 1024, 1536, 2048, 2560, 3072]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rgb = torch.rand(b, 768, device=dev)
        dgb = torch.randint(0, BINS, (b, 768), device=dev)
        tgt = torch.randint(0, BINS, (b, 768), device=dev)
        attr = torch.randn(b, 256, device=dev)
        times = []
        for _ in range(5):  # 5次取中位, 前2次视为预热不计
            torch.cuda.synchronize(); t0 = time.perf_counter()
            logits = gen(dgb, ve(rgb), F.normalize(attr, dim=-1))
            loss = F.cross_entropy(logits.reshape(-1, BINS), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        med = sorted(times)[2]
        peak = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.memory_reserved() / 2**30
        u, f = nvidia_snapshot()          # 张量仍持有 → 驱动视角峰值
        c = win_dedicated_mb(PID)         # 系统计数器视角
        print(f"{b:>6} | {peak:>8.2f}GB | {resv:>8.2f}GB | {u:>8}MiB | {f:>8}MiB | {c:>7}MB | {med:>7.0f}ms")
        del rgb, dgb, tgt, attr, logits, loss
    except torch.cuda.OutOfMemoryError:
        print(f"{b:>6} | OOM (torch分配器)")
        try: torch.cuda.empty_cache()
        except Exception: pass
        break
    except RuntimeError as e:
        print(f"{b:>6} | RuntimeError: {str(e)[:60]}")
        torch.cuda.empty_cache()
        break
print("[对账完毕]")
