"""
显存批次拉满探针 (20261024)
与 train_v22_img2img.py v3 完全相同的模型结构, batch 逐级拉到 OOM,
记录每级: 峰值分配 / 保留 / 全卡空闲 / 前向+反向耗时
"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn, torch.nn.functional as F

IMG, N_POS, BINS, EMB = 16, 768, 16, 256

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
        q = feat + h + self.pos_embed.unsqueeze(0)
        return q @ self.bin_embed.t()

dev = "cuda"
free_b, tot_b = torch.cuda.mem_get_info()
print(f"[环境] 专用显存 空闲{free_b/2**30:.2f}GB / 总{tot_b/2**30:.2f}GB (其它应用占用 {((tot_b-free_b)/2**30):.1f}GB)")
ve, gen = VisualEmbed().to(dev), I2IGen().to(dev)
params = list(ve.parameters()) + list(gen.parameters())
opt = torch.optim.Adam(params, lr=1e-4)

print(f"{'batch':>6} | {'峰值分配':>8} | {'保留':>7} | {'全卡空闲':>8} | {'前向+反向':>9} | loss")
for b in [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rgb = torch.rand(b, 768, device=dev)
        dgb = torch.randint(0, BINS, (b, 768), device=dev)
        tgt = torch.randint(0, BINS, (b, 768), device=dev)
        attr = torch.randn(b, 256, device=dev)
        loss = None
        for _ in range(3):  # 3次取稳定峰值
            t0 = time.time()
            logits = gen(dgb, ve(rgb), F.normalize(attr, dim=-1))
            loss = F.cross_entropy(logits.reshape(-1, BINS), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 1 * 1000
        peak = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.memory_reserved() / 2**30
        free_now, _ = torch.cuda.mem_get_info()
        print(f"{b:>6} | {peak:>7.2f}GB | {resv:>6.2f}GB | {free_now/2**30:>7.2f}GB | {dt:>7.0f}ms | {loss.item():.3f}")
        del rgb, dgb, tgt, attr, logits, loss
    except torch.cuda.OutOfMemoryError as e:
        print(f"{b:>6} | OOM — {str(e)[:70]}")
        torch.cuda.empty_cache()
        break
print("[探针完毕]")
