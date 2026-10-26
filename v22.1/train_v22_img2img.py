"""
═══════════════════════════════════════════════════════════════
V22.1 图生图 M1 — 16×16 量化CE生成头 (白盒验证级)  20261024
═══════════════════════════════════════════════════════════════

设计 (方案已批准, 见 D:/ai/V22.1_图生图M1_记录):
  v1缺陷: (h+pos[p])·bin[k] 中输入h对768位置相同 → 只能表达全局色板, 卡44%。
  v2: 退化图逐位查bin_embed(tied两端共用), 起点≈抄输入; 遮补生效, 但
      bin_embed两头都可训(输入查表/输出分类互抢) + 无局部上下文(去噪瞎猜)。
  v3(当前): ① bin_embed 冻结为固定靶 — 镜像char_embed"冻结固定靶"哲学,
       输入查表/输出分类两端共用同一本冻结视觉词典(正交, 无余弦混淆)
     ② 3×3+5×5 同通道邻域均值(固定权重白盒平滑先验) → mix Linear 混入输入特征,
       σ=0.3噪声经5×5均值σ降到0.06(小于1格宽) → 去噪可解
     ③ 余弦退火 lr→0.1×, 1000轮
  结构: 退化图逐位量化bin → 查冻结bin_embed = q_in
    ctx3/ctx5 = 邻域均值(q_in) → feat = mix(q_in, ctx3, ctx5)
    全图+P3Vis(256D) → input_proj(512→256) → encoder → h(全局身份)
    v4: hp = h ⊙ tanh(W_g·pos_p) 逐位身份门控 — 解锁上色/遮补深处的逐位调色记忆
    logits[p,k] = (feat[p] + hp[p] + pos[p]) · bin_embed[k] → CE(16阶)
  数据: 退化造对 — T1加噪(σ0.1/0.2/0.3) T2遮块(4/6/8px) T3灰度化(上色)
  精度: FP32全程 — 球面归一化是定义要求(V22.1训练难度量化 第七节)
  M1 刻意不走 Gate/P7/ABCDEF — 先最小闭环验证生成端, 接入完整栈归 M2
  显存: 分配上限默认9.5GB硬闸(超限直接OOM, 不静默溢出共享显存)

用法:
  python train_v22_img2img.py --epochs 500     # 正式训练
  python train_v22_img2img.py --smoke          # 冒烟(3轮小数据)
"""
import os
# [v4.1] 不再用 expandable_segments: 探针对账证实它在超额时会静默扩张进共享显存,
#        默认分配器则是响亮 OOM —— 符合"纯显存, 不溢出"的要求
import torch, torch.nn as nn, torch.nn.functional as F
import sys, glob, time, random, argparse
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE, SAVE_DIR, BASE_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3_vision"))
from p3vis import P3Vis

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--n_bins", type=int, default=16, help="每通道量化色阶数")
parser.add_argument("--use_generic", type=int, default=1, help="0=只用带标签小图(打样), 1=加通用图库")
parser.add_argument("--k_labeled", type=int, default=8, help="每张带标签图每轮退化对数")
parser.add_argument("--k_generic", type=int, default=4, help="每张通用图每轮退化对数")
parser.add_argument("--gen_max", type=int, default=300, help="通用图库最多用多少张")
parser.add_argument("--exam_n_generic", type=int, default=30, help="通用图考试集张数")
parser.add_argument("--regen_every", type=int, default=25, help="每N轮重新生成退化对")
parser.add_argument("--exam_every", type=int, default=100, help="每N轮跑一次考试集")
parser.add_argument("--clip_norm", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true", help="冒烟: 3轮+小数据")
args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

LABELED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_data")
GENERIC_DIR = "D:/ai/output/images"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_out")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "logs", f"i2i_{time.strftime('%Y%m%d_%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")
def w(s): log.write(s + "\n"); log.flush(); print(s, flush=True)

w("=" * 70)
w(f"  V22.1 图生图M1 | SEED={SEED} | bins={args.n_bins} | bs={args.batch_size} | lr={args.lr}")
w(f"  epochs={args.epochs} k_labeled={args.k_labeled} k_generic={args.k_generic} gen_max={args.gen_max}")
w("=" * 70)

if args.smoke:
    args.epochs = 3; args.gen_max = 40; args.k_generic = 2
    args.exam_n_generic = 10; args.regen_every = 2; args.exam_every = 1
    w("[SMOKE] 冒烟模式: 3轮 / gen_max=40 / 考试通用图10张")

# ════════════════════════ 数据加载 ════════════════════════
def imread_u8(path):
    """cv2.imread 在 Windows 中文路径下静默失败, 统一走 imdecode"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return img

def imwrite_u8(path, img_bgr):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img_bgr)
    if ok: buf.tofile(path)
    return ok

IMG = 16
def load_rgb16(path):
    """→ (rgb_float[16,16,3] in [0,1], bgr_uint8_for_p3vis) or None"""
    img = imread_u8(path)
    if img is None: return None
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    bgr = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, bgr

labeled_files = sorted(glob.glob(os.path.join(LABELED_DIR, "*.bmp")))
generic_files = sorted(sum([glob.glob(os.path.join(GENERIC_DIR, e)) for e in ("*.jpg", "*.png", "*.jpeg")], []))
_rng_files = np.random.default_rng(SEED)
generic_files = list(np.array(generic_files)[_rng_files.permutation(len(generic_files))][:args.gen_max + args.exam_n_generic])

sources = []  # {name, kind, rgb, bgr}
_bad = 0
for f in labeled_files:
    r = load_rgb16(f)
    if r is None: _bad += 1; continue
    sources.append({"name": os.path.basename(f), "kind": "labeled", "rgb": r[0], "bgr": r[1]})
_generic_pool = []
for f in generic_files:
    r = load_rgb16(f)
    if r is None: _bad += 1; continue
    _generic_pool.append({"name": os.path.basename(f), "kind": "generic", "rgb": r[0], "bgr": r[1]})

# 考试集切分: 末尾5张apple + 1张sun + 末尾N张通用图, 永不参训
_exam_names = {f"apple_{i:02d}.bmp" for i in range(16, 21)} | {"sun_05.bmp"}
exam_labeled = [s for s in sources if s["name"] in _exam_names]
train_labeled = [s for s in sources if s["name"] not in _exam_names]
exam_generic = _generic_pool[-args.exam_n_generic:]
train_generic = _generic_pool[:-args.exam_n_generic]
sources_train = train_labeled + (train_generic if args.use_generic else [])

w(f"[数据] 带标签(train/exam): {len(train_labeled)}/{len(exam_labeled)} | 通用(train/exam): {len(train_generic)}/{len(exam_generic)} | 坏图: {_bad}")
w(f"[数据] 考试集(不参训): {sorted(s['name'] for s in exam_labeled)} + {len(exam_generic)}张通用图")

# ════════════════════════ 退化造对 ════════════════════════
TASKS = ["去噪", "遮补", "上色"]
N_POS = IMG * IMG * 3  # 768个像素位

def degrade(rgb, task, rng):
    """→ (退化rgb, 任务id)"""
    out = rgb.copy()
    if task == 0:    # 去噪
        sigma = float(rng.choice([0.10, 0.20, 0.30]))
        out = np.clip(rgb + rng.normal(0.0, sigma, rgb.shape).astype(np.float32), 0.0, 1.0)
    elif task == 1:  # 遮补
        k = int(rng.choice([4, 6, 8]))
        y = int(rng.integers(0, IMG - k + 1)); x = int(rng.integers(0, IMG - k + 1))
        out[y:y + k, x:x + k] = 0.5
    else:            # 上色: 输入全灰, 目标彩色
        g = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)
        out = np.stack([g, g, g], axis=-1)
    return np.clip(out, 0.0, 1.0), task

def to_bins(rgb):
    """[16,16,3]∈[0,1] → [768] long bin id"""
    return np.clip((rgb.reshape(-1) * args.n_bins).astype(np.int64), 0, args.n_bins - 1)

_p3vis = P3Vis()

def build_pairs(srcs, k_per_img, rng, per_task=False):
    """对每张源图造退化对; P3Vis 特征描述退化后的输入图(引擎读的是条件)
       per_task=True: 每图每任务各造1对(考试集用, 全任务覆盖)"""
    list_rgb, list_degbins, list_bin, list_attr, list_task = [], [], [], [], []
    t0 = time.time()
    for s in srcs:
        tasks = range(3) if per_task else [int(rng.integers(0, 3)) for _ in range(k_per_img)]
        for task in tasks:
            deg, _ = degrade(s["rgb"], task, rng)
            deg_bgr = (deg[:, :, ::-1] * 255.0).astype(np.uint8)
            attr = np.nan_to_num(_p3vis.process_image(deg_bgr).astype(np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)  # [256] 规则特征
            list_rgb.append(deg.reshape(-1))
            list_degbins.append(to_bins(deg))
            list_bin.append(to_bins(s["rgb"]))
            list_attr.append(attr)
            list_task.append(task)
    rgb_t = torch.tensor(np.stack(list_rgb), dtype=torch.float32, device=DEVICE)   # 整集仅~20MB, 直驻显存
    degbin_t = torch.tensor(np.stack(list_degbins), dtype=torch.long, device=DEVICE)
    bin_t = torch.tensor(np.stack(list_bin), dtype=torch.long, device=DEVICE)
    attr_t = torch.tensor(np.stack(list_attr), dtype=torch.float32, device=DEVICE)
    task_t = torch.tensor(list_task, dtype=torch.long, device=DEVICE)
    w(f"  [造对] {len(srcs)}图×{3 if per_task else k_per_img} = {len(list_rgb)}对 | P3Vis+量化 {time.time()-t0:.1f}s")
    return rgb_t, degbin_t, bin_t, attr_t, task_t

# ════════════════════════ 模型 ════════════════════════
class VisualEmbed(nn.Module):
    """16×16像素 → 768 RGB → 256d, 正交init + 球面 (对应 char_embed)"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(768, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
    def forward(self, img):              # img [b,768] in [0,1]
        return F.normalize(self.proj(img), dim=-1)

class I2IGen(nn.Module):
    """M1 生成头 v3: 冻结视觉词典(固定靶) + 双尺度局部上下文 → 768位×16阶CE"""
    def __init__(self, embed_dim=256, n_pos=N_POS, n_bins=args.n_bins, img=IMG):
        super().__init__()
        be = torch.empty(n_bins, embed_dim)
        nn.init.orthogonal_(be)
        self.register_buffer("bin_embed", be)          # 冻结: 输入查表+输出分类两端共用
        self.register_buffer("avg3", self._neighbor_avg(img, 1))  # 3×3 同通道均值(固定)
        self.register_buffer("avg5", self._neighbor_avg(img, 2))  # 5×5 (压σ0.3→0.06)
        self.mix_in = nn.Linear(embed_dim, embed_dim, bias=False)   # 三支路相加 = concat+Linear 的
        self.mix_c3 = nn.Linear(embed_dim, embed_dim, bias=False)   # block-diagonal 折叠(等价, 省峰值)
        self.mix_c5 = nn.Linear(embed_dim, embed_dim, bias=False)
        for m in (self.mix_in, self.mix_c3, self.mix_c5):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 起点接近纯copy
        self.input_proj = nn.Linear(512, embed_dim, bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(n_pos, embed_dim) * 0.1)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)  # v4: 逐位身份门控
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        self.h_scale = nn.Parameter(torch.tensor(2.0))  # v4.1: 球面归一化后的可学习尺度
        nn.init.orthogonal_(self.input_proj.weight)
        for m in self.encoder:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
    @staticmethod
    def _neighbor_avg(img, r):
        """[768,768] 固定均值矩阵: 每像素位=同通道 (2r+1)² 邻域均值"""
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
        q_in = self.bin_embed[deg_bins]                      # [b,768,256] 冻结表查表
        ctx3 = self.avg3 @ q_in                              # 3×3 邻域均值
        ctx5 = self.avg5 @ q_in                              # 5×5
        feat = self.mix_in(q_in) + self.mix_c3(ctx3) + self.mix_c5(ctx5)  # [b,768,256]
        x = self.input_proj(torch.cat([img_emb, vis_attr], dim=-1))
        h = (self.h_scale * F.normalize(self.encoder(x), dim=-1)).unsqueeze(1)  # v4.1: 球面控数值
        hp = h * torch.tanh(torch.clamp(self.gate(self.pos_embed), -2.0, 2.0)).unsqueeze(0)  # v4: 逐位身份, 钳位防放飞
        q = feat + hp + self.pos_embed.unsqueeze(0)
        return q @ self.bin_embed.t()                        # 冻结靶分类 [b,768,16]

visual_embed = VisualEmbed().to(DEVICE)
gen = I2IGen().to(DEVICE)
n_params = sum(p.numel() for m in [visual_embed, gen] for p in m.parameters())
w(f"[模型] VisualEmbed+I2IGen 可训参数: {n_params:,} + 冻结视觉词典{args.n_bins}×256 | 全FP32 | DEVICE={DEVICE}")

# ── 纯显存约束: 本进程分配上限(超限直接OOM报错, 不静默溢出共享显存) ──
if DEVICE.type == "cuda":
    _free_b, _tot_b = torch.cuda.mem_get_info()
    w(f"[显存] 专用显存: 空闲{_free_b/2**30:.1f}GB / 总{_tot_b/2**30:.1f}GB (其余为其它应用占用)")
    _cap_gb = float(os.environ.get("I2I_VRAM_CAP_GB", "9.5"))
    _eff_cap = min(_cap_gb, _free_b / 2**30 - 1.0)  # 尊重当前空闲: 其它应用占着的不算, 预留1GB防溢共享
    _eff_cap = max(0.5, _eff_cap)
    torch.cuda.set_per_process_memory_fraction(min(1.0, _eff_cap / (_tot_b / 2**30)))
    w(f"[显存] 本进程分配上限 {_eff_cap:.1f}GB (用户上限{_cap_gb}GB | 当前空闲{_free_b/2**30:.1f}GB, 预留1GB)")

params = list(visual_embed.parameters()) + list(gen.parameters())
opt = torch.optim.Adam(params, lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)  # v3: 余弦退火

def decode_bins(bins):
    return (bins.float() + 0.5) / args.n_bins

def run_batch(rgb, degbins, target_bin, attr):
    img_emb = visual_embed(rgb)
    logits = gen(degbins, img_emb, F.normalize(attr, dim=-1))     # [b,768,16]
    loss = F.cross_entropy(logits.reshape(-1, args.n_bins), target_bin.reshape(-1))
    with torch.no_grad():
        acc = (logits.argmax(-1) == target_bin).float().mean().item()
    return loss, acc, logits

# ════════════════════════ 考试集 (退化参数固定可复现) ════════════════════════
exam_rng = np.random.default_rng(SEED + 1)
def build_exam():
    return build_pairs(exam_labeled + (exam_generic if args.use_generic else []),
                       None, exam_rng, per_task=True)  # 每图3对(每任务各1, 参数固定)

def exam_eval(tag, save_png=False):
    rgb, degbins, bins, attr, tasks = exam_cache
    with torch.no_grad():
        loss, acc, logits = run_batch(rgb, degbins, bins, attr)
    exact = (logits.argmax(-1) == bins).all(dim=1).float()          # [n] 整图全对
    per_task = {}
    for t in range(3):
        m = tasks == t
        if m.any():
            per_task[TASKS[t]] = (float(((logits.argmax(-1) == bins)[m]).float().mean()),
                                  int(m.sum()), float(exact[m].mean()))
    w(f"  [考试{tag}] 总pixel_acc={acc*100:.2f}% | 整图全对={exact.mean()*100:.1f}% | CE={loss.item():.4f}")
    for tname, (a, n, e) in per_task.items():
        w(f"    {tname}: pixel_acc={a*100:.2f}% ({n}对) 整图全对={e*100:.1f}%")
    if save_png:
        _dump_pngs(rgb, bins, logits, tasks)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return float(acc)

def _dump_pngs(rgb, bins, logits, tasks, max_n=9):
    """input | pred | target 三联图 ×16 放大 (nearest)"""
    pred_bins = logits.argmax(-1)                                    # [n,768]
    shown = 0
    for i in range(rgb.shape[0]):
        if shown >= max_n: break
        inp = rgb[i].reshape(IMG, IMG, 3).cpu().numpy()
        tgt = decode_bins(bins[i]).reshape(IMG, IMG, 3).cpu().numpy()
        prd = decode_bins(pred_bins[i]).reshape(IMG, IMG, 3).cpu().numpy()
        rows = np.concatenate([inp, prd, tgt], axis=1)               # [16,48,3]
        big = cv2.resize((rows * 255).round().astype(np.uint8), (48 * 16, IMG * 16), interpolation=cv2.INTER_NEAREST)
        name = f"exam_T{tasks[i].item()}_{shown:02d}_in-pred-tgt.png"
        imwrite_u8(os.path.join(OUT_DIR, name), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        shown += 1
    w(f"  [png] {shown}张三联图(输入|预测|目标) → {OUT_DIR}")

# ════════════════════════ 训练 ════════════════════════
w(f"\n[造对] 首轮训练对生成中...")
rng_train = np.random.default_rng(SEED * 1000)
# 分开控制 k: 带标签 k_labeled, 通用 k_generic
def build_train():
    rgb_l, dgb_l, bin_l, attr_l, task_l = build_pairs(train_labeled, args.k_labeled, rng_train)
    if not args.use_generic:
        return rgb_l, dgb_l, bin_l, attr_l, task_l   # [修复] use_generic=0 时不再混入通用图
    rgb_g, dgb_g, bin_g, attr_g, task_g = build_pairs(train_generic, args.k_generic, rng_train)
    return (torch.cat([rgb_l, rgb_g]), torch.cat([dgb_l, dgb_g]), torch.cat([bin_l, bin_g]),
            torch.cat([attr_l, attr_g]), torch.cat([task_l, task_g]))
train_cache = build_train()
exam_cache = build_exam()
exam_eval("(训练前基线)", save_png=False)

best_exam_acc = -1.0
n_nan = 0
t_start = time.time()
if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()

for ep in range(1, args.epochs + 1):
    if ep > 1 and (ep - 1) % args.regen_every == 0:
        train_cache = build_train()
    rgb_all, dgb_all, bin_all, attr_all, _ = train_cache
    n = rgb_all.shape[0]
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(SEED * 77 + ep), device="cpu").to(DEVICE)
    ep_loss = ep_acc = 0.0; steps = 0
    if args.batch_size >= n:
        # 满批次: 有放回采样凑满 (probe_vram: 峰值≈4.6MB/样本, batch1024=4.65GB 纯显存顶)
        idx = torch.randint(0, n, (args.batch_size,),
                            generator=torch.Generator().manual_seed(SEED * 13 + ep)).to(DEVICE)
        loss, acc, _ = run_batch(rgb_all[idx], dgb_all[idx], bin_all[idx], attr_all[idx])
        if torch.isfinite(loss):
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
            opt.step(); ep_loss += loss.item(); ep_acc += acc; steps = 1
        else:
            n_nan += 1; opt.zero_grad()
    else:
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss, acc, _ = run_batch(rgb_all[idx], dgb_all[idx], bin_all[idx], attr_all[idx])
            if not torch.isfinite(loss):
                n_nan += 1; opt.zero_grad(); continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
            opt.step()
            ep_loss += loss.item(); ep_acc += acc; steps += 1
    sched.step()
    if ep % 10 == 0 or ep == 1 or ep == args.epochs:
        eta = (time.time() - t_start) / ep * (args.epochs - ep)
        _vram = torch.cuda.memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0
        w(f"  E{ep}/{args.epochs} loss={ep_loss/max(steps,1):.4f} pixel_acc={ep_acc/max(steps,1)*100:.2f}% | GPU常驻{_vram:.0f}MB | {time.time()-t_start:.0f}s ETA{eta:.0f}s | NaN跳步:{n_nan}")
    if ep % args.exam_every == 0 or ep == args.epochs:
        acc_e = exam_eval(f"E{ep}")
        if acc_e > best_exam_acc:
            best_exam_acc = acc_e
            ckpt = os.path.join(SAVE_DIR, "V22_I2I_best.pt")
            torch.save({"visual_embed": visual_embed.state_dict(), "gen": gen.state_dict(),
                        "args": vars(args), "epoch": ep, "exam_acc": acc_e,
                        "n_bins": args.n_bins}, ckpt)
            w(f"  [保存] 最佳考试acc={acc_e*100:.2f}% E{ep} → {ckpt}")

# ════════════════════════ 收尾 ════════════════════════
w("\n" + "=" * 70)
w("[收尾] 最终考试 + png 输出")
_final = exam_eval("(最终)", save_png=True)
_peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0.0
w(f"\n[完成] 总耗时 {time.time()-t_start:.0f}s | 最佳考试acc={best_exam_acc*100:.2f}% | NaN跳步:{n_nan} | GPU峰值{_peak:.2f}GB")
w(f"[日志] {LOG_PATH}")
log.close()
