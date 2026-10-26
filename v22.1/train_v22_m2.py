"""
═══════════════════════════════════════════════════════════════
V22.1 图生图 M2 — 64×64 patch化 + P7 跨patch路由 (20261024)
═══════════════════════════════════════════════════════════════

架构 (规划见 D:/ai/V22.1_图生图M2_规划_20261024.md):
  64×64 退化图 → 16 个 16×16 patch
  每 patch: 768 bins → 冻结bin_embed查表 + 3×3/5×5 均值上下文  (M1 v4.1 机制, batch=b×16)
  P7 跨patch路由: patch_token(VisualEmbed均池化) 自注意力 4×4 栅格 → word_out
    → W_pctx 投影加到对应 patch 全部 768 位                    ← vision 接入引擎第一步
  全局身份 h: sent_vec(P7) + P3Vis(整图) → input_proj → encoder → 球面×h_scale
  逐位门控 hp = h ⊙ tanh(clamp(gate(inpatch_pos),±2))          (v4.1)
  logits = (feat + pctx + hp + patch_pos + inpatch_pos)·bin_embedᵀ → CE(12288位×16阶)
  位置编码因式分解: patch_pos[16,256] + inpatch_pos[768,256]
  局部给均值矩阵, 全局给 P7 —— 跨patch 12288² 均值矩阵不可行也无必要

显存: ~85MB/样本 → batch 64 ≈ 5.4GB 纯显存甜点 (= M1 b1024 的有效规模)
精度: FP32 全程; 分配器: 默认(超额响亮OOM, 不静默溢出)

用法:
  python train_v22_m2.py --epochs 1200               # 正式
  python train_v22_m2.py --smoke                     # 冒烟
"""
import os
# [E5修复] 防碎片分配参数(垃圾回收阈值+分段上限), 对抗 regen 双份缓存+桌面挤压的碎片化OOM;
# 仍不用 expandable_segments(静默溢出共享显存), 超额依旧响亮OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.8,max_split_size_mb:256")
import torch, torch.nn as nn, torch.nn.functional as F
import sys, glob, time, random, argparse
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE, SAVE_DIR, BASE_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3_vision"))
from p3vis import P3Vis
from P7_cross_sent.model import P7WordRouter2048

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.012)
parser.add_argument("--n_bins", type=int, default=16)
parser.add_argument("--use_generic", type=int, default=1)
parser.add_argument("--k_labeled", type=int, default=16)
parser.add_argument("--k_generic", type=int, default=4)
parser.add_argument("--gen_max", type=int, default=300)
parser.add_argument("--exam_n_generic", type=int, default=30)
parser.add_argument("--regen_every", type=int, default=100)
parser.add_argument("--exam_every", type=int, default=100)
parser.add_argument("--clip_norm", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sigma_curriculum", type=int, default=0, help="E4课程开关: 1=三段σ课程, 0=全档(E2最佳配置)")
parser.add_argument("--max_sources", type=int, default=0, help="诊断用: 只用前N张带标签源图(0=全部)")
parser.add_argument("--gate_pair", type=int, default=1, help="[缺陷修复] 身份门控吃(patch,位内位置)交互: 1=修复版, 0=旧版纯inpatch")
parser.add_argument("--encoder_width", type=int, default=256, help="N2: 身份encoder隐层宽度(256=原版)")
parser.add_argument("--spatial_upgrade", type=int, default=0, help="[空间升级] 1=方向线状上下文+patch内邻域注意力, 0=原版固定均值")
parser.add_argument("--transform_v2", type=int, default=0, help="[转换多样性v2] 1=无缝全图邻域+自学习路由+多头注意力+copy/transform门控")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

LABELED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_data")
GENERIC_DIR = "D:/ai/output/images"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_out_m2")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "logs", f"i2i_m2_{time.strftime('%Y%m%d_%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")
def w(s): log.write(s + "\n"); log.flush(); print(s, flush=True)

w("=" * 70)
w(f"  V22.1 图生图M2 | SEED={SEED} | bins={args.n_bins} | bs={args.batch_size} | lr={args.lr}")
w(f"  64×64 patch化 + P7跨patch路由 | epochs={args.epochs} use_generic={args.use_generic}")
w("=" * 70)

if args.smoke:
    args.epochs = 3; args.gen_max = 40; args.k_generic = 2
    args.exam_n_generic = 10; args.regen_every = 2; args.exam_every = 1
    w("[SMOKE] 冒烟模式")

# ════════════════════════ 数据 ════════════════════════
IMG = 64          # 整图边长
PATCH = 16        # patch 边长
NPATCH = 16       # 4×4
N_IN = PATCH * PATCH * 3          # 768 每 patch
N_POS = NPATCH * N_IN             # 12288 总输出位

def imread_u8(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

def imwrite_u8(path, img_bgr):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img_bgr)
    if ok: buf.tofile(path)

def load_rgb(path):
    img = imread_u8(path)
    if img is None: return None
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    bgr = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, bgr

def to_patch_major(rgb):
    """[64,64,3] 行主序 → [16,768] patch-major (p = py*4+px, patch 内 16×16×3 行主序)"""
    return rgb.reshape(4, 16, 4, 16, 3).transpose(0, 2, 1, 3, 4).reshape(NPATCH, N_IN)

def from_patch_major(flat):
    """[12288] patch-major → [64,64,3] (PNG 还原用)"""
    return flat.reshape(4, 4, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(IMG, IMG, 3)

labeled_files = sorted(glob.glob(os.path.join(LABELED_DIR, "*.bmp")))
# [E5修复] 递归 glob: output/images 的图大多在子目录, 非递归只能看到顶层 294 张
generic_files = sorted(sum([glob.glob(os.path.join(GENERIC_DIR, "**", e), recursive=True) for e in ("*.jpg", "*.png", "*.jpeg")], []))
_rng_files = np.random.default_rng(SEED)
generic_files = list(np.array(generic_files)[_rng_files.permutation(len(generic_files))][:args.gen_max + args.exam_n_generic])

sources, _bad = [], 0
for f in labeled_files:
    r = load_rgb(f)
    if r is None: _bad += 1; continue
    sources.append({"name": os.path.basename(f), "kind": "labeled", "rgb": r[0], "bgr": r[1],
                    "rgb_pm": to_patch_major(r[0]).reshape(-1)})
_generic_pool = []
for f in generic_files:
    r = load_rgb(f)
    if r is None: _bad += 1; continue
    _generic_pool.append({"name": os.path.basename(f), "kind": "generic", "rgb": r[0], "bgr": r[1],
                          "rgb_pm": to_patch_major(r[0]).reshape(-1)})

_exam_names = {f"apple_{i:02d}.bmp" for i in range(16, 21)} | {"sun_05.bmp"}
exam_labeled = [s for s in sources if s["name"] in _exam_names]
train_labeled = [s for s in sources if s["name"] not in _exam_names]
exam_generic = _generic_pool[-args.exam_n_generic:]
train_generic = _generic_pool[:-args.exam_n_generic]

w(f"[数据] 带标签(train/exam): {len(train_labeled)}/{len(exam_labeled)} | 通用(train/exam): {len(train_generic)}/{len(exam_generic)} | 坏图: {_bad}")

TASKS = ["去噪", "遮补", "上色"]

SIGMA_POOL = [0.10, 0.20, 0.30]   # E4: 去噪课程当前档(全局; 考试集构建时为全档)
def set_sigma_pool(ep):
    """E4 课程: 前1/3轮只学σ0.1, 中1/3加σ0.2, 后1/3全档 (E4实测负优化, 默认关)"""
    if not args.sigma_curriculum:
        SIGMA_POOL[:] = [0.10, 0.20, 0.30]; return
    e3 = max(1, args.epochs // 3)
    if ep <= e3: SIGMA_POOL[:] = [0.10]
    elif ep <= 2 * e3: SIGMA_POOL[:] = [0.10, 0.20]
    else: SIGMA_POOL[:] = [0.10, 0.20, 0.30]

def degrade(rgb, task, rng):
    out = rgb.copy()
    sigma = 0.0
    if task == 0:
        sigma = float(rng.choice(SIGMA_POOL))
        out = np.clip(rgb + rng.normal(0.0, sigma, rgb.shape).astype(np.float32), 0.0, 1.0)
    elif task == 1:
        k = int(rng.choice([16, 24, 32]))          # 64×64: 遮块同比放大
        y = int(rng.integers(0, IMG - k + 1)); x = int(rng.integers(0, IMG - k + 1))
        out[y:y + k, x:x + k] = 0.5
    else:
        g = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)
        out = np.stack([g, g, g], axis=-1)
    return np.clip(out, 0.0, 1.0), task, sigma

def to_bins(rgb_flat):
    return np.clip((rgb_flat * args.n_bins).astype(np.int64), 0, args.n_bins - 1)

_p3vis = P3Vis()

def build_pairs(srcs, k_per_img, rng, per_task=False):
    """退化对; attr 双路: 整图64×64 P3Vis(全局h) + 每 patch 16×16 P3Vis(E2: 进P7路由)"""
    list_rgb, list_dgb, list_bin, list_attr_p, list_attr_i, list_task, list_sigma = [], [], [], [], [], [], []
    t0 = time.time()
    for s in srcs:
        tasks = range(3) if per_task else [int(rng.integers(0, 3)) for _ in range(k_per_img)]
        for task in tasks:
            deg, task, sig = degrade(s["rgb"], task, rng)
            deg_bgr = (deg[:, :, ::-1] * 255.0).astype(np.uint8)
            attr_i = np.nan_to_num(_p3vis.process_image(deg_bgr).astype(np.float32),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            attr_p = np.empty((NPATCH, 256), dtype=np.float32)     # E2: 16 个 patch 各自属性
            for pi in range(NPATCH):
                py, px = divmod(pi, 4)
                pb = deg_bgr[py * PATCH:(py + 1) * PATCH, px * PATCH:(px + 1) * PATCH]
                attr_p[pi] = np.nan_to_num(_p3vis.process_image(pb).astype(np.float32),
                                           nan=0.0, posinf=0.0, neginf=0.0)
            deg_pm = to_patch_major(deg).reshape(-1)               # [12288] patch-major
            list_rgb.append(deg_pm)
            list_dgb.append(to_bins(deg_pm))
            list_bin.append(to_bins(s["rgb_pm"]))                  # 目标图: 预转换的 patch-major
            list_attr_p.append(attr_p)
            list_attr_i.append(attr_i)
            list_task.append(task)
            list_sigma.append(sig)                                 # E4: 去噪对的σ档位(非去噪=0)
    rgb_t = torch.tensor(np.stack(list_rgb), dtype=torch.float32, device=DEVICE)
    dgb_t = torch.tensor(np.stack(list_dgb), dtype=torch.long, device=DEVICE).reshape(len(list_rgb), NPATCH, N_IN)
    bin_t = torch.tensor(np.stack(list_bin), dtype=torch.long, device=DEVICE).reshape(len(list_rgb), NPATCH, N_IN)
    attrp_t = torch.tensor(np.stack(list_attr_p), dtype=torch.float32, device=DEVICE)   # [N,16,256]
    attri_t = torch.tensor(np.stack(list_attr_i), dtype=torch.float32, device=DEVICE)   # [N,256]
    task_t = torch.tensor(list_task, dtype=torch.long, device=DEVICE)
    sigma_t = torch.tensor(list_sigma, dtype=torch.float32, device=DEVICE)
    w(f"  [造对] {len(srcs)}图×{3 if per_task else k_per_img} = {len(list_rgb)}对(64×64) | per-patch P3Vis+量化 {time.time()-t0:.1f}s | σ池={sorted(set(SIGMA_POOL))}")
    return rgb_t, dgb_t, bin_t, attrp_t, attri_t, task_t, sigma_t

# ════════════════════════ 模型 ════════════════════════
class VisualEmbed(nn.Module):
    """单 patch 16×16像素 → 768 → 256d, 正交+球面 (M1 原样, 逐 patch 调用)"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(768, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
    def forward(self, img):
        return F.normalize(self.proj(img), dim=-1)

class I2IGenM2(nn.Module):
    """M2 生成头: M1 v4.1 逐patch机制 (batch=b×16) + P7跨patch上下文 + 因式分解位置编码"""
    def __init__(self, embed_dim=256, n_patch=NPATCH, n_in=N_IN, n_bins=args.n_bins):
        super().__init__()
        be = torch.empty(n_bins, embed_dim); nn.init.orthogonal_(be)
        self.register_buffer("bin_embed", be)                      # 冻结视觉词典
        self.register_buffer("avg3", self._neighbor_avg(PATCH, 1)) # patch 内 3×3
        self.register_buffer("avg5", self._neighbor_avg(PATCH, 2)) # patch 内 5×5
        self.mix_in = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mix_c3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mix_c5 = nn.Linear(embed_dim, embed_dim, bias=False)
        for m in (self.mix_in, self.mix_c3, self.mix_c5):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        self.pctx_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # P7 word_out → patch上下文
        nn.init.xavier_uniform_(self.pctx_proj.weight, gain=0.1)
        self.attr_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # E2: per-patch P3Vis → patch_token
        nn.init.xavier_uniform_(self.attr_proj.weight, gain=0.1)
        self.input_proj = nn.Linear(512, embed_dim, bias=False)    # sent_vec + P3Vis
        self.encoder = nn.Sequential(nn.Linear(embed_dim, args.encoder_width), nn.GELU(),
                                     nn.Linear(args.encoder_width, embed_dim), nn.GELU())
        for m in self.encoder:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
        self.patch_pos = nn.Parameter(torch.randn(n_patch, embed_dim) * 0.1)
        self.inpatch_pos = nn.Parameter(torch.randn(n_in, embed_dim) * 0.1)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        self.gate_pair = nn.Linear(embed_dim * 2, embed_dim, bias=False)  # [缺陷修复] gate吃(patch,位内位置)交互
        nn.init.xavier_uniform_(self.gate_pair.weight, gain=0.1)
        if args.spatial_upgrade:
            # [空间升级 S1] 8方向线状移位上下文: 边缘/纹理方向可学习(固定均值只会"糊")
            self.shifts = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            self.register_buffer("shiftM", self._shift_matrix(PATCH, self.shifts))   # [8,768,768]
            self.mix_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_a3 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_a5 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_d = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False)
                                        for _ in range(len(self.shifts))])   # block-diagonal 折叠
            for m in [self.mix_q, self.mix_a3, self.mix_a5] + list(self.mix_d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            # [空间升级 S2] patch 内邻域注意力: 输出位←输入位的 768×768 关系矩阵 = 可检视的"像素间规律"
            self.attn_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_o = nn.Linear(embed_dim, embed_dim, bias=False)
            for m in (self.attn_q, self.attn_k, self.attn_v, self.attn_o):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
        if args.transform_v2:
            # [转换多样性 v2] 四件套（详见迭代日志）：无缝全图邻域 / 自学习路由 / 多头注意力 / copy-transform 门控
            offs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)]
            idx = torch.full((len(offs), NPATCH * N_IN), 0, dtype=torch.long)
            msk = torch.zeros(len(offs), NPATCH * N_IN)
            for k, (dy, dx) in enumerate(offs):
                for p in range(NPATCH * N_IN):
                    patch_i, rem = divmod(p, N_IN)
                    y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
                    c = rem % 3
                    py, px = divmod(patch_i, 4)
                    Y, X = py * PATCH + y_in + dy, px * PATCH + x_in + dx
                    if 0 <= Y < IMG and 0 <= X < IMG:      # 跨 patch 的真邻居（无缝）
                        idx[k, p] = ((Y // 16) * 4 + X // 16) * N_IN + (Y % 16) * 48 + (X % 16) * 3 + c
                        msk[k, p] = 1.0
            self.register_buffer("nbr_idx", idx)           # [16,12288] 无缝邻居索引(越界=自身)
            self.register_buffer("nbr_mask", msk)          # [16,12288] 越界权重置零
            self.nbr_lin = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in offs])
            for m in self.nbr_lin: nn.init.xavier_uniform_(m.weight, gain=0.1)
            self.router = nn.Linear(256, len(offs), bias=False)   # 自学习路由: 每位按内容软选 16 支路
            nn.init.xavier_uniform_(self.router.weight, gain=0.1)
            self.attn_q4 = nn.Linear(256, 256, bias=False)        # 多头 patch 内注意力 4×64
            self.attn_k4 = nn.Linear(256, 256, bias=False)
            self.attn_v4 = nn.Linear(256, 256, bias=False)
            self.attn_o4 = nn.Linear(256, 256, bias=False)
            for m in (self.attn_q4, self.attn_k4, self.attn_v4, self.attn_o4):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            self.tgate = nn.Linear(256, 1, bias=False)            # copy/transform 显式门控
            nn.init.xavier_uniform_(self.tgate.weight, gain=0.1)
        self.h_scale = nn.Parameter(torch.tensor(2.0))
        nn.init.orthogonal_(self.input_proj.weight)
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
    @staticmethod
    def _shift_matrix(img, shifts):
        """[K,768,768] 移位矩阵: S_k[p,q]=1 当 q 是 p 沿方向 k 的同通道邻居(越界补零)"""
        n = img * img * 3
        Ms = torch.zeros(len(shifts), n, n)
        for si, (dy, dx) in enumerate(shifts):
            for y in range(img):
                for x in range(img):
                    y2, x2 = y + dy, x + dx
                    if 0 <= y2 < img and 0 <= x2 < img:
                        for c in range(3):
                            Ms[si, (y * img + x) * 3 + c, (y2 * img + x2) * 3 + c] = 1.0
        return Ms
    def forward(self, deg_bins, sent_vec, vis_attr, patch_ctx):
        if args.transform_v2:
            return self._forward_v2(deg_bins, sent_vec, vis_attr, patch_ctx, deg_bins.shape[0])
        """deg_bins [b,16,768] | sent_vec [b,256] P7句向量 | vis_attr [b,256] | patch_ctx [b,16,256] P7 word_out"""
        b = deg_bins.shape[0]
        q_in = self.bin_embed[deg_bins.reshape(b * NPATCH, N_IN)]           # [b*16,768,256]
        ctx3 = self.avg3 @ q_in
        ctx5 = self.avg5 @ q_in
        if args.spatial_upgrade:
            # S1: 逐支路 Linear 相加 (= concat+Linear 的 block-diagonal 折叠, 免 8GB cat 峰值)
            feat = self.mix_q(q_in) + self.mix_a3(ctx3) + self.mix_a5(ctx5)
            for k in range(len(self.shifts)):
                feat = feat + self.mix_d[k](self.shiftM[k] @ q_in)
            # S2: 分块邻域注意力 (查询位 128 一块, 免 768×768 全量分数矩阵)
            k_proj = self.attn_k(q_in + self.inpatch_pos.unsqueeze(0))      # 输入内容+位置 作 K
            v_proj = self.attn_v(q_in)
            outs = []
            for i in range(0, N_IN, 128):
                sl = slice(i, min(N_IN, i + 128))
                qk = self.attn_q(feat[:, sl, :]) @ k_proj.transpose(-1, -2) / (256 ** 0.5)
                outs.append(torch.softmax(qk, dim=-1) @ v_proj)             # [n,ch,768]→[n,ch,256]
            feat = feat + self.attn_o(torch.cat(outs, dim=1))               # 像素转换: 从输入位聚合
        else:
            feat = self.mix_in(q_in) + self.mix_c3(ctx3) + self.mix_c5(ctx5)  # [b*16,768,256]
        feat = feat.reshape(b, NPATCH, N_IN, 256)
        feat = feat + self.pctx_proj(patch_ctx).unsqueeze(2)                # [b,16,1,256] 广播到 patch 内 768 位
        h = (self.h_scale * F.normalize(self.encoder(self.input_proj(
            torch.cat([sent_vec, vis_attr], dim=-1))), dim=-1))             # [b,256] 球面
        if args.gate_pair:
            # [缺陷修复] 门控吃 (patch, 位内位置) 拼接交互 → 恢复 (图像×绝对位置) 完整记忆通路。
            # gp 输入无关, 每 forward 只算一次 [16,768,256] (~13MB), 无批量开销
            gp = torch.tanh(torch.clamp(self.gate_pair(torch.cat(
                [self.patch_pos.unsqueeze(1).expand(-1, N_IN, -1),
                 self.inpatch_pos.unsqueeze(0).expand(NPATCH, -1, -1)], dim=-1)), -2.0, 2.0))
            hp_in = h.unsqueeze(1).unsqueeze(2) * gp.unsqueeze(0)           # [b,16,768,256]
            q = feat + hp_in + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)
        else:
            hp_in = h.unsqueeze(1) * torch.tanh(torch.clamp(
                self.gate(self.inpatch_pos), -2.0, 2.0)).unsqueeze(0)       # [b,768,256] 旧版: 丢patch交互
            q = (feat + hp_in.unsqueeze(1)
                    + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0))
        return torch.einsum('bpwd,kd->bpwk', q, self.bin_embed)             # [b,16,768,16]

    def _forward_v2(self, deg_bins, sent_vec, vis_attr, patch_ctx, b):
        """[转换多样性 v2] 无缝全图邻域 gather + 自学习路由 + 多头 patch 内注意力 + copy/transform 显式门控"""
        q_flat = self.bin_embed[deg_bins.reshape(b, NPATCH * N_IN)]         # [b,12288,256] 输入查表(全图)
        # 1) 无缝邻居分支 × 自学习路由: 每位按输入内容在 16 个空间支路间软选择(跨 patch 真邻居, 无缝)
        router = torch.softmax(self.router(q_flat), dim=-1)                 # [b,12288,16]
        acc = torch.zeros_like(q_flat)
        for k in range(self.nbr_idx.shape[0]):
            g = q_flat.index_select(1, self.nbr_idx[k]) * self.nbr_mask[k].unsqueeze(0).unsqueeze(-1)
            acc = acc + router[..., k:k + 1] * self.nbr_lin[k](g)
        q4 = acc.reshape(b, NPATCH, N_IN, 256)
        # 2) 多头 patch 内注意力(4×64): 位→位的细粒度关系, 每头可学不同规律(颜色延续/边缘/纹理周期)
        Qp, Kp, Vp = self.attn_q4(q4), self.attn_k4(q4), self.attn_v4(q4)
        outs = []
        for hi in range(4):
            sl = slice(hi * 64, (hi + 1) * 64)
            s = Qp[..., sl] @ Kp[..., sl].transpose(-1, -2) / (64 ** 0.5)   # [b,16,768,768]
            outs.append(torch.softmax(s, dim=-1) @ Vp[..., sl])
        q4 = q4 + self.attn_o4(torch.cat(outs, dim=-1))
        # 3) 全局身份 + 逐位门控(同 v1)
        h = (self.h_scale * F.normalize(self.encoder(self.input_proj(
            torch.cat([sent_vec, vis_attr], dim=-1))), dim=-1))
        gp = torch.tanh(torch.clamp(self.gate_pair(torch.cat(
            [self.patch_pos.unsqueeze(1).expand(-1, N_IN, -1),
             self.inpatch_pos.unsqueeze(0).expand(NPATCH, -1, -1)], dim=-1)), -2.0, 2.0))
        q4 = q4 + h.unsqueeze(1).unsqueeze(2) * gp.unsqueeze(0) \
                + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)
        synth = torch.einsum('bpwd,kd->bpwk', q4, self.bin_embed)           # 合成路 logits
        # 4) copy/transform 显式门控: 每个像素位自己决定"抄输入"还是"用合成"
        pos_full = (self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)).reshape(1, NPATCH * N_IN, 256)
        copy_l = ((q_flat + pos_full) @ self.bin_embed.t()).reshape(b, NPATCH, N_IN, 16)
        g = torch.sigmoid(self.tgate(q4)).reshape(b, NPATCH, N_IN, 1)
        return g * synth + (1.0 - g) * copy_l

visual_embed = VisualEmbed().to(DEVICE)
gen = I2IGenM2().to(DEVICE)
p7 = P7WordRouter2048(word_dim=256, inner_dim=256, heads=16, head_dim=16, max_len=128, num_groups=4).to(DEVICE)
n_params = sum(p.numel() for m in [visual_embed, gen, p7] for p in m.parameters())
w(f"[模型] 可训参数: {n_params:,} + 冻结词典{args.n_bins}×256 | P7 跨patch路由已接入 | 全FP32")

# ── 纯显存约束 ──
if DEVICE.type == "cuda":
    _free_b, _tot_b = torch.cuda.mem_get_info()
    w(f"[显存] 专用显存: 空闲{_free_b/2**30:.1f}GB / 总{_tot_b/2**30:.1f}GB")
    _cap_gb = float(os.environ.get("M2_VRAM_CAP_GB", "6.0"))  # [回退修复] 9.5→6.0: 实测峰值5.45GB,
    # 静态大上限在桌面应用中途涨显存时会把本进程挤进共享显存(用户观察到的"泄露虚拟内存")
    _eff_cap = max(0.5, min(_cap_gb, _free_b / 2**30 - 1.0))
    torch.cuda.set_per_process_memory_fraction(min(1.0, _eff_cap / (_tot_b / 2**30)))
    w(f"[显存] 本进程分配上限 {_eff_cap:.1f}GB (超限响亮OOM, 不溢出共享显存)")

params = list(visual_embed.parameters()) + list(gen.parameters()) + list(p7.parameters())
opt = torch.optim.Adam(params, lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

def run_batch(rgb, degbins, target_bin, attr_p, attr_i):
    pe = visual_embed(rgb.reshape(rgb.shape[0] * NPATCH, N_IN)).reshape(rgb.shape[0], NPATCH, 256)
    patch_tokens = pe + gen.attr_proj(F.normalize(attr_p, dim=-1))         # E2: 局部属性进路由
    wo, sv, _, _, _ = p7.forward_batch(patch_tokens, patch_tokens, [NPATCH] * rgb.shape[0], [NPATCH] * rgb.shape[0], last_loss=0.0)
    logits = gen(degbins, sv, F.normalize(attr_i, dim=-1), wo)             # [b,16,768,16]
    loss = F.cross_entropy(logits.reshape(-1, args.n_bins), target_bin.reshape(-1))
    with torch.no_grad():
        acc = (logits.argmax(-1) == target_bin).float().mean().item()
    return loss, acc, logits

# ════════════════════════ 考试 ════════════════════════
exam_rng = np.random.default_rng(SEED + 1)
def build_exam():
    return build_pairs(exam_labeled + (exam_generic if args.use_generic else []),
                       None, exam_rng, per_task=True)

def exam_eval(tag, save_png=False):
    """分块评估: 整批108对峰值~13GB会随桌面显存波动OOM(E5-E600事故), 分块降到训练同级"""
    rgb, degbins, bins, attr_p, attr_i, tasks, sigmas = exam_cache
    n = rgb.shape[0]
    chunk = 24
    loss_sum = acc_sum = 0.0
    pred_all = torch.empty(n, NPATCH * N_IN, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for i in range(0, n, chunk):
            sl = slice(i, min(n, i + chunk))
            loss, acc, logits = run_batch(rgb[sl], degbins[sl], bins[sl], attr_p[sl], attr_i[sl])
            bs_ = sl.stop - sl.start
            loss_sum += loss.item() * bs_; acc_sum += acc * bs_
            pred_all[sl] = logits.argmax(-1).reshape(bs_, -1)
    loss = loss_sum / n; acc = acc_sum / n
    logits = None
    bins_flat = bins.reshape(n, -1)
    exact = (pred_all == bins_flat).all(dim=1).float()                  # [n] 整图全对
    per_task = {}
    for t in range(3):
        m = tasks == t
        if m.any():
            per_task[TASKS[t]] = (float(((pred_all == bins_flat)[m]).float().mean()),
                                  int(m.sum()), float(exact[m].mean()))
    sig_line = ""
    if sigmas is not None:                                             # E4: 去噪按σ分档
        for sv_ in (0.10, 0.20, 0.30):
            m = (tasks == 0) & (sigmas == sv_)
            if m.any():
                sig_line += f" | σ{sv_:.1f}={float(((pred_all == bins_flat)[m]).float().mean())*100:.1f}%"
    w(f"  [考试{tag}] 总pixel_acc={acc*100:.2f}% | 整图全对={exact.mean()*100:.1f}% | CE={loss:.4f}{sig_line}")
    for tname, (a, n_, e) in per_task.items():
        w(f"    {tname}: pixel_acc={a*100:.2f}% ({n_}对) 整图全对={e*100:.1f}%")
    if save_png:
        _dump_pngs(rgb, bins_flat, pred_all, tasks)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return float(acc)

def _dump_pngs(rgb, bins_flat, pred_flat, tasks, max_n=9):
    shown = 0
    for i in range(rgb.shape[0]):
        if shown >= max_n: break
        inp = from_patch_major(rgb[i].cpu())                        # patch-major → [64,64,3]
        tgt = from_patch_major(((bins_flat[i].float() + 0.5) / args.n_bins).cpu())
        prd = from_patch_major(((pred_flat[i].float() + 0.5) / args.n_bins).cpu())
        rows = np.concatenate([inp.numpy(), prd.numpy(), tgt.numpy()], axis=1)
        big = cv2.resize((rows * 255).round().astype(np.uint8), (192 * 8, IMG * 8), interpolation=cv2.INTER_NEAREST)
        imwrite_u8(os.path.join(OUT_DIR, f"m2_exam_T{tasks[i].item()}_{shown:02d}_in-pred-tgt.png"),
                   cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        shown += 1
    w(f"  [png] {shown}张三联图(64×64, ×8放大) → {OUT_DIR}")

# ════════════════════════ 训练 ════════════════════════
rng_train = np.random.default_rng(SEED * 1000)
def build_train():
    _srcs = train_labeled[:args.max_sources] if args.max_sources else train_labeled  # 诊断: 4图记忆测试
    rgb_l, dgb_l, bin_l, ap_l, ai_l, task_l, sg_l = build_pairs(_srcs, args.k_labeled, rng_train)
    if not args.use_generic:
        return rgb_l, dgb_l, bin_l, ap_l, ai_l, task_l, sg_l
    rgb_g, dgb_g, bin_g, ap_g, ai_g, task_g, sg_g = build_pairs(train_generic, args.k_generic, rng_train)
    return (torch.cat([rgb_l, rgb_g]), torch.cat([dgb_l, dgb_g]), torch.cat([bin_l, bin_g]),
            torch.cat([ap_l, ap_g]), torch.cat([ai_l, ai_g]), torch.cat([task_l, task_g]),
            torch.cat([sg_l, sg_g]))

w("\n[造对] 首轮训练对生成中...")
exam_cache = build_exam()          # 考试集: 全σ档固定参数
set_sigma_pool(1)                  # E4 课程从 σ0.1 起步
train_cache = build_train()
exam_eval("(训练前基线)")

best_exam_acc = -1.0
_ckpt_path = os.path.join(SAVE_DIR, "V22_I2I_M2_best.pt")
if os.path.exists(_ckpt_path):     # [修复] 全局最佳保护: 差于历史最佳的运行不再覆盖checkpoint
    try: best_exam_acc = float(torch.load(_ckpt_path, map_location="cpu").get("exam_acc", -1.0))
    except Exception: pass
    w(f"[checkpoint] 历史最佳考试acc={best_exam_acc*100:.2f}% (只被更优运行覆盖)")
n_nan = 0; n_oom = 0
t_start = time.time()
if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()

for ep in range(1, args.epochs + 1):
    if ep > 1 and (ep - 1) % args.regen_every == 0:
        set_sigma_pool(ep)         # E4: 课程档位随 epoch 推进
        train_cache = None;        # [修复] 先放旧缓存再建新, 避免 GPU 上双份共存+碎片
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        train_cache = build_train()
    rgb_all, dgb_all, bin_all, ap_all, ai_all, _, _ = train_cache
    n = rgb_all.shape[0]
    if args.batch_size >= n:
        idx = torch.randint(0, n, (args.batch_size,),
                            generator=torch.Generator().manual_seed(SEED * 13 + ep)).to(DEVICE)
        batches = [idx]
    else:
        perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(SEED * 77 + ep), device="cpu").to(DEVICE)
        batches = [perm[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
    ep_loss = ep_acc = 0.0; steps = 0; consec_err = 0
    for idx in batches:
        try:
            loss, acc, _ = run_batch(rgb_all[idx], dgb_all[idx], bin_all[idx], ap_all[idx], ai_all[idx])
        except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
            torch.cuda.empty_cache(); n_oom += 1; opt.zero_grad(); consec_err += 1
            w(f"  [E{ep}] 批次异常跳过({type(e).__name__}, 累计{n_oom}): {str(e)[:60]}")
            if consec_err > 30:
                w(f"  [E{ep}] 连续30批异常, 疑CUDA上下文损坏, 弃本epoch")
                break
            continue
        consec_err = 0
        if not torch.isfinite(loss):
            n_nan += 1; opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
        opt.step()
        ep_loss += loss.item(); ep_acc += acc; steps += 1
    sched.step()
    if DEVICE.type == "cuda" and ep % 25 == 0:   # [回退修复] 运行中动态防护: 桌面挤压时主动归还缓存
        _fr, _ = torch.cuda.mem_get_info()
        _rs = torch.cuda.memory_reserved()
        if _rs > _fr - 0.75 * 2**30:
            torch.cuda.empty_cache()
            w(f"  [显存] 桌面挤压防护: 归还缓存 (持有{_rs/2**30:.1f}GB / 物理空闲{_fr/2**30:.1f}GB)")
    if ep % 10 == 0 or ep == 1 or ep == args.epochs:
        eta = (time.time() - t_start) / ep * (args.epochs - ep)
        _vram = torch.cuda.memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0
        w(f"  E{ep}/{args.epochs} loss={ep_loss/max(steps,1):.4f} pixel_acc={ep_acc/max(steps,1)*100:.2f}% | GPU常驻{_vram:.0f}MB | {time.time()-t_start:.0f}s ETA{eta:.0f}s | NaN跳步:{n_nan}")
    if ep % args.exam_every == 0 or ep == args.epochs:
        try:
            acc_e = exam_eval(f"E{ep}")
        except torch.OutOfMemoryError:                 # 桌面应用挤显存时跳过该次考试, 不炸训练
            torch.cuda.empty_cache()
            w(f"  [考试E{ep}] OOM跳过(桌面显存挤压)")
            acc_e = -1.0
        if acc_e > best_exam_acc:
            best_exam_acc = acc_e
            ckpt = os.path.join(SAVE_DIR, "V22_I2I_M2_best.pt")
            torch.save({"visual_embed": visual_embed.state_dict(), "gen": gen.state_dict(),
                        "p7": p7.state_dict(), "args": vars(args), "epoch": ep,
                        "exam_acc": acc_e, "n_bins": args.n_bins}, ckpt)
            w(f"  [保存] 最佳考试acc={acc_e*100:.2f}% E{ep} → {ckpt}")

_peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0.0
w(f"\n[完成] 总耗时 {time.time()-t_start:.0f}s | 最佳考试acc={best_exam_acc*100:.2f}% | NaN跳步:{n_nan} | GPU峰值{_peak:.2f}GB")
w(f"[日志] {LOG_PATH}")
log.close()
