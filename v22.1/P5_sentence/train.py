"""
P5 词序列→句子合成 (1000轮详细日志版)
========================================
"""
import torch, torch.nn.functional as F, time, os, sys, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis, contrastive_loss


def load_sentences(path):
    role_map = {"subj": 0, "verb": 1, "obj": 2}
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) != 2: continue
            words, roles = [], []
            for item in parts[1].split():
                w, r = item.split(":")
                words.append(w)
                roles.append(role_map.get(r, -1))
            data.append((words, roles))
    return data


def make_scrambled(word_vecs, roles):
    n = len(roles)
    idx = list(range(n))
    random.shuffle(idx)
    return (torch.stack([word_vecs[i] for i in idx]),
            torch.tensor([roles[i] for i in idx], device=word_vecs.device))


def gpu_mem():
    if DEVICE.type != "cuda": return (0,0,0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2,
            torch.cuda.memory_reserved(DEVICE)/1024**2,
            torch.cuda.max_memory_allocated(DEVICE)/1024**2)


def train():
    ep = 1000; disp = 100; lr = 0.005
    print(f"P5 句子合成 epochs={ep} display={disp} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    sentences = load_sentences("C:/ai/P5_sentence/sentences.txt")
    print(f"[数据] {len(sentences)}句")

    @torch.no_grad()
    def enc(w):
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    encoded = []
    for words, roles in sentences:
        vecs = [enc(w) for w in words]
        if None not in vecs:
            encoded.append((torch.stack(vecs), torch.tensor(roles, device=DEVICE)))
    print(f"[编码] {len(encoded)}句有效")

    model = SentenceSynthesis().to(DEVICE)
    print(f"[P5] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; best_gap = 0.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0
        for wv, roles in encoded:
            correct = model(wv, roles, last_loss=last_loss)
            sv, sr = make_scrambled(wv, roles)
            scr = model(sv, sr, last_loss=last_loss)
            loss, cos_c, cos_s = contrastive_loss(correct, scr.unsqueeze(0))
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += cos_c; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            # === 详细日志 ===
            print(f"\n{'='*70}")
            print(f"[P5] Epoch {epoch:4d} / {ep} | Loss: {al:.6f} | 累计耗时: {time.time()-total_t0:.0f}s")
            print(f"{'='*70}")

            # 1. 位置权重
            print(f"[位置权重] "
                  f"w_subj={model.w_subj.item():.4f}  "
                  f"w_verb={model.w_verb.item():.4f}  "
                  f"w_obj={model.w_obj.item():.4f}")

            # 2. 调制信息
            exp = model.explore_state
            print(f"[探索区] 256D: mean={exp.mean().item():.6f} std={exp.std().item():.6f} "
                  f"range=[{exp.min().item():.4f}, {exp.max().item():.4f}]")
            meta_out = model.meta_fc(exp * min(last_loss * 20.0, 1.0))
            print(f"[元学习调制] 256D: mean={meta_out.mean().item():.6f} std={meta_out.std().item():.6f} "
                  f"range=[{meta_out.min().item():.4f}, {meta_out.max().item():.4f}]")
            print(f"[调制注入强度] L2={meta_out.norm().item():.4f}")

            # 3. 采样3个句子看cos正确/乱序
            print(f"\n[句子采样] 正确cos vs 乱序cos:")
            for wv, roles in encoded[:3]:
                c = model(wv, roles, last_loss=0.0)
                sv, sr = make_scrambled(wv, roles)
                s = model(sv, sr, last_loss=0.0)
                ref = c.detach()
                cc = F.cosine_similarity(c, ref, dim=-1).item()
                cs = F.cosine_similarity(s, ref, dim=-1).item()
                gap = cc - cs
                # 解码句子文本
                word_str = " ".join([f"{w}" for w, r in zip(
                    ["?"]*len(roles), roles)])  # placeholder
                print(f"  正确cos={cc:.4f} 乱序cos={cs:.4f} gap={gap:.4f} {'OK' if gap>0.3 else '...'}")

            # 4. GPU
            a, r, m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB res={r:.0f}MB peak={m:.0f}MB | epoch耗时: {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {epoch_cos/nb:.4f}")

    # 最终测试
    print(f"\n{'#'*60}")
    print("# 最终词序测试")
    gaps = []
    for wv, roles in encoded[:12]:
        correct = model(wv, roles, last_loss=0.0)
        sv, sr = make_scrambled(wv, roles)
        scr = model(sv, sr, last_loss=0.0)
        ref = correct.detach()
        cc = F.cosine_similarity(correct, ref, dim=-1).item()
        cs = F.cosine_similarity(scr, ref, dim=-1).item()
        gaps.append(cc - cs)
    avg_g = sum(gaps) / len(gaps)
    print(f"  平均gap: {avg_g:.3f}")
    print(f"  验收: gap>0.3 → {'PASS' if avg_g>0.3 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
