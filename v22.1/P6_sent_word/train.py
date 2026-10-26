"""
P6 句子→词序列 反向解码 (1000轮详细日志版)
==============================================
冻结P1+P5 → P6解码
"""
import torch, torch.nn.functional as F, time, os, sys, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P6_sent_word.model import SentToWordsDecoder, decode_loss


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


def gpu_mem():
    if DEVICE.type != "cuda": return (0,0,0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2,
            torch.cuda.memory_reserved(DEVICE)/1024**2,
            torch.cuda.max_memory_allocated(DEVICE)/1024**2)


def train():
    ep = 1000; disp = 100; lr = 0.003
    print(f"P6 句子→词序列 epochs={ep} display={disp} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # P5 - load saved checkpoint
    p5 = SentenceSynthesis().to(DEVICE)
    p5_path = os.path.join(SAVE_DIR, "P5_final.pt")
    if os.path.exists(p5_path):
        p5_ckpt = torch.load(p5_path, map_location=DEVICE)
        state = p5_ckpt.get("model_state_dict", p5_ckpt.get("model", {}))
        if state: p5.load_state_dict(state)
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

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
        if None not in vecs and len(vecs) == 3:
            encoded.append((torch.stack(vecs), torch.tensor(roles, device=DEVICE)))
    print(f"[编码] {len(encoded)}句有效 (3词)")

    p6 = SentToWordsDecoder().to(DEVICE)
    print(f"[P6] 参数: {sum(p.numel() for p in p6.parameters()):,}")
    opt = torch.optim.Adam(p6.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; best_cos = 0.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0

        for wv, roles in encoded:
            with torch.no_grad():
                svec = p5(wv, roles, last_loss=0.0)

            pred = p6(svec.unsqueeze(0), last_loss=last_loss)
            true = wv.unsqueeze(0)
            mask = torch.ones(1, 3, device=DEVICE)
            loss, avg_cos = decode_loss(pred, true, mask)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += avg_cos; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P6] Epoch {epoch:4d} / {ep} | Loss: {al:.6f} | avg_cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")
            print(f"{'='*70}")

            # 调制
            exp = p6.explore_state
            meta = p6.meta_fc(exp * min(last_loss*20.0, 1.0))
            print(f"[探索区] 32D: mean={exp.mean().item():.6f} std={exp.std().item():.6f} "
                  f"range=[{exp.min().item():.4f}, {exp.max().item():.4f}]")
            print(f"[元学习调制] 32D: L2={meta.norm().item():.4f} mean={meta.mean().item():.6f} "
                  f"range=[{meta.min().item():.4f}, {meta.max().item():.4f}]")

            # 采样还原
            print(f"\n[还原采样]")
            for wv, roles in encoded[:3]:
                with torch.no_grad():
                    svec = p5(wv, roles, last_loss=0.0)
                    pred = p6(svec.unsqueeze(0), last_loss=0.0)
                true = wv
                cos_list = []
                for i in range(3):
                    cs = F.cosine_similarity(pred[0,i:i+1], true[i:i+1], dim=-1).item()
                    cos_list.append(f"{cs:.4f}")
                print(f"  词cos: [{', '.join(cos_list)}]")

            a, r, m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB res={r:.0f}MB peak={m:.0f}MB | {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f}")

    # 最终
    print(f"\n{'#'*60}")
    print("# P6 1000轮最终还原测试")
    all_cos = []
    for wv, roles in encoded[:8]:
        with torch.no_grad():
            svec = p5(wv, roles, last_loss=0.0)
            pred = p6(svec.unsqueeze(0), last_loss=0.0)
        true = wv
        cs = [F.cosine_similarity(pred[0,i:i+1], true[i:i+1], dim=-1).item() for i in range(3)]
        all_cos.extend(cs)
        print(f"  [{cs[0]:.4f}, {cs[1]:.4f}, {cs[2]:.4f}]")
    avg = sum(all_cos)/len(all_cos)
    print(f"  平均: {avg:.4f}")
    print(f"  验收: cos>0.9 → {'PASS' if avg>0.9 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
