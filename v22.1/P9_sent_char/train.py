"""
P9 句子→字符序列 反向解码 (P8的反向)
======================================
P5句子向量 → 还原各字符向量 → cos vs P1字符嵌入
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P9_sent_char.model import SentToChars


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


def train():
    ep = 200; disp = 50; lr = 0.003
    print(f"P9 句子→字符 epochs={ep} display={disp} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # P5
    p5 = SentenceSynthesis().to(DEVICE)
    p5_path = os.path.join(SAVE_DIR, "P5_final.pt")
    if os.path.exists(p5_path):
        p5_ckpt = torch.load(p5_path, map_location=DEVICE)
        p5.load_state_dict(p5_ckpt.get("model_state_dict", p5_ckpt.get("model", {})))
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

    sentences = load_sentences("C:/ai/P5_sentence/sentences.txt")
    print(f"[数据] {len(sentences)}句")

    @torch.no_grad()
    def enc_word(w):
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    @torch.no_grad()
    def enc_char(c):
        if c not in p1_c2i: return None
        content = p1.char_content(torch.tensor([p1_c2i[c]], device=DEVICE))
        pos = p1.pos_encoder.pe[0:1]
        return torch.cat([pos[0], content[0]])

    encoded = []
    for words, roles in sentences:
        wvecs = [enc_word(w) for w in words]
        if None in wvecs: continue
        chars = list("".join(words))
        cvecs = [enc_char(c) for c in chars]
        if None in cvecs: continue
        encoded.append((torch.stack(wvecs), torch.stack(cvecs),
                        torch.tensor(roles, device=DEVICE), len(chars)))
    print(f"[编码] {len(encoded)}句有效")

    # P9
    model = SentToChars(max_chars=10).to(DEVICE)
    print(f"[P9] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0

        for wvecs, cvecs, roles, nc in encoded:
            with torch.no_grad():
                svec = p5(wvecs, roles, last_loss=0.0)  # [64]

            pred = model(svec.unsqueeze(0), nc, last_loss=last_loss)  # [1, nc, 32]
            true = cvecs.unsqueeze(0)  # [1, nc, 32]
            sims = F.cosine_similarity(pred, true, dim=-1)
            loss = (1.0 - sims).mean()
            cos = sims.mean().item()

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += cos; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P9] Epoch {epoch:4d} / {ep} | Loss: {al:.6f} | cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")
            # 采样
            for wvecs, cvecs, roles, nc in encoded[:2]:
                with torch.no_grad():
                    svec = p5(wvecs, roles, last_loss=0.0)
                    pred = model(svec.unsqueeze(0), nc, last_loss=0.0)
                true = cvecs.unsqueeze(0)
                char_cos = [f"{F.cosine_similarity(pred[0,i:i+1], true[0,i:i+1], dim=-1).item():.3f}" for i in range(nc)]
                print(f"  字符还原cos: {char_cos}")
            print(f"{'='*70}\n")
        elif epoch % 25 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f}")

    print(f"\n{'#'*60}")
    coses = []
    for wvecs, cvecs, roles, nc in encoded:
        with torch.no_grad():
            svec = p5(wvecs, roles, last_loss=0.0)
            pred = model(svec.unsqueeze(0), nc, last_loss=0.0)
        true = cvecs.unsqueeze(0)
        sims = F.cosine_similarity(pred, true, dim=-1)
        coses.append(sims.mean().item())
    avg = sum(coses)/len(coses)
    print(f"# P9 平均还原cos: {avg:.4f}")
    print(f"# 验收: cos>0.7 → {'PASS' if avg>0.7 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
