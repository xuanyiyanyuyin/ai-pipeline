"""
P8 字符序列→句子 (与P5词级对齐)
==================================
同一句子的字符序列 → P8 → 句子向量
同一句子的词序列   → P5 → 句子向量
Loss: cos(P8输出, P5输出)
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P8_char_sent.model import CharToSent


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
    ep = 300; disp = 50; lr = 0.005
    print(f"P8 字符→句子 epochs={ep} display={disp} lr={lr}\n")

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

    # 数据
    sentences = load_sentences("C:/ai/P5_sentence/sentences.txt")
    print(f"[数据] {len(sentences)}句")

    # 编码
    @torch.no_grad()
    def enc_word(w):
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    @torch.no_grad()
    def enc_char(c):
        """单字符: 直接用P1的char_content + 固定pos0"""
        if c not in p1_c2i: return None
        idx = p1_c2i[c]
        content = p1.char_content(torch.tensor([idx], device=DEVICE))  # [1, 24]
        pos = p1.pos_encoder.pe[0:1]  # [1, 8] — 位置0
        return torch.cat([pos[0], content[0]])  # [32]

    encoded = []
    for words, roles in sentences:
        # 词级编码
        wvecs = [enc_word(w) for w in words]
        if None in wvecs: continue
        # 字符级编码: 把每个字拆开
        chars = list("".join(words))
        cvecs = [enc_char(c) for c in chars]
        if None in cvecs: continue
        encoded.append((torch.stack(wvecs), torch.stack(cvecs),
                        torch.tensor(roles, device=DEVICE)))
    print(f"[编码] {len(encoded)}句有效")

    # P8
    model = CharToSent(max_len=12).to(DEVICE)
    print(f"[P8] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; best_cos = 0.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0

        for wvecs, cvecs, roles in encoded:
            # P8: 字符→句子
            s_char = model(cvecs, last_loss=last_loss)

            # P5: 词→句子 (目标)
            with torch.no_grad():
                s_word = p5(wvecs, roles, last_loss=0.0)

            loss = 1.0 - F.cosine_similarity(s_char.unsqueeze(0), s_word.unsqueeze(0), dim=-1)
            cos = 1.0 - loss.item()

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += cos; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P8] Epoch {epoch:4d} / {ep} | Loss: {al:.6f} | cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")
            print(f"{'='*70}")

            # 调制
            exp = model.explore_state
            meta = model.meta_fc(exp * min(last_loss*20.0, 1.0))
            print(f"[调制] L2={meta.norm().item():.4f}")

            # 采样
            print(f"\n[采样] 字符序列 → 句子向量 cos vs P5(词级):")
            for wvecs, cvecs, roles in encoded[:3]:
                s_char = model(cvecs, last_loss=0.0)
                with torch.no_grad():
                    s_word = p5(wvecs, roles, last_loss=0.0)
                cs = F.cosine_similarity(s_char.unsqueeze(0), s_word.unsqueeze(0), dim=-1).item()
                chars_str = " ".join(["".join(words) for words in [["?"]]])  # placeholder
                print(f"  cos(字符→句子, 词→句子) = {cs:.4f}")

            a,r,m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB | {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 25 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f}")

    # 最终
    print(f"\n{'#'*60}")
    cos_list = []
    for wvecs, cvecs, roles in encoded:
        s_char = model(cvecs, last_loss=0.0)
        with torch.no_grad():
            s_word = p5(wvecs, roles, last_loss=0.0)
        cs = F.cosine_similarity(s_char.unsqueeze(0), s_word.unsqueeze(0), dim=-1).item()
        cos_list.append(cs)
    avg = sum(cos_list) / len(cos_list)
    print(f"# P8 平均cos: {avg:.4f}")
    print(f"# 验收: cos>0.9 → {'PASS' if avg>0.9 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
