"""
P6桥接训练: P8(字符→句子)输出 → P6解码 → P1词向量
=====================================================
关键: P8和P5都生成句子向量作为P6输入, 消除域偏移
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P8_char_sent.model import CharToSent
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


def train():
    ep = 500; disp = 50; lr = 0.003
    print(f"P6桥接训练 epochs={ep} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # P5
    p5 = SentenceSynthesis().to(DEVICE)
    p5p = os.path.join(SAVE_DIR, "P5_final.pt")
    if os.path.exists(p5p):
        p5.load_state_dict(torch.load(p5p, map_location=DEVICE).get("model_state_dict", {}))
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

    # P8
    p8 = CharToSent(max_len=12).to(DEVICE)
    p8p = os.path.join(SAVE_DIR, "P8_final.pt")
    if os.path.exists(p8p):
        p8.load_state_dict(torch.load(p8p, map_location=DEVICE).get("model_state_dict", {}))
    for p in p8.parameters(): p.requires_grad = False; p8.eval()

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

    # 编码: 词级 + 字符级
    encoded = []
    for words, roles in sentences:
        wv = [enc_word(w) for w in words]
        if None in wv or len(wv) != 3: continue
        chars = list("".join(words))
        cv = [enc_char(c) for c in chars]
        if None in cv: continue
        encoded.append((torch.stack(wv), torch.stack(cv), torch.tensor(roles, device=DEVICE)))

    print(f"[编码] {len(encoded)}句")

    # P6 初始化
    p6 = SentToWordsDecoder().to(DEVICE)
    if os.path.exists(os.path.join(SAVE_DIR, "P6_final.pt")):
        p6.load_state_dict(torch.load(os.path.join(SAVE_DIR, "P6_final.pt"),
                          map_location=DEVICE).get("model_state_dict", {}))
    opt = torch.optim.Adam(p6.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0

        for wv, cv, roles in encoded:
            # P8: 字符→句子向量
            with torch.no_grad():
                svec_p8 = p8(cv, last_loss=0.0)
                svec_p5 = p5(wv, roles, last_loss=0.0)

            # P6训练: 随机选P8或P5输入
            if epoch % 2 == 0:
                svec = svec_p5  # P5
            else:
                svec = svec_p8  # P8 (桥接!)

            pred = p6(svec.unsqueeze(0), last_loss=last_loss)
            true = wv.unsqueeze(0)
            mask = torch.ones(1, 3, device=DEVICE)
            loss, avg_cos = decode_loss(pred, true, mask)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += avg_cos; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; ac = epoch_cos / nb

        if epoch % disp == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"[P6-Bridge] Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")

    # 保存
    torch.save({"model_state_dict": p6.state_dict(), "avg_cos": ac},
               os.path.join(SAVE_DIR, "P6_bridge.pt"))
    print(f"[保存] P6_bridge.pt | avg_cos={ac:.4f}")


if __name__ == "__main__":
    train()
