"""
P9 统一管线训练 (1000轮, 精简版)
==================================
字符序列 → 句子向量 + 词向量
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P9_pipeline.model import UnifiedPipeline, cosine_loss


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
    ep = 1000; disp = 100; lr = 0.003
    print(f"P9统一管线 epochs={ep} lr={lr}\n")

    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    c2i = p1_ckpt["char2idx"]

    sentences = load_sentences("C:/ai/P5_sentence/sentences.txt")
    print(f"[数据] {len(sentences)}句")

    @torch.no_grad()
    def enc_w(w):
        c1, c2 = w[0], w[0] if len(w)==1 else w[1]
        if c1 not in c2i or c2 not in c2i: return None
        return p1(torch.tensor([[c2i[c1],c2i[c2]]],device=DEVICE),last_loss=0.0)[0]

    @torch.no_grad()
    def enc_c(c):
        if c not in c2i: return None
        ct = p1.char_content(torch.tensor([c2i[c]],device=DEVICE)); ps = p1.pos_encoder.pe[0:1]
        return torch.cat([ps[0], ct[0]])

    encoded = []
    for words, roles in sentences:
        wv = [enc_w(w) for w in words]
        if None in wv or len(wv)!=3: continue
        cv = [enc_c(c) for c in list("".join(words))]
        if None in cv: continue
        encoded.append((torch.stack(wv), torch.stack(cv), torch.tensor(roles,device=DEVICE)))

    print(f"[编码] {len(encoded)}句")
    model = UnifiedPipeline(max_words=3).to(DEVICE)
    print(f"[P9] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; total_t0 = time.time()

    for epoch in range(1, ep+1):
        t0 = time.time(); el = 0.0; ec = 0.0; nb = 0
        for wv, cv, roles in encoded:
            sent, wvp = model(cv.unsqueeze(0), last_loss=last_loss)
            mask = torch.ones(1, 3, device=DEVICE)
            loss, cs = cosine_loss(wvp, wv.unsqueeze(0), mask)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); ec += cs; nb += 1; last_loss = loss.item()

        al = el/nb; ac = ec/nb
        if epoch % disp == 0 or epoch == 1:
            elapsed = time.time()-t0
            print(f"\n{'='*70}")
            print(f"[P9] Epoch {epoch:4d}/{ep} | Loss: {al:.6f} | cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")
            # 采样
            wv, cv, roles = encoded[0]
            with torch.no_grad():
                sent, wvp = model(cv.unsqueeze(0), last_loss=0.0)
            cs_list = [f"{F.cosine_similarity(wvp[0,i:i+1],wv[i:i+1],dim=-1).item():.3f}" for i in range(3)]
            print(f"  词还原cos: [{', '.join(cs_list)}]")
            # 调制
            mod = model.meta(model.explore * min(last_loss*20,1.0))
            print(f"[调制] L2={mod.norm().item():.4f}")
            print(f"[GPU] {torch.cuda.memory_allocated()//1048576}MB | {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f}")

    print(f"\n{'#'*60}\n# P9完成 | 最终cos: {ac:.4f}\n{'#'*60}")


if __name__ == "__main__":
    train()
