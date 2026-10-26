"""
P2 词语→字符 反向解析 训练脚本
================================
加载冻结的P1 → P2解码器还原字符向量
Loss: 1 - (cos_sim(c1)+cos_sim(c2))/2
验收: 平均余弦相似度 > 90%
"""
import torch
import torch.nn.functional as F
import time, os, sys, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P2_word_char.model import WordToCharDecoder, load_p1_frozen, cosine_loss


# ==================== 交互设置 ====================

def get_user_settings():
    print(f"\n{'='*50}")
    print(f"P2 词语->字符 反向解码 训练设置")
    print(f"{'='*50}")

    try:
        s = input(f"  训练轮次 (默认100): ").strip()
        epochs = int(s) if s else 100
    except ValueError:
        epochs = 100

    try:
        s = input(f"  每多少轮显示一次日志 (默认10): ").strip()
        disp = int(s) if s else 10
    except ValueError:
        disp = 10

    try:
        s = input(f"  学习率 (默认0.001): ").strip()
        lr = float(s) if s else 0.001
    except ValueError:
        lr = 0.001

    print(f"\n  epochs={epochs} display={disp} lr={lr}")
    print(f"{'='*50}\n")
    return epochs, disp, lr


# ==================== 数据 ====================

def load_word_list(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = re.match(r'!([^@]+)@(.)@(.)', line)
            if m:
                entries.append((m.group(1), m.group(2), m.group(3)))
            elif len(line) == 2:
                entries.append((line, line[0], line[1]))
    return entries


def entries_to_pairs(entries, char2idx):
    return [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]


# ==================== 日志 ====================

def gpu_mem():
    if DEVICE.type != "cuda": return (0, 0, 0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2,
            torch.cuda.memory_reserved(DEVICE)/1024**2,
            torch.cuda.max_memory_allocated(DEVICE)/1024**2)


def log_step(epoch, loss, sim1, sim2, word, idx2char, pair, pred_c1, pred_c2, real_c1, real_c2, t_ms):
    c1_raw, c2_raw = pair[0, 0].item(), pair[0, 1].item()
    print(f"\n{'─'*65}")
    print(f"[P2] Epoch {epoch:4d} | Loss: {loss:.6f} | "
          f"字1余弦: {sim1:.3f} | 字2余弦: {sim2:.3f} | 平均: {(sim1+sim2)/2:.3f}")
    print(f"  输入词: '{word}' ({idx2char[c1_raw]}+{idx2char[c2_raw]})")
    print(f"  还原字1  vs 原始字1: cos={sim1:.4f}")
    print(f"  还原字2  vs 原始字2: cos={sim2:.4f}")
    print(f"  还原字1 前8D(pos): [{', '.join(f'{v:.2f}' for v in pred_c1[0,:8].tolist())}]")
    print(f"  原始字1 前8D(pos): [{', '.join(f'{v:.2f}' for v in real_c1[0,:8].tolist())}]")
    a, r, m = gpu_mem()
    print(f"  GPU: alloc={a:.1f}MB res={r:.1f}MB peak={m:.1f}MB | {t_ms:.1f}ms")
    print(f"{'─'*65}")


# ==================== 训练 ====================

def train():
    epochs, disp_interval, lr = get_user_settings()

    # P1 checkpoint
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    if not os.path.exists(p1_path):
        p1_path = os.path.join(SAVE_DIR, "P1_final.pt")
    print(f"[P1] 加载: {p1_path}")

    # 数据
    entries = load_word_list(WORD_LIST_PATH)
    _, char2idx, idx2char, word2idx, idx2word = {}, {}, {}, {}, {}
    char_set = set()
    for _, c1, c2 in entries:
        char_set.add(c1); char_set.add(c2)
    char_list = sorted(char_set)
    char2idx = {c: i for i, c in enumerate(char_list)}
    idx2char = {i: c for c, i in char2idx.items()}
    all_pairs = [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]
    num_words = len(entries)
    print(f"[数据] {num_words}个词 | {len(char_list)}个唯一字符")

    # 加载冻结P1
    p1, ckpt = load_p1_frozen(CharToWordModel, p1_path, DEVICE)
    print(f"[P1] 已冻结 | 最佳Top-1: {ckpt.get('top1', '?')}")

    # P2解码器
    p2 = WordToCharDecoder().to(DEVICE)
    print(f"[P2] 参数: {sum(p.numel() for p in p2.parameters()):,}")

    optimizer = torch.optim.Adam(p2.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # 训练
    last_loss = 1.0; best_avg_sim = 0.0; n = num_words
    for epoch in range(1, epochs + 1):
        t0 = time.time(); epoch_loss = 0.0; nb = 0
        epoch_sim1 = 0.0; epoch_sim2 = 0.0
        perm = torch.randperm(n)

        for bs in range(0, n, BATCH_SIZE):
            be = min(bs + BATCH_SIZE, n)
            idxs = perm[bs:be]

            pair_ids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)

            # P1前向(冻结) → 词语向量
            with torch.no_grad():
                _, _, full = p1.get_char_vectors(pair_ids)
                # full[b,2,128]: 原始字符向量 (ground truth)
                real_c1 = full[:, 0, :]  # [b, 32]
                real_c2 = full[:, 1, :]  # [b, 32]

                word_vec = p1(pair_ids, last_loss=last_loss)  # [b, 32]

            # P2前向 → 还原字符
            t1 = time.time()
            pred_c1, pred_c2 = p2(word_vec)
            loss, sim1, sim2 = cosine_loss(pred_c1, pred_c2, real_c1, real_c2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_ms = (time.time() - t1) * 1000
            epoch_loss += loss.item(); nb += 1
            epoch_sim1 += sim1; epoch_sim2 += sim2
            last_loss = loss.item()

        avg_loss = epoch_loss / nb
        avg_sim1 = epoch_sim1 / nb
        avg_sim2 = epoch_sim2 / nb
        avg_sim = (avg_sim1 + avg_sim2) / 2
        elapsed = time.time() - t0

        if epoch % disp_interval == 0 or epoch == 1 or epoch == epochs:
            log_step(epoch, avg_loss, avg_sim1, avg_sim2,
                     entries[idxs[0].item()][0], idx2char, pair_ids[0:1],
                     pred_c1, pred_c2, real_c1, real_c2, t_ms)

        # 简短日志
        if epoch % disp_interval != 0:
            print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | "
                  f"cos_c1: {avg_sim1:.3f} cos_c2: {avg_sim2:.3f} | avg: {avg_sim:.3f} | {elapsed:.1f}s")

        if avg_sim > best_avg_sim:
            best_avg_sim = avg_sim
            torch.save({
                "epoch": epoch, "model_state_dict": p2.state_dict(),
                "avg_cos_sim": avg_sim, "cos_c1": avg_sim1, "cos_c2": avg_sim2,
            }, os.path.join(SAVE_DIR, "P2_best.pt"))

    # 最终
    print(f"\n{'#'*60}")
    print(f"# P2训练完成 | 最佳平均余弦: {best_avg_sim:.2%}")
    print(f"# 验收: >90% → {'PASS' if best_avg_sim>=0.90 else 'FAIL'}")
    print(f"{'#'*60}")

    torch.save({
        "epoch": epochs, "model_state_dict": p2.state_dict(),
        "avg_cos_sim": avg_sim,
    }, os.path.join(SAVE_DIR, "P2_final.pt"))
    print(f"[保存] {SAVE_DIR}/P2_final.pt")


if __name__ == "__main__":
    train()
