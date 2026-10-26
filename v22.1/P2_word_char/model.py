"""
P2 词语→字符 反向解码器 (128D统一版)
=====================================
输入: P1输出的128D词语向量
输出: 还原的 字1(128D) + 字2(128D)

架构: 128D → Shared(256D→256D) → split → Char1 Head(256→128) + Char2 Head(256→128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordToCharDecoder(nn.Module):
    def __init__(self, word_dim=128, hidden_dim=256):
        super().__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(word_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.c1_head = nn.Linear(hidden_dim, word_dim)
        self.c2_head = nn.Linear(hidden_dim, word_dim)

        # 探索区 + 元学习区 (128D, P2自控调制强度)
        self.explore_state = nn.Parameter(torch.randn(word_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(word_dim, word_dim, bias=False))
        self.mod_strength = nn.Parameter(torch.tensor(0.05))  # P2自己学需要多强

    def forward(self, word_vector, last_loss=1.0):
        h = self.shared(word_vector)  # [b, 256]
        c1 = self.c1_head(h)          # [b, 128]
        c2 = self.c2_head(h)          # [b, 128]

        # Normalize: 余弦损失只用方向
        c1 = F.normalize(c1, dim=-1)
        c2 = F.normalize(c2, dim=-1)

        # 探索区 + 元学习区 — P2自己学会调制该多强
        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        mod = mod * self.mod_strength  # 可学习缩放
        c1 = F.normalize(c1 + mod.unsqueeze(0).expand(c1.shape[0], -1), dim=-1)
        c2 = F.normalize(c2 + mod.unsqueeze(0).expand(c2.shape[0], -1), dim=-1)

        return c1, c2


def load_p1_frozen(p1_model_class, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    p1 = p1_model_class(ckpt["num_chars"], ckpt["num_words"]).to(device)
    p1.load_state_dict(ckpt["model_state_dict"])
    for param in p1.parameters():
        param.requires_grad = False
    p1.eval()
    return p1, ckpt


def cosine_loss(pred_c1, pred_c2, real_c1, real_c2):
    sim1 = F.cosine_similarity(pred_c1, real_c1, dim=-1)
    sim2 = F.cosine_similarity(pred_c2, real_c2, dim=-1)
    avg_sim = (sim1 + sim2) / 2.0
    return (1.0 - avg_sim).mean(), sim1.mean().item(), sim2.mean().item()
