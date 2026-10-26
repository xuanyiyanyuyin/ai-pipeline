"""
P6 句子→词序列 反向解码 (位置嵌入V6)
==========================================
输入: 句子向量(256D)
输出: 词序列 (各128D)

架构: 16个平行头, 每头接收h+P_i (位置嵌入)
  head[i](h + pos_embed[i]) → word_i
  位置嵌入提供唯一起点, 天然防重复, 无需残差减法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentToWordsDecoder(nn.Module):
    def __init__(self, sent_dim=256, word_dim=128, hidden_dim=256, max_words=16):
        super().__init__()
        self.max_words = max_words

        self.encoder = nn.Sequential(
            nn.Linear(sent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )

        # 位置嵌入: 每位置独立向量, 提供唯一起点
        self.pos_embed = nn.Parameter(torch.randn(max_words, hidden_dim) * 2.0)

        # 平行提取头
        self.extract = nn.ModuleList([
            nn.Linear(hidden_dim, word_dim) for _ in range(max_words)
        ])
        for head in self.extract:
            nn.init.xavier_uniform_(head.weight, gain=0.1)

    def forward(self, sent_vec, last_loss=1.0, gate=None):
        b = sent_vec.shape[0]
        h = self.encoder(sent_vec)  # [b, 256]

        # gate调控: explore→meta学习何时开/关编码维度
        if gate is not None:
            h = h * gate  # gate already [b,256] or [256], broadcast works

        words = []
        for i in range(self.max_words):
            hi = h + self.pos_embed[i].unsqueeze(0)
            w = self.extract[i](hi)
            words.append(w)

        return torch.stack(words, dim=1)  # [b, max_words, 128]


def decode_loss(pred_words, true_words, mask):
    sims = F.cosine_similarity(pred_words, true_words, dim=-1)
    masked_sims = sims * mask
    n_valid = mask.sum() + 1e-8
    avg_sim = masked_sims.sum() / n_valid
    return (1.0 - avg_sim), avg_sim.item()
