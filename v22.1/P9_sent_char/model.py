"""
P9 句子→字符序列 反向解码 (128D统一版)
========================================
输入: 句子向量(256D)
输出: 还原各字符向量 [c1(128D), ..., cn(128D)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentToChars(nn.Module):
    def __init__(self, sent_dim=256, char_dim=128, hidden_dim=256, max_chars=10):
        super().__init__()
        self.max_chars = max_chars
        self.char_dim = char_dim

        self.encoder = nn.Sequential(
            nn.Linear(sent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.char_heads = nn.ModuleList([
            nn.Linear(hidden_dim, char_dim) for _ in range(max_chars)
        ])

        self.explore_state = nn.Parameter(torch.randn(char_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(char_dim, char_dim, bias=False))

    def forward(self, sent_vec, n_chars, last_loss=1.0):
        b = sent_vec.shape[0]
        h = self.encoder(sent_vec)  # [b, 256]

        chars = []
        for i in range(n_chars):
            w = self.char_heads[i](h)  # [b, 128]
            loss_factor = min(last_loss * 20.0, 1.0)
            mod = self.meta_fc(self.explore_state * loss_factor)
            w = w + mod.unsqueeze(0).expand(b, -1)
            chars.append(w)

        return torch.stack(chars, dim=1)  # [b, n_chars, 128]
