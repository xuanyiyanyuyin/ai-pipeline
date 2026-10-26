"""
P8 字符序列→句子 (128D统一版)
===============================
输入: 字符序列 [c1,...,cn] 各128D
输出: 256D句子向量
训练目标: 与P5词级句子向量余弦对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CharToSent(nn.Module):
    def __init__(self, char_dim=128, seq_pos_dim=4, sent_pos_dim=64, sent_sem_dim=192, max_len=15):
        super().__init__()
        self.char_dim = char_dim
        self.sent_dim = sent_pos_dim + sent_sem_dim

        self.register_buffer('seq_pe', self._build_pe(max_len, seq_pos_dim))
        self.seq_pos_dim = seq_pos_dim

        self.pos_fuse = nn.Linear(char_dim + seq_pos_dim, sent_pos_dim, bias=False)  # 132→64
        self.sem_fuse = nn.Linear(char_dim, sent_sem_dim, bias=False)                # 128→192

        self.explore_state = nn.Parameter(torch.randn(sent_pos_dim + sent_sem_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False))

    def _build_pe(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, char_vecs, last_loss=1.0):
        n = char_vecs.shape[0]
        device = char_vecs.device

        seq_idx = torch.arange(n, device=device).clamp(0, self.seq_pe.shape[0] - 1)
        seq_p = self.seq_pe[seq_idx]  # [n, 4]
        combined = torch.cat([char_vecs, seq_p], dim=-1)  # [n, 132]

        fused_pos_raw = combined.sum(dim=0)
        sent_pos = self.pos_fuse(fused_pos_raw[:self.char_dim + self.seq_pos_dim])  # [64]

        fused_sem_raw = char_vecs.sum(dim=0)
        sent_sem = self.sem_fuse(fused_sem_raw)  # [192]

        sent_vec = torch.cat([sent_pos, sent_sem])  # [256]

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        sent_vec = sent_vec + mod

        return sent_vec
