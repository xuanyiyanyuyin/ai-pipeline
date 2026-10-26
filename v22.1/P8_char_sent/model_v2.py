"""
P8 字符序列→句子 (128D统一版, V2修复)
======================================
修复:
1. 池化: 用位置编码乘法替代sum(dim=0), 保留字符序信息
2. explore_state: 改为输入依赖的条件调制
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

        # 修复: 输入依赖的条件调制
        self.condition_mod = nn.Sequential(
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False),
            nn.Tanh(),
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False),
        )

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
        
        # 修复: 位置编码与内容相乘 (保留字符序)
        pos_factor = 1.0 + seq_p.sum(dim=-1, keepdim=True) * 0.1  # [n, 1]
        char_vecs_weighted = char_vecs * pos_factor  # [n, 128]
        
        combined = torch.cat([char_vecs_weighted, seq_p], dim=-1)  # [n, 132]

        # 修复: 加权求和替代sum(dim=0)
        fused_pos_raw = (combined * pos_factor).sum(dim=0)  # [132]
        sent_pos = self.pos_fuse(fused_pos_raw[:self.char_dim + self.seq_pos_dim])  # [64]

        fused_sem_raw = (char_vecs_weighted * pos_factor).sum(dim=0)  # [128]
        sent_sem = self.sem_fuse(fused_sem_raw)  # [192]

        sent_vec = torch.cat([sent_pos, sent_sem])  # [256]

        # 修复: 输入依赖的条件调制
        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.condition_mod(sent_vec) * loss_factor
        sent_vec = sent_vec + mod

        return sent_vec
