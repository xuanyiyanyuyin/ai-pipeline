"""
P5 词序列→句子合成 (128D统一版)
================================
每词角色加权±融合 + 句位置编码 → 256D句子向量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SentenceSynthesis(nn.Module):
    def __init__(self, word_dim=128, pos_dim=8, sem_dim=120,
                 seq_pos_dim=4, sent_pos_dim=64, sent_sem_dim=192):
        super().__init__()
        self.pos_dim = pos_dim
        self.sem_dim = sem_dim
        self.seq_pos_dim = seq_pos_dim
        self.sent_pos_dim = sent_pos_dim
        self.sent_sem_dim = sent_sem_dim
        self.combined_pos_dim = pos_dim + seq_pos_dim  # 12

        self.w_subj = nn.Parameter(torch.tensor(1.0))
        self.w_verb = nn.Parameter(torch.tensor(1.0))
        self.w_obj  = nn.Parameter(torch.tensor(1.0))

        self.register_buffer('seq_pe', self._build_seq_pe(20))

        self.pos_fuse = nn.Linear(self.combined_pos_dim, sent_pos_dim, bias=False)  # 12→64
        self.sem_fuse = nn.Linear(sem_dim, sent_sem_dim, bias=False)                 # 120→192

        self.explore_state = nn.Parameter(torch.randn(sent_pos_dim + sent_sem_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False))

    def _build_seq_pe(self, max_len):
        pe = torch.zeros(max_len, self.seq_pos_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.seq_pos_dim, 2).float() * (-math.log(100.0) / self.seq_pos_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, word_vecs, roles, last_loss=1.0):
        n = word_vecs.shape[0]
        device = word_vecs.device

        pos_w = word_vecs[:, :self.pos_dim]   # [n, 8]
        sem_w = word_vecs[:, self.pos_dim:]   # [n, 120]

        seq_idx = torch.arange(n, device=device).clamp(0, self.seq_pe.shape[0] - 1)
        seq_p = self.seq_pe[seq_idx]  # [n, 4]
        combined = torch.cat([pos_w, seq_p], dim=-1)  # [n, 12]

        fused_pos = torch.zeros(self.combined_pos_dim, device=device)
        for i in range(n):
            r = roles[i].item() if torch.is_tensor(roles[i]) else roles[i]
            p = combined[i]
            if r == 0:    fused_pos = fused_pos + p * self.w_subj
            elif r == 1:  fused_pos = fused_pos + p * self.w_verb
            elif r == 2:  fused_pos = fused_pos - p * self.w_obj
            else:         fused_pos = fused_pos + p

        sent_pos = self.pos_fuse(fused_pos)  # [64]
        combined_sem = sem_w.sum(dim=0)
        combined_sem = F.normalize(combined_sem, dim=-1)
        sent_sem = self.sem_fuse(combined_sem)  # [192]

        sent_vec = torch.cat([sent_pos, sent_sem])  # [256]

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        sent_vec = sent_vec + mod

        return sent_vec


def contrastive_loss(sent_correct, sent_scrambled, margin=0.3):
    ref = sent_correct.detach()
    cos_c = F.cosine_similarity(sent_correct, ref, dim=-1)
    cos_s = F.cosine_similarity(sent_scrambled, ref.unsqueeze(0).expand(sent_scrambled.shape[0], -1), dim=-1).mean()
    pos_l = F.relu(1.0 - margin - cos_c)
    neg_l = F.relu(cos_s - (cos_c - margin))
    return pos_l + neg_l, cos_c.item(), cos_s.item()
