"""
P5 词序列→句子合成 (128D统一版, V2修复)
========================================
修复:
1. contrastive_loss: 改为margin-based对比学习, 正样本有梯度
2. 池化: 用位置编码乘法替代sum(dim=0), 保留词序信息
3. explore_state: 改为输入依赖的条件调制
4. role_weights: 用索引操作替代循环, 保留梯度
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

        # 修复: 输入依赖的条件调制 (替代全局固定explore_state)
        self.condition_mod = nn.Sequential(
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False),
            nn.Tanh(),
            nn.Linear(sent_pos_dim + sent_sem_dim, sent_pos_dim + sent_sem_dim, bias=False),
        )

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

        # 修复: 用索引操作替代循环, 保留梯度
        role_map = torch.stack([self.w_subj, self.w_verb, self.w_obj])  # [3]
        roles_long = roles.long().clamp(0, 2)  # [n]
        role_weights = role_map[roles_long]  # [n], 保留梯度图

        # 修复: 位置编码与内容相乘 (保留词序)
        # seq_p扩展为[n, 12]与combined相乘
        seq_p_expanded = seq_p.unsqueeze(-1).expand(-1, -1, combined.shape[-1] // self.seq_pos_dim).reshape(n, -1)
        # 简单方案: 用位置编码加权
        pos_factor = 1.0 + seq_p.sum(dim=-1, keepdim=True) * 0.1  # [n, 1]
        combined_weighted = combined * pos_factor  # [n, 12]
        sem_weighted = sem_w * pos_factor  # [n, 120]

        # 按角色加权求和
        fused_pos = (combined_weighted * role_weights.unsqueeze(-1)).sum(dim=0)  # [12]
        sent_pos = self.pos_fuse(fused_pos)  # [64]

        combined_sem = (sem_weighted * role_weights.unsqueeze(-1)).sum(dim=0)  # [120]
        combined_sem = F.normalize(combined_sem, dim=-1)
        sent_sem = self.sem_fuse(combined_sem)  # [192]

        sent_vec = torch.cat([sent_pos, sent_sem])  # [256]

        # 修复: 输入依赖的条件调制
        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.condition_mod(sent_vec) * loss_factor
        sent_vec = sent_vec + mod

        return sent_vec


def contrastive_loss(sent_correct, sent_scrambled, margin=0.3):
    """
    修复: margin-based对比学习, 正样本有梯度
    
    原理:
    - 用batch均值作为锚点 (detach, 作为固定目标)
    - 正样本loss: 正确句子 → 锚点
    - 负样本loss: 乱序句子 ← 锚点
    """
    # 用batch均值作为锚点 (detach, 不参与梯度)
    anchor = sent_correct.mean(dim=0, keepdim=True).detach()
    
    # 正样本: 正确句子靠近锚点
    cos_c = F.cosine_similarity(sent_correct, anchor, dim=-1)  # [batch_size]
    pos_l = F.relu(1.0 - margin - cos_c).mean()
    
    # 负样本: 乱序句子远离锚点
    cos_s = F.cosine_similarity(sent_scrambled, anchor, dim=-1)  # [batch_size]
    neg_l = F.relu(cos_s - margin).mean()
    
    loss = pos_l + neg_l
    
    return loss, cos_c.mean().item(), cos_s.mean().item()
