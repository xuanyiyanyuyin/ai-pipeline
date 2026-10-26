"""
P9 统一管线: 字符序列→句子→词向量 (128D统一版)
================================================
输入: 字符序列 [c1,...,cn] 各128D
输出: sentence_vec(256D), word_vecs[n_words, 128D]
"""
import torch, torch.nn as nn, torch.nn.functional as F, math


class UnifiedPipeline(nn.Module):
    def __init__(self, char_dim=128, word_dim=128, sent_dim=256, max_words=5,
                 seq_pos_dim=4, hidden_dim=256):
        super().__init__()
        self.max_words = max_words
        self.char_dim = char_dim

        self.register_buffer('seq_pe', self._build_pe(15, seq_pos_dim))
        self.seq_pos_dim = seq_pos_dim
        self.pos_fuse = nn.Linear(char_dim + seq_pos_dim, sent_dim // 2, bias=False)  # 132→128
        self.sem_fuse = nn.Linear(char_dim, sent_dim // 2, bias=False)                # 128→128

        self.word_encoder = nn.Sequential(
            nn.Linear(sent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.word_heads = nn.ModuleList([nn.Linear(hidden_dim, word_dim) for _ in range(max_words)])

        self.explore = nn.Parameter(torch.randn(sent_dim) * 1.0)
        self.meta = nn.Sequential(nn.Linear(sent_dim, sent_dim, bias=False))

    def _build_pe(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, char_vecs, last_loss=1.0):
        b, nc, _ = char_vecs.shape

        seq_idx = torch.arange(nc, device=char_vecs.device).clamp(0, self.seq_pe.shape[0]-1)
        seq_p = self.seq_pe[seq_idx].unsqueeze(0).expand(b, -1, -1)
        combined = torch.cat([char_vecs, seq_p], dim=-1)
        fused = combined.sum(dim=1)
        sent_pos = self.pos_fuse(fused[:, :self.char_dim+self.seq_pos_dim])
        sent_sem = self.sem_fuse(char_vecs.sum(dim=1))
        sent_vec = torch.cat([sent_pos, sent_sem], dim=-1)

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta(self.explore * loss_factor)
        sent_vec = sent_vec + mod.unsqueeze(0).expand(b, -1)

        h = self.word_encoder(sent_vec)
        word_vecs = torch.stack([head(h) for head in self.word_heads], dim=1)

        return sent_vec, word_vecs


def cosine_loss(word_vecs_pred, true_word_vecs, mask):
    nw = min(word_vecs_pred.shape[1], true_word_vecs.shape[1])
    sims = F.cosine_similarity(word_vecs_pred[:, :nw, :], true_word_vecs[:, :nw, :], dim=-1)
    masked = (sims * mask).sum() / (mask.sum() + 1e-8)
    return 1.0 - masked, masked.item()
