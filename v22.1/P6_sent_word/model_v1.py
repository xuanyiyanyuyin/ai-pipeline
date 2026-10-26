"""
P6 V1 最简版 — P5的精确逆运算
==============================
无位置嵌入, 无encoder, 无gate, 一个矩阵乘
"""
import torch, torch.nn as nn

class SentToWordsDecoderV1(nn.Module):
    """P5逆: sent_vec[256] → Linear → [max_words, 128]"""
    def __init__(self, sent_dim=256, word_dim=128, max_words=128):
        super().__init__()
        self.max_words = max_words
        self.decode = nn.Linear(sent_dim, max_words * word_dim, bias=False)
        nn.init.xavier_uniform_(self.decode.weight, gain=0.1)

    def forward(self, sent_vec, last_loss=1.0, gate=None):
        flat = self.decode(sent_vec)  # [b, max_words*128]
        return flat.view(sent_vec.shape[0], self.max_words, -1)
