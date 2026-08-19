# 03_4.4_Encoder_and_decoder_architectures

"""
Lecture: /4_Coding_Attention_Mechanisms
Content: 4.4 编码器与解码器架构
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """多头自注意力 (与 4.3 / 5.2 中的实现保持同步)。"""

    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "embed_size 必须能被 heads 整除"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, v, k, q, mask: Optional[torch.Tensor] = None):
        N, vl, kl, ql = v.shape[0], v.shape[1], k.shape[1], q.shape[1]
        v = v.reshape(N, vl, self.heads, self.head_dim)
        k = k.reshape(N, kl, self.heads, self.head_dim)
        q = q.reshape(N, ql, self.heads, self.head_dim)
        v = self.values(v); k = self.keys(k); q = self.queries(q)
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attn, v]).reshape(N, ql, self.embed_size)
        return self.fc_out(out)


class EncoderBlock(nn.Module):
    """编码器层: 自注意力 + 前馈网络, 均带残差与 LayerNorm (预归一化)。"""

    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, heads)
        self.n1 = nn.LayerNorm(embed_size)
        self.n2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(self, x):
        x = self.n1(x + self.attn(x, x, x))
        return self.n2(x + self.ff(x))


class DecoderBlock(nn.Module):
    """解码器层: 掩码自注意力 + 交叉注意力 + 前馈网络。"""

    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.sa = MultiHeadAttention(embed_size, heads)   # 掩码自注意力
        self.ca = MultiHeadAttention(embed_size, heads)   # 交叉注意力 (查编码器输出)
        self.n1 = nn.LayerNorm(embed_size)
        self.n2 = nn.LayerNorm(embed_size)
        self.n3 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(self, x, encoder_out, mask: Optional[torch.Tensor] = None):
        x = self.n1(x + self.sa(x, x, x, mask))       # 掩码自注意力
        x = self.n2(x + self.ca(encoder_out, encoder_out, x))  # 交叉注意力
        return self.n3(x + self.ff(x))


class Transformer(nn.Module):
    """极简序列到序列 Transformer: 编码器-解码器框架。"""

    def __init__(self, src_vocab: int, tgt_vocab: int, embed_size: int,
                 heads: int, num_layers: int, ff_expansion: int, max_len: int):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, embed_size)
        self.tgt_emb = nn.Embedding(tgt_vocab, embed_size)
        self.pos = nn.Embedding(max_len, embed_size)
        self.encoder = nn.ModuleList([EncoderBlock(embed_size, heads, ff_expansion)
                                      for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(embed_size, heads, ff_expansion)
                                      for _ in range(num_layers)])
        self.out = nn.Linear(embed_size, tgt_vocab)

    def forward(self, src, tgt):
        s_len, t_len = src.shape[1], tgt.shape[1]
        src = self.src_emb(src) + self.pos(torch.arange(s_len).to(src.device))
        tgt = self.tgt_emb(tgt) + self.pos(torch.arange(t_len).to(tgt.device))
        mask = torch.tril(torch.ones(t_len, t_len)).unsqueeze(0).unsqueeze(0).to(src.device)
        for layer in self.encoder:
            src = layer(src)
        for layer in self.decoder:
            tgt = layer(tgt, src, mask)
        return self.out(tgt)


if __name__ == "__main__":
    model = Transformer(src_vocab=50, tgt_vocab=50, embed_size=64, heads=4,
                        num_layers=2, ff_expansion=4, max_len=32)
    src = torch.randint(0, 50, (2, 10))
    tgt = torch.randint(0, 50, (2, 10))
    out = model(src, tgt)
    print("Seq2Seq Transformer 输出形状:", tuple(out.shape))
    print("编码器层数:", len(model.encoder), "| 解码器层数:", len(model.decoder))