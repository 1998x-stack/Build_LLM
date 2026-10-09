# 03_4.4_Encoder_and_decoder_architectures

"""Lecture 4.4: minimal pre-norm encoder/decoder Transformer."""

from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_llm.nn.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)
        return x + self.ff(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, heads)
        self.cross_attn = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed, mask)
        query = self.norm2(x)
        x = x + self.cross_attn(encoder_out, encoder_out, query)
        return x + self.ff(self.norm3(x))


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        embed_size: int,
        heads: int,
        num_layers: int,
        ff_expansion: int,
        max_len: int,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, embed_size)
        self.tgt_emb = nn.Embedding(tgt_vocab, embed_size)
        self.pos = nn.Embedding(max_len, embed_size)
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(embed_size, heads, ff_expansion)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, ff_expansion)
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(embed_size, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pos = torch.arange(src.shape[1], device=src.device)
        tgt_pos = torch.arange(tgt.shape[1], device=tgt.device)
        src_hidden = self.src_emb(src) + self.pos(src_pos)
        tgt_hidden = self.tgt_emb(tgt) + self.pos(tgt_pos)
        mask = torch.tril(
            torch.ones(
                tgt.shape[1],
                tgt.shape[1],
                device=tgt.device,
                dtype=torch.bool,
            )
        ).view(1, 1, tgt.shape[1], tgt.shape[1])
        for layer in self.encoder:
            src_hidden = layer(src_hidden)
        for layer in self.decoder:
            tgt_hidden = layer(tgt_hidden, src_hidden, mask)
        return self.out(tgt_hidden)


if __name__ == "__main__":
    model = Transformer(50, 50, 64, 4, 2, 4, 32)
    src = torch.randint(0, 50, (2, 10))
    tgt = torch.randint(0, 50, (2, 10))
    out = model(src, tgt)
    print("Seq2Seq Transformer output shape:", tuple(out.shape))
