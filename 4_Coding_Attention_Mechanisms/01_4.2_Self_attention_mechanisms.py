# 01_4.2_Self_attention_mechanisms

"""Lecture 4.2: self-attention using the canonical full-width Q/K/V implementation."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.attention import MultiHeadAttention


class SelfAttention(MultiHeadAttention):
    """Teaching alias preserving the original lecture class name."""


if __name__ == "__main__":
    torch.manual_seed(0)
    module = SelfAttention(embed_size=64, heads=4)
    x = torch.randn(2, 6, 64)
    causal = torch.tril(torch.ones(6, 6)).view(1, 1, 6, 6)
    out = module(x, x, x, causal)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(out.shape))
    print("Q projection:", module.queries.in_features, "->", module.queries.out_features)
