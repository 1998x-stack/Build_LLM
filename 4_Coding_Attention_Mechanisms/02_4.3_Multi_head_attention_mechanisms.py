# 02_4.3_Multi_head_attention_mechanisms

"""Lecture 4.3: multi-head attention with full embedding-space projections."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.attention import MultiHeadAttention


if __name__ == "__main__":
    torch.manual_seed(0)
    module = MultiHeadAttention(embed_size=64, heads=4)
    values = torch.randn(2, 7, 64)
    keys = torch.randn(2, 7, 64)
    query = torch.randn(2, 5, 64)
    mask = torch.ones(2, 1, 5, 7)
    out = module(values, keys, query, mask)
    print("Output shape:", tuple(out.shape))
    print("Each Q/K/V projection is 64 -> 64 before splitting into 4 heads.")
