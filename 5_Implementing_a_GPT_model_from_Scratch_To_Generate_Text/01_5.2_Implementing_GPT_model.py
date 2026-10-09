# 01_5.2_Implementing_GPT_model

"""Lecture 5.2: assemble the canonical GPT implementation."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.attention import GPT, GPTBlock, MultiHeadAttention

GPTSelfAttention = MultiHeadAttention

__all__ = ["GPT", "GPTBlock", "GPTSelfAttention"]


if __name__ == "__main__":
    model = GPT(
        vocab_size=1000,
        embed_size=128,
        num_layers=4,
        heads=4,
        forward_expansion=4,
        dropout=0.1,
        max_length=64,
    )
    tokens = torch.randint(0, 1000, (2, 16))
    logits = model(tokens)
    print("GPT logits shape:", tuple(logits.shape))
    print("Parameters:", f"{model.num_parameters():,}")
    print("Architecture: pre-norm + causal MHA + GELU MLP + final LayerNorm")
