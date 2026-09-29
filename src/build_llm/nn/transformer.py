from __future__ import annotations

import torch
import torch.nn as nn

from build_llm.config import GPTConfig
from build_llm.nn.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with attention and MLP residual paths."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_length=config.context_length,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias,
            backend=config.attention_backend,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
