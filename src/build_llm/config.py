from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttentionBackend = Literal["manual", "sdpa"]


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for the canonical GPT implementation."""

    vocab_size: int
    context_length: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    mlp_ratio: int = 4
    qkv_bias: bool = False
    tie_embeddings: bool = False
    attention_backend: AttentionBackend = "manual"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("d_model, n_heads and n_layers must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be 'manual' or 'sdpa'")
