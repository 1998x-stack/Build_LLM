"""Compatibility facade for the canonical build_llm implementation.

New code should import from build_llm directly. The classes here preserve
legacy constructor names used by chapter scripts while delegating to the
canonical implementation.
"""
from __future__ import annotations

import torch

from build_llm.config import GPTConfig
from build_llm.model.gpt import GPTModel
from build_llm.nn.attention import (
    MultiHeadAttention as _MultiHeadAttention,
)
from build_llm.nn.transformer import TransformerBlock


class MultiHeadAttention(_MultiHeadAttention):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__(
            embed_size,
            heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.embed_size = embed_size
        self.heads = heads

    @property
    def values(self):
        return self.v_proj

    @property
    def keys(self):
        return self.k_proj

    @property
    def queries(self):
        return self.q_proj

    @property
    def fc_out(self):
        return self.out_proj


class GPTBlock(TransformerBlock):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
        max_length: int = 1024,
    ) -> None:
        cfg = GPTConfig(
            vocab_size=1,
            context_length=max_length,
            d_model=embed_size,
            n_heads=heads,
            n_layers=1,
            dropout=dropout,
            mlp_ratio=forward_expansion,
        )
        super().__init__(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            shape = mask.shape[-2:]
            expected = torch.tril(
                torch.ones(
                    shape,
                    device=mask.device,
                    dtype=mask.dtype,
                )
            )
            candidate = mask.reshape(-1, *shape)[0]
            if not torch.equal(candidate, expected):
                raise ValueError(
                    "GPTBlock accepts only the standard causal mask"
                )
        return super().forward(x)


class GPT(GPTModel):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device: str | None = None,
        forward_expansion: int = 4,
        dropout: float = 0.0,
        max_length: int = 1024,
        *,
        attention_backend: str = "manual",
    ) -> None:
        del device
        cfg = GPTConfig(
            vocab_size=vocab_size,
            context_length=max_length,
            d_model=embed_size,
            n_heads=heads,
            n_layers=num_layers,
            dropout=dropout,
            mlp_ratio=forward_expansion,
            attention_backend=attention_backend,
        )
        super().__init__(cfg)
        self.embed_size = embed_size

    @property
    def word_embedding(self):
        return self.token_embedding

    @property
    def layers(self):
        return self.blocks

    @property
    def fc_out(self):
        return self.lm_head
