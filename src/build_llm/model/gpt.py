from __future__ import annotations

import torch
import torch.nn as nn

from build_llm.config import GPTConfig
from build_llm.nn.transformer import TransformerBlock


class GPTModel(nn.Module):
    """Canonical GPT-style decoder-only language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape (batch, sequence)"
            )
        _, seq_len = token_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"context_length={self.config.context_length}"
            )
        positions = torch.arange(
            seq_len, device=token_ids.device
        )
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        if trainable_only:
            params = (
                p for p in self.parameters() if p.requires_grad
            )
        else:
            params = self.parameters()
        return sum(p.numel() for p in params)
