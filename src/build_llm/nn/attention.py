from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """General multi-head attention using full-width Q/K/V projections.

    The legacy lecture API order (values, keys, query) is preserved so the
    encoder/decoder teaching examples remain easy to compare with earlier code.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(keys))
        v = self._split_heads(self.v_proj(values))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            keep = mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(
            query.size(0), query.size(1), self.d_model
        )
        return self.out_proj(out)


class CausalSelfAttention(nn.Module):
    """GPT causal self-attention with readable manual and optimized SDPA backends."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        backend: str = "manual",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if backend not in {"manual", "sdpa"}:
            raise ValueError("backend must be 'manual' or 'sdpa'")
        if backend == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "This PyTorch build does not provide scaled_dot_product_attention"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length
        self.dropout = dropout
        self.backend = backend
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        causal = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal.view(1, 1, context_length, context_length),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        if channels != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, got {channels}"
            )
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length={self.context_length}"
            )

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(
                ~mask, torch.finfo(scores.dtype).min
            )
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(
                weights, p=self.dropout, training=self.training
            )
            y = weights @ v

        y = y.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        return self.resid_dropout(self.out_proj(y))
