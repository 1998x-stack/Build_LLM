from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 20
    temperature: float = 0.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 when provided")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Autoregressively extend tokens on the model's current device."""
    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape (batch, sequence)"
        )
    device = _model_device(model)
    token_ids = token_ids.to(device)
    context_length = int(model.config.context_length)
    was_training = model.training
    model.eval()
    try:
        for _ in range(config.max_new_tokens):
            idx_cond = token_ids[:, -context_length:]
            logits = model(idx_cond)[:, -1, :]
            if config.top_k is not None:
                k = min(config.top_k, logits.size(-1))
                threshold = torch.topk(
                    logits, k
                ).values[:, [-1]]
                logits = logits.masked_fill(
                    logits < threshold, float("-inf")
                )
            if config.temperature == 0:
                next_id = torch.argmax(
                    logits, dim=-1, keepdim=True
                )
            else:
                probs = F.softmax(
                    logits / config.temperature, dim=-1
                )
                next_id = torch.multinomial(
                    probs, num_samples=1
                )
            token_ids = torch.cat(
                (token_ids, next_id), dim=1
            )
    finally:
        if was_training:
            model.train()
    return token_ids
