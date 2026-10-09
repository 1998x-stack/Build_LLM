from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    was_training = model.training
    model.eval()
    device = _model_device(model)
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += targets.numel()
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("cannot evaluate an empty batch iterable")
    return total_loss / total_tokens


def perplexity(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    return math.exp(evaluate_loss(model, batches))
