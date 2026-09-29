from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from build_llm.evaluation import evaluate_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float | None = 1.0
    seed: int = 42
    device: str | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self, model: torch.nn.Module, config: TrainConfig
    ) -> None:
        if config.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.model = model
        self.config = config
        seed_everything(config.seed)
        if config.device is not None:
            device = torch.device(config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self, batches: Any) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for inputs, targets in batches:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        if steps == 0:
            raise ValueError(
                "cannot train on an empty batch iterable"
            )
        return total_loss / steps

    def fit(
        self, train_batches: Any, val_batches: Any | None = None
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for _ in range(self.config.epochs):
            record = {
                "train_loss": self.train_epoch(train_batches)
            }
            if val_batches is not None:
                record["val_loss"] = evaluate_loss(
                    self.model, val_batches
                )
            history.append(record)
        return history
