# 02_6.3_Pretraining_process

"""Lecture 6.3: pretrain the canonical GPT model on next-token prediction."""

from pathlib import Path
import re
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_llm.data import NextTokenDataset
from build_llm.training import TrainConfig, Trainer
from common.attention import GPT

GPTDataset = NextTokenDataset


def build_vocab(text: str) -> Dict[str, int]:
    return {
        token: index
        for index, token in enumerate(
            sorted(set(re.findall(r"\S+", text.lower())))
        )
    }


def train(model, dataloader, n_epochs, lr=0.01):
    device = str(next(model.parameters()).device)
    trainer = Trainer(
        model,
        TrainConfig(
            epochs=n_epochs,
            learning_rate=lr,
            weight_decay=0.0,
            grad_clip=1.0,
            device=device,
            seed=0,
        ),
    )
    records = trainer.fit(dataloader)
    history = [record["train_loss"] for record in records]
    for epoch, loss in enumerate(history, start=1):
        print(f"Epoch {epoch}/{n_epochs} average loss: {loss:.4f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(0)
    text = (
        "the cat sat on the mat. the dog sat too. "
        "the cat ran fast across the yard. the dog chased the cat."
    )
    vocab = build_vocab(text)
    token_ids = [vocab[word] for word in re.findall(r"\S+", text.lower())]
    context_length = 6
    loader = DataLoader(
        GPTDataset(token_ids, context_length),
        batch_size=2,
        shuffle=True,
    )
    model = GPT(
        vocab_size=len(vocab),
        embed_size=32,
        num_layers=2,
        heads=2,
        forward_expansion=4,
        dropout=0.0,
        max_length=context_length,
    )
    epochs = 3
    history = train(model, loader, n_epochs=epochs, lr=0.03)
    print("Loss decreased:", history[-1] < history[0])
