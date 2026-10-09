#!/usr/bin/env python3
"""Train a tiny GPT end-to-end on a deterministic toy corpus."""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import DataLoader

from build_llm.config import GPTConfig
from build_llm.data import NextTokenDataset
from build_llm.generation import GenerationConfig, generate
from build_llm.model import GPTModel
from build_llm.training import TrainConfig, Trainer


def main() -> None:
    words = (
        "the cat sat on the mat the dog sat on the rug "
        "the cat chased the dog the dog chased the cat "
    ).split() * 8
    vocab = {
        token: index
        for index, token in enumerate(sorted(Counter(words)))
    }
    inverse = {index: token for token, index in vocab.items()}
    token_ids = [vocab[word] for word in words]

    context = 8
    loader = DataLoader(
        NextTokenDataset(token_ids, context),
        batch_size=16,
        shuffle=True,
    )
    model = GPTModel(
        GPTConfig(
            vocab_size=len(vocab),
            context_length=context,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
    )
    trainer = Trainer(
        model,
        TrainConfig(
            epochs=8,
            learning_rate=0.02,
            weight_decay=0.0,
            seed=0,
        ),
    )
    history = trainer.fit(loader)
    print(
        "train loss:",
        f"{history[0]['train_loss']:.4f}",
        "->",
        f"{history[-1]['train_loss']:.4f}",
    )

    seed = torch.tensor([[vocab["the"], vocab["cat"]]])
    generated = generate(
        model,
        seed,
        GenerationConfig(max_new_tokens=10, temperature=0.0),
    )[0].tolist()
    print("generated:", " ".join(inverse[index] for index in generated))


if __name__ == "__main__":
    main()
