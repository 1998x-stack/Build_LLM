# 03_6.4_Evaluating_pretrained_model

"""Lecture 6.4: evaluate perplexity and generate text with the canonical model."""

from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_llm.data import NextTokenDataset
from build_llm.evaluation import perplexity as _perplexity
from build_llm.generation import GenerationConfig, generate as _generate
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


def perplexity(model, dataloader) -> float:
    return _perplexity(model, dataloader)


def generate(
    model,
    start: Sequence[int],
    id_to_token: Dict[int, str],
    n_new: int,
) -> List[str]:
    seed = torch.tensor([list(start)], dtype=torch.long)
    ids = _generate(
        model,
        seed,
        GenerationConfig(
            max_new_tokens=n_new,
            temperature=0.0,
        ),
    )[0].tolist()
    return [id_to_token[index] for index in ids]


if __name__ == "__main__":
    torch.manual_seed(0)
    text = (
        "the cat sat on the mat. the dog sat too. "
        "the cat ran fast across the yard."
    )
    vocab = build_vocab(text)
    id_to_token = {index: token for token, index in vocab.items()}
    token_ids = [vocab[word] for word in re.findall(r"\S+", text.lower())]
    context_length = 4
    loader = DataLoader(
        GPTDataset(token_ids, context_length),
        batch_size=3,
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
    trainer = Trainer(
        model,
        TrainConfig(
            epochs=3,
            learning_rate=0.03,
            weight_decay=0.0,
            device="cpu",
            seed=0,
        ),
    )
    trainer.fit(loader)
    print("Perplexity:", round(perplexity(model, loader), 3))
    sample = generate(
        model,
        [vocab["the"], vocab["cat"]],
        id_to_token,
        n_new=6,
    )
    print("Generated:", " ".join(sample))
