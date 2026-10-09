# 01_6.2_Data_preparation

"""Lecture 6.2: prepare sliding-window next-token training samples."""

from pathlib import Path
import re
import sys
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_llm.data import NextTokenDataset

GPTDataset = NextTokenDataset


def build_vocab(text: str) -> Dict[str, int]:
    tokens = sorted(set(re.findall(r"\S+", text.lower())))
    return {token: index for index, token in enumerate(tokens)}


if __name__ == "__main__":
    text = (
        "the cat sat on the mat. the dog sat too. "
        "the cat ran fast across the yard."
    )
    vocab = build_vocab(text)
    token_ids = [vocab[word] for word in re.findall(r"\S+", text.lower())]
    dataset = GPTDataset(token_ids, context_length=4)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    inputs, targets = next(iter(loader))
    print("Vocabulary size:", len(vocab))
    print("Input batch shape:", tuple(inputs.shape))
    print("Target batch shape:", tuple(targets.shape))
