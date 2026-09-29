from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class NextTokenDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    """Sliding-window next-token prediction dataset."""

    def __init__(
        self,
        token_ids: Sequence[int],
        context_length: int,
        stride: int = 1,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if len(token_ids) <= context_length:
            raise ValueError(
                "token_ids must contain more than context_length tokens"
            )
        self.token_ids = list(token_ids)
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        return (
            (len(self.token_ids) - self.context_length - 1)
            // self.stride
            + 1
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        stop = start + self.context_length
        x = torch.tensor(
            self.token_ids[start:stop], dtype=torch.long
        )
        y = torch.tensor(
            self.token_ids[start + 1 : stop + 1],
            dtype=torch.long,
        )
        return x, y
