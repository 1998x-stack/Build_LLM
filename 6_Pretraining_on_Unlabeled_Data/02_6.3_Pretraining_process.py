# 02_6.3_Pretraining_process

"""
Lecture: /6_Pretraining_on_Unlabeled_Data
Content: 6.3 预训练过程
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


class GPTDataset(Dataset):
    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.token_ids[index: index + self.context_length]),
                torch.tensor(self.token_ids[index + 1: index + self.context_length + 1]))


class GPT(nn.Module):
    """临时内联 GPT (最终由 Phase 2 的 common/attention.py 取代)。"""

    def __init__(self, vocab_size: int, embed_size: int, num_layers: int,
                 num_heads: int, max_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_size, num_heads, 4 * embed_size,
                                        batch_first=True, norm_first=True)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_size)
        self.out_head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1]).to(x.device))
        for blk in self.blocks:
            x = blk(x)
        return self.out_head(self.final_norm(x))


def build_vocab(text: str) -> Dict[str, int]:
    return {t: i for i, t in enumerate(sorted(set(re.findall(r"\S+", text.lower()))))}


def train(model, dataloader, n_epochs, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(n_epochs):
        total = 0.0
        n = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        history.append(total / n)
        print(f"Epoch {epoch + 1}/{n_epochs} 平均损失: {history[-1]:.4f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(0)
    text = ("the cat sat on the mat. the dog sat too. "
            "the cat ran fast across the yard. the dog chased the cat.")
    vocab = build_vocab(text)
    token_ids = [vocab[w] for w in re.findall(r"\S+", text.lower())]

    ctx_len = 6
    dataset = GPTDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = GPT(vocab_size=len(vocab), embed_size=32, num_layers=2,
                num_heads=2, max_len=ctx_len)
    history = train(model, loader, n_epochs=12)
    print("训练完成。首轮损失:", round(history[0], 4),
          "末轮损失:", round(history[-1], 4))
    assert history[-1] < history[0], "预训练应降低损失"
    print("预训练损失下降: True")