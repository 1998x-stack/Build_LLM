# 02_6.3_Pretraining_process

"""
Lecture: /6_Pretraining_on_Unlabeled_Data
Content: 6.3 预训练过程
"""

# 与 common/attention.py 保持同步 (canonical) — 见 README
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict


class GPTDataset(Dataset):
    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.token_ids[index: index + self.context_length]),
                torch.tensor(self.token_ids[index + 1: index + self.context_length + 1]))


class MultiHeadAttention(nn.Module):
    """多头自注意力：按 head_dim 投影、按 head_dim 缩放、支持 (可选) 掩码。

    Args:
        embed_size (int): 嵌入维度。
        heads (int): 注意力头数 (需整除 embed_size)。
    """
    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "embed_size 必须能被 heads 整除"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, v, k, q, mask: Optional[torch.Tensor] = None):
        N, v_len, k_len, q_len = q.shape[0], v.shape[1], k.shape[1], q.shape[1]
        v = v.reshape(N, v_len, self.heads, self.head_dim)
        k = k.reshape(N, k_len, self.heads, self.head_dim)
        q = q.reshape(N, q_len, self.heads, self.head_dim)
        v = self.values(v); k = self.keys(k); q = self.queries(q)
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attn = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attn, v]).reshape(N, q_len, self.heads * self.head_dim)
        return self.fc_out(out)


class GPTBlock(nn.Module):
    """单层 GPT(decoder) 变换器块：因果自注意力 + 前馈网络，均带残差与 LayerNorm + dropout。"""
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.dropout(self.norm1(x + self.attention(x, x, x, mask)))
        return self.dropout(self.norm2(x + self.feed_forward(x)))


class GPT(nn.Module):
    """GPT 语言模型：token+位置嵌入 → N 层 GPTBlock(内置因果掩码) → LayerNorm → 词表线性头。"""
    def __init__(self, vocab_size, embed_size, num_layers, heads, device,
                 forward_expansion, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [GPTBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)   # 内置因果掩码
        for layer in self.layers:
            out = layer(out, mask)
        return self.fc_out(out)


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
                heads=2, device="cpu", forward_expansion=4, dropout=0.0,
                max_length=ctx_len)
    history = train(model, loader, n_epochs=12)
    print("训练完成。首轮损失:", round(history[0], 4),
          "末轮损失:", round(history[-1], 4))
    assert history[-1] < history[0], "预训练应降低损失"
    print("预训练损失下降: True")
