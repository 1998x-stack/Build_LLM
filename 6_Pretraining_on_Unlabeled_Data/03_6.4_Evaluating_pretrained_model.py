# 03_6.4_Evaluating_pretrained_model.py

"""
Lecture: /6_Pretraining_on_Unlabeled_Data
Content: 6.4 评估预训练模型
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Sequence


# ---- 复用 6.3 的模型与数据集 ----
# 与 common/attention.py 保持同步 (canonical) — 见 README
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


def build_vocab(text):
    return {t: i for i, t in enumerate(sorted(set(re.findall(r"\S+", text.lower()))))}


def perplexity(model, dataloader) -> float:
    """在数据上计算困惑度 exp(平均交叉熵)。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    return float(math.exp(total_loss / total_tokens))


@torch.no_grad()
def generate(model, start: Sequence[int], id_to_token: Dict[int, str], n_new: int) -> List[str]:
    """自回归采样生成文本。"""
    model.eval()
    ids = list(start)
    window = model.position_embedding.num_embeddings
    for _ in range(n_new):
        # 只喂入末尾窗口，避免位置索引超过 pos_emb 的长度 (max_len)
        x = torch.tensor([ids[-window:]]).long()
        logits = model(x)[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
    return [id_to_token[i] for i in ids]


if __name__ == "__main__":
    torch.manual_seed(0)
    text = ("the cat sat on the mat. the dog sat too. "
            "the cat ran fast across the yard.")
    vocab = build_vocab(text)
    id_to_token = {i: t for t, i in vocab.items()}
    token_ids = [vocab[w] for w in re.findall(r"\S+", text.lower())]

    ctx_len = 4
    from torch.utils.data import Dataset
    class GPTDataset(Dataset):
        """滑动窗口语言模型数据集。"""
        def __init__(self, ids, c):
            self.ids = ids; self.c = c
        def __len__(self): return len(self.ids) - self.c
        def __getitem__(self, i):
            return (torch.tensor(self.ids[i:i+self.c]),
                    torch.tensor(self.ids[i+1:i+self.c+1]))
    dataset = GPTDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    model = GPT(vocab_size=len(vocab), embed_size=32, num_layers=2,
                heads=2, device="cpu", forward_expansion=4, dropout=0.0,
                max_length=ctx_len)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(60):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb).view(-1, len(vocab)), yb.view(-1))
            loss.backward()
            opt.step()

    ppl = perplexity(model, loader)
    print(f"困惑度 (perplexity): {ppl:.3f} (有限且大于 0: {ppl > 0})")
    start = [vocab["the"], vocab["cat"]]
    sample = generate(model, start, id_to_token, n_new=8)
    print("生成文本:", " ".join(sample))