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
from typing import List, Dict, Sequence


# ---- 复用 6.3 的模型与数据集 ----
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_dim, num_heads, 4 * embed_dim,
                                        batch_first=True, norm_first=True)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1]).to(x.device))
        for blk in self.blocks:
            x = blk(x)
        return self.out_head(self.final_norm(x))


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
    window = model.pos_emb.num_embeddings
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

    model = GPT(len(vocab), 32, 2, 2, ctx_len)
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