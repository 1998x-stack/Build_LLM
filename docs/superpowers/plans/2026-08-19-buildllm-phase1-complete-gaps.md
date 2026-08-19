# Build_LLM — Phase 1 (Complete the Gaps) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement every empty lecture stub required by the design so the whole "LLM from scratch" walkthrough has runnable, self-contained code (or clean doc-only stubs) across Chapters 2, 4, and 6.

**Architecture:** New lecture files are self-contained (each `python3 file.py` runs standalone) and bilingual (Chinese+English). Implemented files: `4.5` (hand-rolled self-attention), `4.4` (encoder/decoder), `2.5` (GPT architecture demo), `6.2` (data prep), `6.3` (training loop, with a **temporary inline GPT**), `6.4` (evaluation). Doc-only files get `.py` replaced by a clean stub. Phase 2 will extract a canonical `common/attention.py` and dedup.

**Tech Stack:** Python 3.9, NumPy 2.0, PyTorch 2.8 (CPU), tiktoken 0.14. No pytest in this phase (added in Phase 2).

## Global Constraints

- Working dir: `/Users/x/Desktop/1998x-stack/00-仓库/04-深度学习与CV/从零实现与消融/Build_LLM`
- Every implemented `.py` must run standalone via `python3 <file>.py` and exit 0.
- Docstrings: bilingual (Chinese primary, English to clarify); every method has Args/Returns where applicable.
- Bilingual/self-contained conventions per the design spec (`docs/superpowers/specs/2026-08-19-professionalize-build-llm-design.md`).
- `torch`, `numpy`, `tiktoken` are installed. Do NOT install new packages in Phase 1.
- Commit after each task. Message format: `feat: <chapter> <section> <description>`.

---

### Task 1: Implement `04_4.5_Implementing_self_attention_mechanisms.py` (hand-rolled self-attention)

**Files:**
- Modify: `4_Coding_Attention_Mechanisms/04_4.5_Implementing_self_attention_mechanisms.py`

**Interfaces:**
- Produces: `compute_attention_numpy(q, k, v, causal=True) -> np.ndarray` and `compute_attention_torch(q, k, v, causal=True) -> torch.Tensor`, both shape `(seq_len, d)`, plus a `__main__` that prints the max numerical difference (must be `< 1e-5`).

This file is currently an empty 7-line stub. Replace its entire contents with:

```python
# 04_4.5_Implementing_self_attention_mechanisms

"""
Lecture: /4_Coding_Attention_Mechanisms
Content: 4.5 从零实现自注意力机制
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import Any


def compute_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                            causal: bool = True) -> np.ndarray:
    """
    用 NumPy 从零实现缩放点积自注意力 (scaled dot-product self-attention)。

    步骤: 计算得分 Q·Kᵀ/√d → (可选) 因果掩码 → softmax 归一化 → 加权求和。

    Args:
        q (np.ndarray): 查询矩阵, 形状 (seq_len, d)。
        k (np.ndarray): 键矩阵, 形状 (seq_len, d)。
        v (np.ndarray): 值矩阵, 形状 (seq_len, d)。
        causal (bool): 是否使用因果掩码 (每个 token 只能看到自己及之前的位置)。

    Returns:
        np.ndarray: 自注意力的输出, 形状 (seq_len, d)。
    """
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)                       # (seq_len, seq_len)
    if causal:
        seq_len = q.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))  # 数值稳定
    weights = exp / exp.sum(axis=-1, keepdims=True)
    return weights @ v


def compute_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            causal: bool = True) -> torch.Tensor:
    """
    等价的 PyTorch / einsum 实现，用于与 NumPy 版本做数值校验。

    Args:
        q (torch.Tensor): 查询矩阵, 形状 (seq_len, d)。
        k (torch.Tensor): 键矩阵, 形状 (seq_len, d)。
        v (torch.Tensor): 值矩阵, 形状 (seq_len, d)。
        causal (bool): 是否使用因果掩码。

    Returns:
        torch.Tensor: 自注意力输出, 形状 (seq_len, d)。
    """
    d_k = q.shape[-1]
    scores = torch.einsum("qd,kd->qk", q, k) / np.sqrt(d_k)
    if causal:
        seq_len = q.shape[0]
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return weights @ v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq_len, d = 6, 8
    X = rng.normal(size=(seq_len, d))

    out_np = compute_attention_numpy(X, X, X)
    out_pt = compute_attention_torch(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32),
    )
    diff = float(np.abs(out_np - out_pt.numpy()).max())
    print("NumPy 输出形状:", out_np.shape)
    print("因果掩码下第一行仅依赖自身，其余各 position 正常。")
    print(f"NumPy 与 Torch 最大数值差异: {diff:.3e}")
```

- [ ] **Step 1: Write the file** using the content above (full overwrite).
- [ ] **Step 2: Run and verify**
  Run: `python3 "4_Coding_Attention_Mechanisms/04_4.5_Implementing_self_attention_mechanisms.py"`
  Expected: prints shapes `(6, 8)` and a max difference on the order of `1e-7` (must be `< 1e-6`), exit 0.
- [ ] **Step 3: Commit**
```bash
git add "4_Coding_Attention_Mechanisms/04_4.5_Implementing_self_attention_mechanisms.py"
git commit -m "feat: 4.5 implement self-attention from scratch (numpy + torch)"
```

---

### Task 2: Implement `4.4` Encoder/decoder architectures

**Files:**
- Modify: `4_Coding_Attention_Mechanisms/03_4.4_Encoder_and_decoder_architectures.py`

**Interfaces:**
- Produces: `MultiHeadAttention`, `EncoderBlock`, `DecoderBlock`, `Transformer` (self-contained nn.Modules), `__main__` prints output shape `(2, tgt_len, 50)`.

- [ ] **Step 1: Write the file** — full overwrite:

```python
# 03_4.4_Encoder_and_decoder_architectures

"""
Lecture: /4_Coding_Attention_Mechanisms
Content: 4.4 编码器与解码器架构
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """多头自注意力 (与 4.3 / 5.2 中的实现保持同步)。"""

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
        N, vl, kl, ql = v.shape[0], v.shape[1], k.shape[1], q.shape[1]
        v = v.reshape(N, vl, self.heads, self.head_dim)
        k = k.reshape(N, kl, self.heads, self.head_dim)
        q = q.reshape(N, ql, self.heads, self.head_dim)
        v = self.values(v); k = self.keys(k); q = self.queries(q)
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attn, v]).reshape(N, ql, self.embed_size)
        return self.fc_out(out)


class EncoderBlock(nn.Module):
    """编码器层: 自注意力 + 前馈网络, 均带残差与 LayerNorm (预归一化)。"""

    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, heads)
        self.n1 = nn.LayerNorm(embed_size)
        self.n2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(self, x):
        x = self.n1(x + self.attn(x, x, x))
        return self.n2(x + self.ff(x))


class DecoderBlock(nn.Module):
    """解码器层: 掩码自注意力 + 交叉注意力 + 前馈网络。"""

    def __init__(self, embed_size: int, heads: int, ff_expansion: int):
        super().__init__()
        self.sa = MultiHeadAttention(embed_size, heads)   # 掩码自注意力
        self.ca = MultiHeadAttention(embed_size, heads)   # 交叉注意力 (查编码器输出)
        self.n1 = nn.LayerNorm(embed_size)
        self.n2 = nn.LayerNorm(embed_size)
        self.n3 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expansion * embed_size),
            nn.GELU(),
            nn.Linear(ff_expansion * embed_size, embed_size),
        )

    def forward(self, x, encoder_out, mask: Optional[torch.Tensor] = None):
        x = self.n1(x + self.sa(x, x, x, mask))       # 掩码自注意力
        x = self.n2(x + self.ca(encoder_out, encoder_out, x))  # 交叉注意力
        return self.n3(x + self.ff(x))


class Transformer(nn.Module):
    """极简序列到序列 Transformer: 编码器-解码器框架。"""

    def __init__(self, src_vocab: int, tgt_vocab: int, embed_size: int,
                 heads: int, num_layers: int, ff_expansion: int, max_len: int):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, embed_size)
        self.tgt_emb = nn.Embedding(tgt_vocab, embed_size)
        self.pos = nn.Embedding(max_len, embed_size)
        self.encoder = nn.ModuleList([EncoderBlock(embed_size, heads, ff_expansion)
                                      for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(embed_size, heads, ff_expansion)
                                      for _ in range(num_layers)])
        self.out = nn.Linear(embed_size, tgt_vocab)

    def forward(self, src, tgt):
        s_len, t_len = src.shape[1], tgt.shape[1]
        src = self.src_emb(src) + self.pos(torch.arange(s_len).to(src.device))
        tgt = self.tgt_emb(tgt) + self.pos(torch.arange(t_len).to(tgt.device))
        mask = torch.tril(torch.ones(t_len, t_len)).unsqueeze(0).unsqueeze(0).to(src.device)
        for layer in self.encoder:
            src = layer(src)
        for layer in self.decoder:
            tgt = layer(tgt, src, mask)
        return self.out(tgt)


if __name__ == "__main__":
    model = Transformer(src_vocab=50, tgt_vocab=50, embed_size=64, heads=4,
                        num_layers=2, ff_expansion=4, max_len=32)
    src = torch.randint(0, 50, (2, 10))
    tgt = torch.randint(0, 50, (2, 10))
    out = model(src, tgt)
    print("Seq2Seq Transformer 输出形状:", tuple(out.shape))
    print("编码器层数:", len(model.encoder), "| 解码器层数:", len(model.decoder))
```

- [ ] **Step 2: Run and verify**
Run: `python3 "4_Coding_Attention_Mechanisms/03_4.4_Encoder_and_decoder_architectures.py"`
Expected: `Seq2Seq Transformer 输出形状: (2, 10, 50)`, exit 0.
- [ ] **Step 3: Commit**
```bash
git add "4_Coding_Attention_Mechanisms/03_4.4_Encoder_and_decoder_architectures.py"
git commit -m "feat: 4.4 implement encoder/decoder architectures"
```

---

### Task 3: Implement `2.5` GPT architecture demo

**Files:**
- Modify: `2_Understanding_Large_Language_Models/04_2.5_A_closer_look_at_the_GPT_architecture.py`

**Interfaces:**
- Produces: prints a GPT‑2‑style architecture summary (embed dim, layers, heads, total params) and a block diagram.

- [ ] **Step 1: Write the file** — full overwrite:

```python
# 04_2.5_A_closer_look_at_the_GPT_architecture

"""
Lecture: /2_Understanding_Large_Language_Models
Content: 2.5 深入剖析 GPT 架构
"""

import torch
import torch.nn as nn


class GPT(nn.Module):
    """GPT-2 风格的解码器专用语言模型 (仅用于架构演示，代码来自第 5 章)。"""

    def __init__(self, vocab_size, embed_size, num_layers, heads, max_len):
        super().__init__()
        self.embed_size = embed_size
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_size, heads, 4 * embed_size,
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


if __name__ == "__main__":
    # 以 GPT-2 (124M) 的量级为参考: 12 层, 768 维, 12 头
    cfg = dict(vocab_size=50257, embed_size=768)
    model = GPT(vocab_size=cfg["vocab_size"], embed_size=cfg["embed_size"],
                num_layers=12, max_len=1024)
    total = sum(p.numel() for p in model.parameters())
    layers = len(model.blocks)

    print("===== GPT 架构概览 (GPT-2 约 124M 参数量级) =====")
    print(f"词典大小 (vocab_size):        {cfg['vocab_size']}")
    print(f"嵌入维度 (embed_size):        {cfg['embed_size']}")
    print(f"解码器层数 (num_layers):      {layers}")
    print(f"注意力头数 (num_heads):       12")
    print(f"最大上下文长度 (max_len):     1024")
    print(f"总参数量 (total params):      {total:,}")
    print()
    print("结构:  token_embedding → 位置编码 → [Masked Self-Attention → FFN]*12 → LayerNorm → Linear(→ vocab)")
```

- [ ] **Step 2: Run and verify**
Run: `python3 "2_Understanding_Large_Language_Models/04_2.5_A_closer_look_at_the_GPT_architecture.py"`
Expected: prints the config table (12 blocks, 768 embed, total params ~140M, exit 0).
- [ ] **Step 3: Commit**
```bash
git add "2_Understanding_Large_Language_Models/04_2.5_A_closer_look_at_the_GPT_architecture.py"
git commit -m "feat: 2.5 add GPT architecture demo"
```

---

### Task 4: Implement `6.2` Data preparation

**Files:**
- Modify: `6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py`

**Interfaces:**
- Produces: `build_vocab(text) -> (vocab, int)` or list of `(token, id)`; `GPTDataset(ids, context_length)` (a `torch.utils.data.Dataset`); `__main__` prints input/target sample shapes `(context_length,)`.

- [ ] **Step 1: Write the file** — full overwrite:

```python
# 01_6.2_Data_preparation

"""
Lecture: /6_Pretraining_on_Unlabeled_Data
Content: 6.2 数据准备
"""

import re
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class GPTDataset(Dataset):
    """滑动窗口构造 (输入, 目标) 样本的语言模型数据集。

    每个样本: inputs = tokens[i : i+context], targets = tokens[i+1 : i+context+1]。
    """

    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.token_ids[index: index + self.context_length]
        targets = self.token_ids[index + 1: index + self.context_length + 1]
        return torch.tensor(inputs), torch.tensor(targets)


def build_vocab(text: str) -> Dict[str, int]:
    """从文本构建词汇表 (简单空格分词, 排序保证确定性)。"""
    tokens = sorted(set(re.findall(r"\S+", text.lower())))
    return {token: i for i, token in enumerate(tokens)}


if __name__ == "__main__":
    text = ("the cat sat on the mat. the dog sat too. "
            "the cat ran fast across the yard.")
    vocab = build_vocab(text)
    token_ids = [vocab[w] for w in re.findall(r"\S+", text.lower())]

    ctx_len = 4
    dataset = GPTDataset(token_ids, ctx_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    print("词汇表大小:", len(vocab))
    print("数据集样本数:", len(dataset))
    inputs, targets = next(iter(loader))
    print("输入批次形状:", tuple(inputs.shape), "目标批次形状:", tuple(targets.shape))
    print("示例输入 token id:", inputs[0].tolist(), "-> 目标:", targets[0].tolist())
```

- [ ] **Step 2: Run and verify**
Run: `python3 "6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py"`
Expected: prints vocab size, sample count, input/target batch shapes `(2, 4)`, exit 0.
- [ ] **Step 3: Commit**
```bash
git add "6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py"
git commit -m "feat: 6.2 implement data preparation (GPTDataset)"
```

---

### Task 5: Implement `6.3` Pretraining process

**Files:**
- Modify: `6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py`

**Interfaces:**
- Consumes: Task 4's `GDPTDataset`/`build_vocab` pattern (redefine locally to stay self-contained).
- Produces: a temporary inline `GPT` + `train()` loop; `__main__` prints per-epoch loss; last epoch loss must be lower than first.

- [ ] **Step 1: Write the file** — full overwrite:

```python
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
```

- [ ] **Step 2: Run and verify**
Run: `python3 "6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py"`
Expected: per-epoch loss printed, final loss lower than first, trailing `预训练损失: True`, exit 0.
- [ ] **Step 3: Commit**
```bash
git add "6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py"
git commit -m "feat: 6.3 implement pretraining loop"
```

---

### Task 6: Implement `6.4` Evaluating pretrained model

**Files:**
- Modify: `6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py`

**Interfaces:**
- Consumes: Task 5's inline `GPT` + `GPTDataset`.
- Produces: `generate(model, start_tokens, token_to_id, id_to_token, max_new)`, a perplexity computation; `__main__` prints finite perplexity and a generated sentence.

- [ ] **Step 1: Write the file** — full overwrite:

```python
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
    for _ in range(n_new):
        x = torch.tensor([ids]).long()
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
```

- [ ] **Step 2: Run and verify**
Run: `python3 "6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py"`
Expected: finite perplexity and a generated sentence with plausible tokens, exit 0.
- [ ] **Step 3: Commit**
```bash
git add "6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py"
git commit -m "feat: 6.4 implement evaluation (perplexity + generation)"
```

---

### Task 7: Convert doc-only conceptual stubs

**Files:**
- Modify: `2_Understanding_Large_Language_Models/00_2.1….py` … `04_2.5….py` (except `2.5`, done in Task 3), `4_Coding_Attention_Mechanisms/00_4.1…py`, `6_Pretraining_on_Unlabeled_Data/00_6.1…py`

**Interfaces:**
- Produces: each `.py` retains the lecture header and parser-comment pointing to the matching `.md`.

- [ ] **Step 1: For each doc-only file, overwrite with the stub template**
Files: `00_2.1 What_is_a_LLM`, `01_2.2 Applications_of_LLMs`, `02_2.3 Stages…`, `03_2.4 Utilizing_large_datasets`, `00_4.1 Introduction_to_attention_mechanisms`, `00_6.1 Concept_of_pretraining`.
Template (replace `NN_N.N_Title` and the chapter directory accordingly):

```python
# NN_N.N_Title

"""
Lecture: /<Chapter>
Content: <NN_N.N_title> (概念性章节)
"""

# 概念性章节，无独立代码实现；请参阅对应 .md 文件。
# 本章节核心概念与图示见 "NN_N.N_....md"。
```

- [ ] **Step 2: Run and verify each** (should exit 0)
`for f in "2_Understanding_Large_Language_Models/00_2.1_What_is_a_LLM.py" ... "6_Pretraining_on_Unlabeled_Data/00_6.1_Concept_of_pretraining.py"; do python3 "$f" || echo "FAIL $f"; done`
- [ ] **Step 3: Commit**
```bash
git add 2_Understanding_Large_Language_Models 4_Coding_Attention_Mechanisms/00_4.1* 6_Pretraining_on_Unlabeled_Data/00_6.1*
git commit -m "chore: mark conceptual lectures as doc-only stubs"
```

### Task 8: Full-phase smoke verification

- [ ] **Step 1: Run every lecture `.py`**
Run for every non-doc-only `.py` (Ch 3, 4, 5, and new 2.5/4.4/4.5/6.x):
```bash
for f in $(find . -name "*.py" -not -path "./.git/*" -not -path "./docs/*" -not -name "main.py"); do
  timeout 120 python3 "$f" >/dev/null 2>&1 && echo "PASS $f" || echo "FAIL $f"
done
```
Expected: no `FAIL` lines (or only for stubs that intentionally run trivially—verify those print nothing).
- [ ] **Step 2: Commit any leftover cleanup**
```bash
git add -A && git commit -m "chore: phase 1 verification pass" || echo "nothing to commit"
```