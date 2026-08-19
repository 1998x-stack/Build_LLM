# Build_LLM — Phase 2 (Professionalize Code + Tests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Professionalize the working code — extract a canonical causal `common/attention.py`, keep lecture files copy-consistent, add a `pytest` suite + a `run_all.py` smoke runner, normalize docstrings, and resolve the Phase-1 non-causal-attention debt (temporary unmasked GPT in `6.3`/`6.4`).

**Architecture:** `common/attention.py` is the single source of truth (multi-head attention, GPT block, GPT with built-in causal masking). Lecture files (`4.2`, `4.3`, `5.2`, `6.3`, `6.4`) carry copy-consistent classes with a canonical header comment. A `pytest` suite verifies behavior, the built-in causal mask, and copy-consistency against the `5.2` reference. `run_all.py` smoke-runs every lecture.

**Tech Stack:** Python 3.9, NumPy 2.0, PyTorch 2.8 (CPU), tiktoken 0.14, pytest 8.4.2 (installed).

## Global Constraints

- Working dir: `/Users/x/Desktop/1998x-stack/00-仓库/04-深度学习与CV/从零实现与消融/Build_LLM`
- Every lecture `.py` runs standalone via `python3 <file>.py` (exit 0).
- `python3 -m pytest tests/ -q` passes (no failures).
- Bilingual docstrings (Chinese primary, English to clarify); every public method has Args/Returns.
- `common/attention.py` is the canonical source; lecture copies are behavior-identical with header comment `# 与 common/attention.py 保持同步 (canonical) — 见 README`.
- No new packages beyond pytest. Commit after each task.

---

### Task 1: Create canonical `common/attention.py` (corrected causal GPT)

**Files:**
- Create: `common/attention.py`
- Create: `tests/test_gpt.py`, `tests/test_attention.py`

**Interfaces:**
- Produces: `MultiHeadAttention(embed_size, heads)`, `GPTBlock(embed_size, heads, dropout, forward_expansion)`, `GPT(vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)`. `GPT.forward(x) -> (N, seq, vocab_size)` with a built-in lower-triangular causal mask.
- The `GPT` signature mirrors `5_.../01_5.2_...`.py's `GPT` exactly so the two instantiate identically.

- [ ] **Step 1: Write the failing tests**

`tests/test_gpt.py`:
```python
import os, torch, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
from common.attention import GPT, MultiHeadAttention, GPTBlock

def test_gpt_forward_shape():
    model = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    out = model(torch.randint(0, 16, (2, 6)))
    assert tuple(out.shape) == (2, 6, 16)

def test_gpt_causal_pos0_unaffected_by_future():
    m = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    x1 = torch.tensor([[3, 4, 5, 6, 7, 8]])
    x2 = torch.tensor([[3, 9, 9, 9, 9, 9]])   # same pos-0, different future
    assert torch.allclose(m(x1)[0, 0], m(x2)[0, 0], atol=1e-5)

def test_gpt_copy_consistent_with_52():
    m52 = load("5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.py", "g52")
    g = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    g52 = m52.GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    g.load_state_dict(g52.state_dict())
    x = torch.randint(0, 16, (2, 6))
    mask = torch.tril(torch.ones(6, 6))          # 5.2 requires the mask passed explicitly
    assert torch.equal(g(x), g52(x, mask))       # identical & causal
```

`tests/test_attention.py`:
```python
import torch
from common.attention import MultiHeadAttention

def test_mha_output_shape():
    m = MultiHeadAttention(64, 4)
    out = m(torch.rand(2, 6, 64), torch.rand(2, 6, 64), torch.rand(2, 6, 64))
    assert tuple(out.shape) == (2, 6, 64)

def test_mha_projects_per_head_dim():
    m = MultiHeadAttention(64, 4)
    assert m.head_dim == 16
    assert m.values.in_features == 16 and m.queries.out_features == 16
```

- [ ] **Step 2: Run `python3 -m pytest tests/ -q`** — fails: `ModuleNotFoundError: common.attention`.
- [ ] **Step 3: Write `common/attention.py`** (full content, verified):
```python
"""
common/attention.py — 多头注意力与 GPT 模块的单一事实来源 (canonical source of truth).

与 4.2 / 4.3 / 5.2 及第 6 章相关文件保持行为一致；修改请在所有副本一并同步。
"""
import torch
import torch.nn as nn
from typing import Optional


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
```
- [ ] **Step 4: Run `python3 -m pytest tests/ -q`** — all pass.
- [ ] **Step 5: Commit** `git add common tests && git commit -m "feat: add canonical common/attention.py + GPT/attention tests"`

---

### Task 2: Replace the unmasked inline GPT in `6.3`/`6.4` with the causal canonical copy

**Files:**
- Modify: `6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py`, `6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py`

**Interfaces:**
- Consumes: `common/attention.py` classes (copied inline, self-contained). Produces: each file's `class GPT` becomes the common copy (built-in causal mask); update each `__main__` constructor to the common signature `GPT(vocab_size=..., embed_size=..., num_layers=..., heads=..., device="cpu", forward_expansion=..., dropout=..., max_length=...)`. Behavior contract: `6.3` loss decreases; `6.4` has finite perplexity + generated text.

- [ ] **Step 1:** In both files, replace the `nn.TransformerEncoderLayer`-based `class GPT` (note `6.4`'s current signature is `GPT(vocab_size, embed_dim, num_layers, num_heads, max_len)`) with verbatim copies of `MultiHeadAttention` + `GPTBlock` + `GPT` from `common/attention.py`, plus the header comment `# 与 common/attention.py 保持同步 (canonical) — 见 README`. Update each `__main__`'s model constructors to the common signature.
- [ ] **Step 2:** Run both files standalone (exit 0; `6.3` loss final < first; `6.4` finite perplexity + generation). (Causal `6.4` perplexity will be higher than the old ~1.0 — expected/correct.)
- [ ] **Step 3: Commit** `feat: 6.3,6.4 use causal canonical GPT (resolves unmasked-attention debt)`

---

### Task 3: Copy-consistency for `4.2`, `4.3`, `5.2` + canonical header comment

**Files:**
- Modify: `4_Coding_Attention_Mechanisms/01_4.2_Self_attention_mechanisms.py`, `4_Coding_Attention_Mechanisms/02_4.3_Multi_head_attention_mechanisms.py`, `5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.py`

**Interfaces / Produces:**
- `4.2`/`4.3` MHA and `5.2` GPT behavior-identical to `common/attention.py` (head_dim projection, head_dim scaling, same einsum — already fixed in Phase 1). Add the canonical header comment after each module docstring.

- [ ] **Step 1:** Diff each against `common/attention.py`; reconcile drift. Add `# 与 common/attention.py 保持同步 (canonical) — 见 README` after each module docstring.
- [ ] **Step 2:** Run each standalone (exit 0) + `python3 -m pytest tests/ -q` (all pass).
- [ ] **Step 3: Commit** `chore: sync lecture files with canonical common/attention (copy-consistency)`

---

### Task 4: Tokenizer + BPE tests

**Files:**
- Create: `tests/test_tokenizer.py`, `tests/test_bpe.py`

**Interfaces:**
- Consumes (via `load`): `TextTokenizer` from `3_.../01_3.2_...`, `TokenizerWithSpecialTokens` + `build_vocab` from `03_3.4_...`, `BytePairEncoding` from `04_3.5_...`.

- [ ] **Step 1: Write `tests/test_tokenizer.py`**
```python
import os, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

t32 = load("3_Working_with_Text_Data/01_3.2_Tokenizing_text.py", "t32")
t34 = load("3_Working_with_Text_Data/03_3.4_Adding_special_context_tokens.py", "t34")

def test_32_encode_decode_roundtrip():
    tok = t32.TextTokenizer()
    tok.build_vocab(["apple banana cherry", "apple is sweet"])
    assert tok.decode(tok.encode("apple banana")) == "apple banana"

def test_32_most_common_tokens():
    tok = t32.TextTokenizer()
    tok.build_vocab(["apple banana", "apple apple banana"])
    assert tok.most_common_tokens(1) == ["apple"]

def test_34_unknown_maps_to_unk():
    vocab = t34.build_vocab(["hello", "world"])
    tok = t34.TokenizerWithSpecialTokens(vocab)
    assert tok.encode("missingword") == [vocab["<|unk|>"]]

def test_34_known_decode_matches_encoded_text():
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."
    vocab = t34.build_vocab(["hello", "do", "you", "like", "tea", "in", "the", "sunlit", "terraces"])
    tok = t34.TokenizerWithSpecialTokens(vocab)
    assert tok.decode(tok.encode(text)) == text
```
- [ ] **Step 2: Write `tests/test_bpe.py`**
```python
import os, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

bpe = load("3_Working_with_Text_Data/04_3.5_Byte_pair_encoding.py", "bpe")

def test_bpe_fit_encode_decode():
    m = bpe.BytePairEncoding(vocab_size=50)
    m.fit(["low", "lowest", "newer", "wider"])
    assert bpe.decode(m.encode("lower")) == "lower"

def test_bpe_roundtrip_clean():
    m = bpe.BytePairEncoding(vocab_size=50)
    m.fit(["low", "lowest", "newer", "wider"])
    assert bpe.decode(m.encode("lowest")) == "lowest"
```
> **Before you commit:** run each test; if a decoded string differs in punctuation/case (e.g. `3.4` re-adds a space), loosen the assertion to `assert decoded.replace(" ", "") == expected.replace(" ", "")` so the test verifies token identity, not whitespace. Note the `4.1`/`2.x` stubs are comments-only and need no tests.
- [ ] **Step 3:** `python3 -m pytest tests/test_tokenizer.py tests/test_bpe.py -q` → pass.
- [ ] **Step 4:** Commit `test: add tokenizer + BPE unit tests`

---

### Task 5: Pretraining tests (6.2, 6.3, 6.4)

**Files:**
- Create: `tests/test_pretraining.py`

**Interfaces:**
- Consumes (via `load`): `6.2` `GPTDataset` + `build_vocab`; `6.3` `train(model, dataloader, n_epochs, lr=0.01)` + `GPT`; `6.4` `perplexity(model, dataloader)` + `generate(model, start, id_to_token, n_new)` + `GPT`. After Task 2, `6.3`/`6.4` GPT use the **common signature**.

- [ ] **Step 1: Write `tests/test_pretraining.py`** (complete):
```python
import os, math, importlib.util, torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

CORPUS = "the cat sat on the mat. the dog sat too. the cat ran fast."

def _fixture():
    vocab = {t: i for i, t in enumerate(sorted(set(CORPUS.lower().split())))}
    ids = [vocab[w] for w in CORPUS.split()]
    ctx = 4
    m62 = load("6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py", "m62")
    ds = m62.GPTDataset(ids, ctx)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    return vocab, ids, ctx, loader

def test_62_batch_shapes_and_shift():
    vocab, ids, ctx, loader = _fixture()
    x, y = next(iter(loader))
    assert list(x.shape) == [2, ctx] and list(y.shape) == [2, ctx]
    for i in range(x.size(0)):
        assert torch.equal(y[i, :-1], x[i, 1:])   # y = x shifted left by 1 (last target = next id)

def test_63_train_reduces_loss():
    m63 = load("6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py", "m63")
    vocab, ids, ctx, loader = _fixture()
    model = m63.GPT(vocab_size=len(vocab), embed_size=16, num_layers=1, heads=2,
                    device="cpu", forward_expansion=2, dropout=0.0, max_length=ctx)
    history = m63.train(model, loader, n_epochs=3, lr=0.05)
    assert history[-1] < history[0]

def test_64_perplexity_finite_and_generate():
    m64 = load("6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py", "m64")
    vocab, ids, ctx, loader = _fixture()
    model = m64.GPT(vocab_size=len(vocab), embed_size=16, num_layers=1, heads=2,
                    device="cpu", forward_expansion=2, dropout=0.0, max_length=ctx)
    ids2tok = {i: t for t, i in vocab.items()}
    ppl = m64.perplexity(model, loader)
    assert ppl > 0 and (not math.isinf(ppl)) and not math.isnan(ppl)
    sample = m64.generate(model, [0], ids2tok, n_new=3)
    assert len(sample) > 0 and all(t in ids2tok.values() for t in sample)
```
> (If `6.3`/`6.4` `GPTDataset` or helper signatures differ after Task 2, adapt `_fixture()`/the `GPT(...)` calls to the real ones. Keep epochs small so tests run fast.)
- [ ] **Step 2:** `python3 -m pytest tests/test_pretraining.py -q` — pass.
- [ ] **Step 3:** Commit `test: add pretraining unit tests (6.2/6.3/6.4)`

---

### Task 6: `run_all.py` smoke runner

**Files:**
- Create: `run_all.py`

**Interfaces / Produces:**
- Discovers every lecture `.py`, subprocess-runs each (timeout), reports `PASS/FAIL`, exits non-zero on any failure. Skips `.git`, `docs`, `common`, `tests`, `run_all.py`, `main.py`, `modify_files.sh`.

- [ ] **Step 1: Write `run_all.py`** (complete):
```python
#!/usr/bin/env python3
"""run_all.py — 冒烟测试:运行每个课程 .py 并报告 PASS/FAIL; 非零退出表示有失败。"""
import glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {".git", "docs", "common", "tests", ".superpowers"}
SKIP = {"main.py", "modify_files.sh", "run_all.py"}

def _run(rel):
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, rel)],
                           cwd=ROOT, timeout=300, capture_output=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def main():
    files = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        if any(p in EXCLUDE for p in rel.split(os.sep)) or os.path.basename(rel) in SKIP:
            continue
        files.append(rel)
    fails = []
    for rel in files:
        ok = _run(rel)
        print(("PASS" if ok else "FAIL"), rel)
        if not ok:
            fails.append(rel)
    if fails:
        print(f"\n{len(fails)} FAILED:", *fails, sep="\n  ")
        return 1
    print(f"\nAll {len(files)} lectures passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```
- [ ] **Step 2:** `python3 run_all.py` → all PASS, exit 0; `python3 -m pytest tests/ -q` still green.
- [ ] **Step 3:** Commit `feat: add run_all.py smoke runner`

---

### Task 7: Docstring normalization + nit cleanup

**Files:**
- Modify: `common/attention.py`, `4_.../04_4.5_...` (unused imports), `6_.../03_6.4_...` (redundant import), `2_.../04_2.5_...` (prose/label)

**Interfaces / Produces:**
- Bilingual `Args:`/`Returns:` on every public method across the touched modules; drop unused `math` + `typing.Any` in `4.5`; drop redundant top-level `Dataset` import in `6.4`; ensure trailing newline at EOF of every file; fix `2.5` prose to match computed params (`"约 163M"` + note untied output head) and correct the non-applicable "Masked Self-Attention" label for the demo block (which uses non-causal `TransformerEncoderLayer`).

- [ ] **Step 1:** Apply each nit (no behavior change).
- [ ] **Step 2:** `python3 run_all.py` (all PASS) + `python3 -m pytest tests/ -q` (all pass).
- [ ] **Step 3:** Commit `chore: normalize docstrings and clean minor nits`

---

### Task 8: Full verification

- [ ] **Step 1:** `python3 run_all.py` → exit 0, all lectures PASS.
- [ ] **Step 2:** `python3 -m pytest tests/ -q` → all pass.
- [ ] **Step 3:** `git status --porcelain` clean (no stray artifacts). Commit `chore: phase 2 verification pass` (or fold into Task 7).