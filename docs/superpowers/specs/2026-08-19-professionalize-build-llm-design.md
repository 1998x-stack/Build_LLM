# Design: Professionalize & Complete `Build_LLM`

Date: 2026-08-19
Status: Approved (user reviewed Sections 1–4)

## 1. Goal

Make `Build_LLM` a complete, correct, self-contained, shareable "LLM from scratch"
walkthrough. It currently has real implementations only for `3.x`, `4.2`, `4.3`, `5.2`;
the remaining lecture files are empty stubs.

Decisions locked during brainstorming:
- **Phasing (Q1/D):** all of the below, delivered as three independent sub-projects.
- **Audience (Q3):** personal now, shareable later; treat content as self-contained & professional.
- **Language (Q4):** bilingual (Chinese primary + English where it clarifies).
- **Code org (Q6-C):** each lecture `.py` stays self-contained; deduplicate via a canonical module noted in README.
- **Testing (Q7-C):** both a `pytest` suite (for core algorithms) and a `run_all.py` smoke runner.
- **Conceptual split (Q2-C / Q5):** some conceptual sections get runnable demos, others stay doc-only (detailed below).

## 2. Repo Layout (Target)

```
Build_LLM/
├── README.md                  → rewritten professional index/usage (Phase 3)
├── run_all.py                 → smoke runner; executes every lecture __main__ (Phase 2)
├── common/attention.py        → canonical MultiHeadAttention + GPTBlock + GPT (Phase 2)
├── tests/                     → pytest suite for core components (Phase 2)
├── 2_…/ 3_…/ 4_…/ 5_…/ 6_…/   → lecture .py + .md pairs (Phase 1 fills gaps)
└── Build a Large Language Model.pdf  → unchanged reference
└── main.py, modify_files.sh    → frozen generator utilities (documented, not re-run)
```

## 3. Phase 1 — Complete the Gaps

### 3.1 Implement from scratch (real, runnable code)

| File | Content |
|---|---|
| `4.5 Implementing self-attention` | Hand-rolled causal self-attention (numpy-first, then torch einsum): Q·Kᵀ/√d, causal mask, softmax, weighted sum. Fully built out. |
| `4.4 Encoder/decoder architectures` | Forward-passable `EncoderBlock`, `DecoderBlock` (with cross-attention), tiny seq2seq `Transformer`; forward pass on dummy data. |
| `2.5 GPT architecture` | Prints full GPT layer-stack (embed dims, # blocks, heads, param counts) reusing the `5.2`-style GPT. |
| `6.2 Data preparation` | `GPTDatasetV1`-style sliding-window batching → `(inputs, targets)`, PyTorch `Dataset` + dataloader demo. |
| `6.3 Pretraining process` | Full training loop: corpus → tokenize → batches → GPT → cross-entropy → Adam → gradient clipping → per-epoch loss. |
| `6.4 Evaluating pretrained model` | Perplexity + `generate()` text sampling; print samples (finite perplexity). |

### 3.2 Doc-only (`.md` filled, `.py` becomes a clean stub pointing to the `.md`)

Files: `2.1`, `2.2`, `2.3`, `2.4`, `4.1` (attention intro), `6.1` (pretraining concept).
Their `.py` keeps the lecture header plus a one-line comment: `# 概念性章节，见对应 .md`.

### 3.3 Dependency note

`6.3` needs GPT. To keep phases independent, Phase 1 places a **temporary self-contained GPT**
inside `6_…`; Phase 2 dedups it into `common/attention.py`. This is an accepted, deliberate
temporary duplication.

## 4. Phase 2 — Professionalize Code + Tests

- **`common/attention.py`:** canonical `MultiHeadAttention`, `GPTBlock`, `GPT` (the corrected
  versions). Bilingual, type-hinted docstrings. Used by `tests/` and referenced by lecture files.
- **Self-contained lecture copies:** `4.2`, `4.3`, `5.2`, new Ch-6 files get copy-consistent
  copies (identical where overlapping). Header comment: `# 与 common/attention.py 保持同步 (canonical) — 见 README`.
- **Docstring normalization:** consistent bilingual pattern, Args/Returns everywhere, consistent type hints.
- **`tests/` (pytest):**
  - `test_tokenizer.py` — `3.2` round-trip, `3.4` special tokens.
  - `test_bpe.py` — `3.5` fit/encode/decode on `["low","lowest","newer","wider"]` + round-trip.
  - `test_attention.py` — `4.2`/`4.3`/`common` output shapes + causal-mask behavior.
  - `test_gpt.py` — `5.2`/`common` forward shape + regression for the fixed linear-layer dimension bug.
  - `test_pretraining.py` — `6.2` batch shapes, `6.3` loss decreases over epoch, `6.4` perplexity finite + generation text.
  - Copy-consistency tests: matching shapes/behavior between lecture copies and `common/`.
- **`run_all.py`:** discover lecture `*.py`, subprocess-run each with timeout, report PASS/FAIL,
  non-zero exit on failure. Skips `tests/` and `common/`.

## 5. Phase 3 — Professionalize Docs

- **README:** overview, learning path, structure tree, setup/usage, canonical-module note, TOC,
  verification section, credits.
- **`.md` lectures:** normalize heading structure (`# file`, `## 背景/概念`, `## 代码实现`,
  `## 运行结果`, `## 总结`); fill thin conceptual `.md`s; cross-link related lectures; match
  README links to real filenames.
- **Scaffolding scripts frozen:** `main.py` / `modify_files.sh` documented as generator utilities
  only; not re-run (would overwrite hand-written content).

## 6. Non-Goals / YAGNI

- No parallel/multi-model training, no distributed training.
- No telemetry, packaging (no `pyproject.toml`/PyPI), no external CI integration beyond `run_all.py` + pytest exit codes.
- No dedup of `3.x` tokenizers into `common/` (only attention/GPT per the Q6-C decision).

## 7. Verification Strategy

- Every generated `.py` must run standalone (`python3 file.py`).
- `run_all.py` passes (non-zero exit only if a lecture fails).
- `pytest tests/ -q` passes for the core components and regression tests.
- Docs: README links resolve to real files; cross-links resolve.

## 8. Risks / Notes

- Temporary GPT duplication in `6.3` (Phase 1) resolved in Phase 2.
- Hand-rolled `4.5` must be numerically validated against the torch einsum version.
- `pytest` is not installed in the base env; Phase 2 adds it. `torch`/`tiktoken` are installed.
- Bilingual consistency: primary Chinese, English to clarify (existing convention).