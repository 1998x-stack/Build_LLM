# Build_LLM — Phase 3 (Professionalize Docs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the documentation professional and consistent for a shareable repo — a proper README, uniform lecture `.md` structure with cross-references, and frozen/clearly-documented scaffolding utilities. (This is the final phase of the 3-phase plan.)

**Architecture:** No code-logic changes — docs only. README becomes the entry point (overview, learning path, structure, usage, canonical-module note, TOC, verification). Each lecture `.md` gets a normalized structure (H2 topic heading, short 简介, existing body preserved, a `## 相关阅读 / Related` cross-link footer). Scaffolding scripts get a "generator only, do not re-run" header note.

**Tech Stack:** Markdown only. Verification is link/consistency checks (no pytest — Phase 2 test suite must stay green, but docs don't change code).

## Global Constraints

- Working dir: `/Users/x/Desktop/1998x-stack/00-仓库/04-深度学习与CV/从零实现与消融/Build_LLM`
- **Do NOT alter any `.py`, `common/`, `tests/`, or `run_all.py` content** (docs only). `git diff` for each task must touch only docs.
- `python3 run_all.py` still exits 0 and `python3 -m pytest tests/ -q` still passes (must remain true — verify after each doc task that no code changed).
- All Markdown links must resolve to real files in the repo (verify with the link-check command in Task 5).
- Bilingual: preserve existing Chinese content; README can be bilingual (Chinese primary + English) consistent with the code.
- Markdown lint cleanliness: no `# -level` placeholder headings (like `# accuracy, precision...`), no `./` double slash in README links, no bare malformed anchors.
- Commit after each task: `docs: ...`.

---

### Task 1: Rewrite `README.md` (professional overview)

**Files:**
- Rewrite: `README.md`

**Interfaces / Produces:**
- A complete, professional README. Content MUST include, in order:
  1. **Title + one-line overview** (build a GPT-style LLM from scratch, following Raschka's *Build a Large Language Model*), bilingual.
  2. **Learning path**: chapters 2–6 with a one-line description each (2=concepts, 3=text/tokenization, 4=attention, 5=GPT, 6=pretraining+eval).
  3. **Repo structure tree** (`common/`, `tests/`, `run_all.py`, per-chapter lecture dirs, `.pdf` reference).
  4. **Setup / usage**: `python3 -m pip install --user torch numpy tiktoken pytest`, then `python3 run_all.py` (smoke-run every lecture) and `python3 -m pytest tests/ -q` (test suite).
  5. **Canonical module note (core design decision)**: `common/attention.py` = single source of truth for MHA/GPTBlock/GPT; lecture files `4.2/4.3/5.2/6.3/6.4` carry copy-consistent copies kept in sync (header comment `与 common/attention.py 保持同步`).
  6. **Chapter table of contents**: for each chapter, a table/link list of lectures linking BOTH the `.py` and the `.md` with real relative paths (fix the `./` double slash and the `self-attention`→`self_attention` label you saw in the old README).
  7. **`位置`/status**: which chapters are implemented vs doc-only (2.x conceptual are doc-only; document this).
  8. **Scaffolding note**: `main.py` / `modify_files.sh` are **generator utilities used at repo creation; do not re-run** (they would overwrite hand-written content).
  9. **Credits / references**: link Raschka repo + book (PDF in repo), the ChatGPT transcript link if kept.

- [ ] **Step 1: Write the new README** per the above (bilingual; preserve the existing `.` link targets so they resolve — verify with the Task 5 link checker). Use `./Chapter/File` (single slash), not `.//Chapter/File`.
- [ ] **Step 2: Verify** every relative link target exists: `python3 - <<'PY'` scanning for `](...)` and `os.path.exists` — no broken links.
- [ ] **Step 3: Confirm no code changed**: `git status --porcelain` shows only `README.md` modified; `python3 run_all.py` unchanged (exit 0) and `pytest tests/ -q` still 14 passed (run once).
- [ ] **Step 4: Commit** `docs: rewrite README as professional overview`

---

### Task 2: Normalize lecture `.md` structure (chapters 2–3)

**Files:**
- Modify: all `.md` under `2_Understanding_Large_Language_Models/` and `3_Working_with_Text_Data/`

**Interfaces / Produces:**
For each `.md`: keep the `# <filename>` title and the `"""Lecture..."""` header; then:
1. Rename the **first** content heading from `### N.N 中文标题` to `## N.N 中文标题` (top-level topic at H2).
2. **Remove any stray `# `-level headings** mid-file that are actually copied-code comment lines (e.g. `# 读取文本`, `# 使用正则表达式…`, `# 初始化模型`): convert them to `#### ` sub-headings or back into prose/lists as appropriate — never leave a fake H1.
3. Keep all existing body content intact (do not delete factual paragraphs).
4. Append a footer:
```
---
### 相关代码 / Related code
- 对应实现: [`NN_N.T_title.py`](NN_N.T_title.py)  (same directory, relative link)
```
and where cross-references are natural (e.g. tokenizer → BPE), add one "参见" line.

- [ ] **Step 1:** Apply normalization to every chapter-2 and chapter-3 `.md` (exact file list: 5 in ch2, 5 in ch3).
- [ ] **Step 2:** Verify: no surviving `^# ` headings except the H1 title + the H3 footer (grep incl. `grep` manual); body content essentially preserved; run `git diff --stat` to confirm docs-only.
- [ ] **Step 3:** Run `python3 run_all.py` (exit 0) + `python3 -m pytest tests/ -q` (unchanged) to confirm no code drift.
- [ ] **Step 4: Commit** `docs: normalize lecture .md structure (chapters 2-3)`

---

### Task 3: Normalize lecture `.md` structure (chapters 4–6)

**Files:**
- All `.md` under `4_Coding_Attention_Mechanisms/`, `5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/`, `6_Pretraining_on_Unlabeled_Data/`

**Interfaces / Produces:**
Same rules as Task 2, applied to chapters 4–6 (5 files + 4 files + 4 files = 13 `.md`). Special cases to fix explicitly:
- `6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.md`: remove the stray `# accuracy, precision, recall, f1 = evaluate_model(model, eval_data_loader)` comment-line heading — it's a placeholder fragment; delete it (or fold into prose as a code comment inside a code block).
- `3.3`/`6.3`/`6.4` mid-file `# code` headings → `##` or prose per the rules (those belong to THIS task only if in ch4-6; ch3 done in Task 2).
- Attention/GPT lectures: add cross-references e.g. `4.5 → 4.2/4.3` and `4.4 → 5.2`, `6.3/6.4 → common/attention.py` + `5.2`.

- [ ] **Step 1:** Apply to chapters 4–6 exact list (13 files; ch2-3 already done).
- [ ] **Step 2:** Verify same as Task 2 (grep for stray `# ` H1 headings; `git diff --stat` docs-only).
- [ ] **Step 3:** `python3 run_all.py` (exit 0) + `python3 -m pytest tests/ -q` (unchanged).
- [ ] **Step 4:** Commit `docs: normalize lecture .md structure (chapters 4-6)`

---

### Task 4: Scaffolding-script freeze + README canonical link check

**Files:**
- Edit: `main.py` header comment, `modify_files.sh` header echo
- Doc note in README already covered by Task 1 step 8; here ensure the script files carry the "generator only" header too.

**Interfaces / Produces:**
- Add a header comment block to `main.py` and a leading `echo`/comment to `modify_files.sh`, both stating: this is a one-time repo-scaffolding generator; **do not re-run** (it overwrites README/skeletons). No behavior change (still runnable but documented as unsafe to re-run).
- Optionally wrap `main.py`'s main block so a re-run refuses unless an env flag is set (e.g. `BUILD_LLM_GEN=1`) — optional; if added, update `run_all.py`'s skip set is NOT needed because run_all already skips `main.py`.

- [ ] **Step 1:** Add the header/guard comment to `main.py` + `modify_files.sh` (behavior unchanged — no re-run guard unless you add the `FORCE_BUILD_LLM_GEN=1` guard; if you add the guard, add a matching echo in `modify_files.sh`).
- [ ] **Step 2:** Verify README + all `.md` links resolve with the Task-5 checker; `git diff --stat` touches only `main.py`, `modify_files.sh`, `README.md`, `.md` files.
- [ ] **Step 3:** `python3 run_all.py` (exit 0) + `python3 -m pytest tests/ -q` (unchanged).
- [ ] **Step 4:** Commit `docs: freeze scaffolding scripts as create-only generators`

---

### Task 5: Full doc + repo verification

**Files:**
- none (verification only) — may add `tools/check_links.py` if you like

**Interfaces / Produces:**
- Link checker (bash or a tiny python script) that scrapes every Markdown file in the repo (README + all lectures + docs) and asserts every `](relative-path)` target exists on disk. Print count of links checked and any broken ones; fail (non-zero) if any broken.

- [ ] **Step 1:** Write/run a link checker. One-liner approach:
```bash
python3 - <<'PY'
import glob,os,re
bad=[]
for md in glob.glob('**/*.md',recursive=True):
    for m in re.findall(r'\]\(([^)#]+)\)', open(md,encoding='utf-8').read()):
        t=m.split('#')[0]
        if t and not t.startswith(('http://','https://')) and not os.path.exists(t):
            bad.append((md,t))
print('broken:',bad); assert not bad
PY
```
- [ ] **Step 2:** `python3 run_all.py` → exit 0; `python3 -m pytest tests/ -q` → unchanged (14 pass).
- [ ] **Step 3:** `git status --porcelain` → clean (no stray `.npy`).
- [ ] **Step 4:** Commit `chore: phase 3 verification` (or nothing to commit).