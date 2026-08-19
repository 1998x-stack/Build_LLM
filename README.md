# Build a Large Language Model（从零实现 GPT）

> 从零实现一个 GPT 风格的大语言模型，逐章对照 Sebastian Raschka 的 *Build a Large Language Model (From Scratch)*（原书 PDF 见仓库根目录 [Build a Large Language Model.pdf](./Build a Large Language Model.pdf)）。
> Build a GPT-style LLM from scratch, following Sebastian Raschka's *Build a Large Language Model*, chapter by chapter.

本仓库是**教育性质**的从零实现仓库（第 2–6 章），每个小节（lecture）包含一份可独立运行的 `.py` 实现和一份配套的 `.md` 讲解文档。代码全程使用中文注释与讲解，便于中文学习者对照阅读。

---

## 学习路径 / Learning Path

按顺序阅读，每章建立一层能力：

| 章节 | 主题 | 一句话说明 |
|------|------|-----------|
| [第 2 章](#chapter-2) | 理解大语言模型 | 概念启蒙：LLM 是什么、应用场景、构建与使用阶段的整体图景（概念性/无代码）。 |
| [第 3 章](#chapter-3) | 文本数据与分词 | 动手处理文本：word embeddings、token 化、token→ID、特殊上下文 token、BPE（字节对编码）。 |
| [第 4 章](#chapter-4) | 注意力机制 | 从自注意力到多头注意力（MHA），以及 Encoder/Decoder 架构与标准注意力实现。 |
| [第 5 章](#chapter-5) | 从零实现 GPT 模型 | 组装 GPT 模型：GPT 基础、完整实现（含归一化、掩码注意力、Transformer 模块）、训练与文本生成应用。 |
| [第 6 章](#chapter-6) | 基于无标注数据的预训练 | 预训练：数据准备、预训练过程、以及用评估指标（困惑度/准确率）评估预训练模型。 |

### 章节目录 / Chapter Table of Contents

> 每个课程同时提供 `.py`（代码实现）与 `.md`（讲解笔记）两份文件，链接均为仓库内的真实相对路径。

#### <a id="chapter-2"></a>第 2 章 · 理解大语言模型 / Understanding Large Language Models

| Lecture | 代码 `.py` | 笔记 `.md` |
|---------|-----------|-----------|
| 2.1 What is an LLM | [py](./2_Understanding_Large_Language_Models/00_2.1_What_is_a_LLM.py) | [md](./2_Understanding_Large_Language_Models/00_2.1_What_is_a_LLM.md) |
| 2.2 Applications of LLMs | [py](./2_Understanding_Large_Language_Models/01_2.2_Applications_of_LLMs.py) | [md](./2_Understanding_Large_Language_Models/01_2.2_Applications_of_LLMs.md) |
| 2.3 Stages of building and using LLMs | [py](./2_Understanding_Large_Language_Models/02_2.3_Stages_of_building_and_using_LLMs.py) | [md](./2_Understanding_Large_Language_Models/02_2.3_Stages_of_building_and_using_LLMs.md) |
| 2.4 Utilizing large datasets | [py](./2_Understanding_Large_Language_Models/03_2.4_Utilizing_large_datasets.py) | [md](./2_Understanding_Large_Language_Models/03_2.4_Utilizing_large_datasets.md) |
| 2.5 A closer look at the GPT architecture | [py](./2_Understanding_Large_Language_Models/04_2.5_A_closer_look_at_the_GPT_architecture.py) | [md](./2_Understanding_Large_Language_Models/04_2.5_A_closer_look_at_the_GPT_architecture.md) |

> **状态**：第 2 章为**概念性讲解**，`.py` 为指向对应 `.md` 的占位 stub，无独立代码实现（详见 [状态说明](#位置status)）。

#### Chapter 3：文本数据处理 / Working with Text Data

| Lecture | `.py` | `.md` |
|---------|-------|-------|
| 3.1 Understanding word embeddings | [py](./3_Working_with_Text_Data/00_3.1_Understanding_word_embeddings.py) | [md](./3_Working_with_Text_Data/00_3.1_Understanding_word_embeddings.md) |
| 3.2 Tokenizing text | [py](./3_Working_with_Text_Data/01_3.2_Tokenizing_text.py) | [md](./3_Working_with_Text_Data/01_3.2_Tokenizing_text.md) |
| 3.3 Converting tokens into token IDs | [py](./3_Working_with_Text_Data/02_3.3_Converting_tokens_into_token_IDs.py) | [md](./3_Working_with_Text_Data/02_3.3_Converting_tokens_into_token_IDs.md) |
| 3.4 Adding special context tokens | [py](./3_Working_with_Text_Data/03_3.4_Adding_special_context_tokens.py) | [md](./3_Working_with_Text_Data/03_3.4_Adding_special_context_tokens.md) |
| 3.5 Byte pair encoding | [py](./3_Working_with_Text_Data/04_3.5_Byte_pair_encoding.py) | [md](./3_Working_with_Text_Data/04_3.5_Byte_pair_encoding.md) |

#### Chapter 4：编码注意力机制 / Coding Attention Mechanisms

| Lecture | `.py` | `.md` |
|---------|-------|-------|
| 4.1 Introduction to attention mechanisms | [py](./4_Coding_Attention_Mechanisms/00_4.1_Introduction_to_attention_mechanisms.py) | [md](./4_Coding_Attention_Mechanisms/00_4.1_Introduction_to_attention_mechanisms.md) |
| 4.2 Self-attention mechanisms | [py](./4_Coding_Attention_Mechanisms/01_4.2_Self_attention_mechanisms.py) | [md](./4_Coding_Attention_Mechanisms/01_4.2_Self_attention_mechanisms.md) |
| 4.3 Multi-head attention mechanisms | [py](./4_Coding_Attention_Mechanisms/02_4.3_Multi_head_attention_mechanisms.py) | [md](./4_Coding_Attention_Mechanisms/02_4.3_Multi_head_attention_mechanisms.md) |
| 4.4 Encoder and decoder architectures | [py](./4_Coding_Attention_Mechanisms/03_4.4_Encoder_and_decoder_architectures.py) | [md](./4_Coding_Attention_Mechanisms/03_4.4_Encoder_and_decoder_architectures.md) |
| 4.5 Implementing self-attention mechanisms | [py](./4_Coding_Attention_Mechanisms/04_4.5_Implementing_self_attention_mechanisms.py) | [md](./4_Coding_Attention_Mechanisms/04_4.5_Implementing_self_attention_mechanisms.md) |

> 说明：4.1 为概念性讲解（stub），其余 4.2–4.5 均已实现。4.2/4.3 与 `common/attention.py` 保持同步（见 [核心设计](#核心设计规范-canonical-modules)）。

#### Chapter 5：从零实现 GPT 模型 / Implementing a GPT Model from Scratch

| Lecture | `.py` | `.md` |
|---------|-------|-------|
| 5.1 Basics of GPT model | [py](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/00_5.1_Basics_of_GPT_model.py) | [md](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/00_5.1_Basics_of_GPT_model.md) |
| 5.2 Implementing GPT model | [py](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.py) | [md](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.md) |
| 5.3 Training GPT model | [py](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/02_5.3_Training_GPT_model.py) | [md](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/02_5.3_Training_GPT_model.md) |
| 5.4 Applications of GPT model | [py](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/03_5.4_Applications_of_GPT_model.py) | [md](./5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/03_5.4_Applications_of_GPT_model.md) |

> 说明：5.2 与 `common/attention.py` 保持同步。

#### Chapter 6：基于无标注数据的预训练 / Pretraining on Unlabeled Data

| Lecture | `.py` | `.md` |
|---------|-------|-------|
| 6.1 Concept of pretraining | [py](./6_Pretraining_on_Unlabeled_Data/00_6.1_Concept_of_pretraining.py) | [md](./6_Pretraining_on_Unlabeled_Data/00_6.1_Concept_of_pretraining.md) |
| 6.2 Data preparation | [py](./6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py) | [md](./6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.md) |
| 6.3 Pretraining process | [py](./6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py) | [md](./6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.md) |
| 6.4 Evaluating pretrained model | [py](./6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py) | [md](./6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.md) |

> 说明：6.1 为概念性讲解（stub），6.2–6.4 均已实现；6.3/6.4 与 `common/attention.py` 保持同步。

---

## 仓库结构 / Repository Structure

```
Build_LLM/
├── 2_Understanding_Large_Language_Models/   # 第2章：概念讲解（.py stub + .md）
│   ├── 00_2.1_What_is_a_LLM.{py,md}
│   └── ...05 个 lecture
├── 3_Working_with_Text_Data/                 # 第3章：文本/分词（.py + .md）
├── 4_Coding_Attention_Mechanisms/            # 第4章：注意力机制（.py + .md）
├── 5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/  # 第5章：GPT 模型
├── 6_Pretraining_on_Unlabeled_Data/          # 第6章：预训练与评估
│
├── common/
│   ├── __init__.py
│   └── attention.py                          # 多头注意力/GPT 模块的单一事实来源 (canonical)
├── tests/                                    # 单元测试（pytest）
│   ├── test_attention.py
│   ├── test_bpe.py
│   ├── test_gpt.py
│   ├── test_pretraining.py
│   └── test_tokenizer.py
├── run_all.py                                # 冒烟测试：运行每个 lecture .py 并报告 PASS/FAIL
├── main.py                                   # (生成器，见下注) 仓库脚手架生成脚本，勿重跑
├── modify_files.sh                           # (生成器) 批量改名脚本，勿重跑
├── "Build a Large Language Model.pdf"        # Raschka 原书 PDF 参考
└── README.md                                 # 本文件
```

> 注：`main.py` 与 `modify_files.sh` 是**仓库创建时的一次性脚手架生成工具**，会覆盖/改写已手工书写的内容 — **请勿再运行**。日常只使用 `run_all.py` 与 `pytest`（见下）。

---

## 环境与使用 / Setup & Usage

```bash
# 1) 安装依赖（用户级安装即可）
python3 -m pip install --user torch numpy tiktoken pytest

# 2) 冒烟运行每一个 lecture 代码（PASS/FAIL 汇总，非零退出码表示有失败）
python3 run_all.py

# 3) 运行单元测试套件（共 14 个用例）
python3 -m pytest tests/ -q
```

---

## 核心设计规范 / Canonical Modules

**`common/attention.py` 是「多头注意力 + GPT 模块」（`MultiHeadAttention`、`GPTBlock`、`GPT`）的单一事实来源。**

对应到各章节的实现文件（`4.2`、`4.3`、`5.2`、`6.3`、`6.4`）均内嵌与之一致的**拷贝副本**，并在文件头以 `# 与 common/attention.py 保持同步 (canonical) — 见 README` 标注。修改前请确认在所有副本一并同步：

```
common/attention.py                                (canonical / 单一事实来源)
4_Coding_Attention_Mechanisms/01_4.2_..._self*.py
4_Coding_Attention_Mechanisms/02_4.3_..._multi*.py
5_.../01_5.2_..._GPT*.py
6_.../02_6.3_..._pretrain*.py
6_.../03_6.4_..._evaluat*.py
```

---

## 状态 / Status

| 章节 | 状态 | 实现说明 |
|------|------|---------|
| 第 2 章 | 概念性 / doc-only | 2.1–2.5 全部为**讲解文档**；对应 `.py` 为指向 `.md` 的 stub（无代码实现）。 |
| 第 3 章 | 已实现 | 3.1–3.5 全部有可运行 `.py` + 配套 `.md`。 |
| 第 4 章 | 4.1 doc-only / 4.2–4.5 已实现 | 4.1 为概念 stub；4.2–4.5 实现注意力。 |
| 第 5 章 | 已实现 | 5.1–5.4 全部有 `.py` + `.md`。 |
| 第 6 章 | 6.1 doc-only / 6.2–6.4 已实现 | 6.1 为概念 stub；6.2–6.4 实现预训练与评估。 |

> `python3 run_all.py` 仅对**已实现**的 lecture 运行（stub 也各自无害退出），全部应 PASS。

---

## 致谢与参考 / Credits & References

- 原仓库（代码/结构灵感）：[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- 原书（PDF，仓库根目录）：[Build a Large Language Model.pdf](./Build a Large Language Model.pdf)
- 配套 ChatGPT 讲解（可选）：https://chatgpt.com/share/b8ecb0bb-3f85-48ff-aaae-768851d8910e