# Build a Large Language Model（从零实现 GPT）

一个以“教学可读性 + 数学正确性 + 可验证工程实践”为目标的 GPT 从零实现仓库。

本仓库保留逐章学习路径，同时增加一套 canonical implementation，避免章节代码长期复制后产生数学与行为漂移。参考 Sebastian Raschka 的 Build a Large Language Model (From Scratch) 与 LLMs-from-scratch 项目，但代码与测试在本仓库中独立维护。

## 当前设计

仓库分为两层：

1. Chapters 2–6：教学层。用于逐步解释 tokenization、attention、GPT、pretraining 与 evaluation。
2. src/build_llm：canonical core。用于正确性测试、训练、生成、checkpoint 与性能对比。

核心实现遵循以下约束：

- Q/K/V 在完整 embedding 维度上投影后再 split heads。
- GPT block 使用 pre-norm residual structure。
- MLP 使用 GELU。
- GPT 输出头之前存在 final LayerNorm。
- causal mask 由 attention module 管理，不依赖构造时 device 字符串。
- 模型设备由 model.to(device) 决定；generation/evaluation 跟随模型当前设备。
- manual attention 与 PyTorch SDPA 后端可做数值一致性验证。

详细设计见 docs/architecture.md。

## 快速开始

要求 Python 3.10+。

    python -m pip install -e ".[dev]"
    python -m pytest -q
    python run_all.py

也可以运行统一质量门：

    make check

训练一个极小 GPT 示例：

    python examples/train_tiny_gpt.py

比较 manual attention 与 PyTorch SDPA：

    python tools/benchmark_attention.py

## Repository Structure

    Build_LLM/
    ├── 2_Understanding_Large_Language_Models/
    ├── 3_Working_with_Text_Data/
    ├── 4_Coding_Attention_Mechanisms/
    ├── 5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/
    ├── 6_Pretraining_on_Unlabeled_Data/
    ├── src/build_llm/
    │   ├── config.py
    │   ├── nn/
    │   ├── model/
    │   ├── data/
    │   ├── training/
    │   ├── generation.py
    │   └── evaluation.py
    ├── common/
    ├── examples/
    ├── tools/
    ├── tests/
    ├── pyproject.toml
    └── run_all.py

common/attention.py 是旧教学接口的兼容 facade。新代码优先从 build_llm 导入。

## Correctness Gates

测试不仅检查 shape，还检查以下性质：

- full-width Q/K/V projection
- causal future-token isolation
- manual attention 与 SDPA 数值对齐
- finite backward gradients
- pre-norm + GELU + final LayerNorm
- next-token dataset shift
- tiny-corpus training loss decreases
- finite perplexity
- deterministic greedy generation
- checkpoint save/load roundtrip
- lecture-to-canonical conformance
- CPU 必测；CUDA/MPS 在可用环境中条件测试

## Performance

manual backend 用于学习公式和调试；sdpa backend 用于展示现代 PyTorch optimized attention。两者共享同一模型接口，并通过数值测试防止语义漂移。

## Repository Hygiene

原仓库中的 macOS .DS_Store、一次性脚手架脚本以及直接 vendored 的书籍 PDF 不再保留在版本控制中。书籍请通过作者/出版社的合法渠道获取。

## References

- Sebastian Raschka, LLMs-from-scratch: https://github.com/rasbt/LLMs-from-scratch
- Build a Large Language Model (From Scratch), Manning / Sebastian Raschka

## Scope

这是教育与实验仓库，不以分布式大规模训练框架为目标。当前重点是：正确、可读、可测、可复现，并保留从手写实现到优化实现的清晰演进路径。
