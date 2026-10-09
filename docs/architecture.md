# Build_LLM Architecture

## Goals

Build_LLM 同时服务两个目标：逐章教学，以及一套可被测试和复用的 canonical GPT core。教学层强调理解过程；canonical core 强调行为唯一性、数学正确性和工程可验证性。

## Layering

教学层位于 Chapters 2–6。核心层位于 src/build_llm。

common/attention.py 仅作为兼容 facade，保留旧教程构造参数。新功能不应继续堆入 common。

## Canonical GPT

GPTModel 的数据流：

    token embedding + position embedding
        -> embedding dropout
        -> N x pre-norm TransformerBlock
        -> final LayerNorm
        -> vocabulary projection

TransformerBlock：

    x = x + Attention(LN(x))
    x = x + MLP(LN(x))

MLP 使用 GELU。

## Attention Correctness

旧实现先把 embedding reshape 成多个 head，再分别执行 head_dim 到 head_dim 的线性层。这会把 Q/K/V projection 限制为按 head 分块的映射。

当前实现先执行完整 d_model 到 d_model 的 Q/K/V projection，再 split heads。这与标准 multi-head attention 的参数化一致。

CausalSelfAttention 提供两个后端：

- manual：显式 QK transpose、scale、causal mask、softmax、weighted sum，便于教学。
- sdpa：PyTorch scaled_dot_product_attention，作为优化路径。

测试要求 dropout 关闭时二者数值一致。

## Device Policy

模型对象不保存 cpu、cuda 或 mps 字符串。设备状态以参数实际所在 device 为唯一事实来源。

因此：

- forward 输入应位于模型 device。
- generation 会将 seed token 移到模型 device。
- evaluation/training 会将 batch 移到模型 device。
- checkpoint 默认可 map 到 CPU 后再由调用方迁移。

## Training Boundary

Trainer 负责 optimizer、gradient clipping、seed 与 train/eval loop。GPTModel 只负责 forward，不包含 optimizer 或训练状态。

NextTokenDataset 只负责构建 shifted next-token samples。

generation.py 和 evaluation.py 与训练循环解耦。

## Verification Strategy

质量门分为：

1. unit tests：attention、config、dataset、generation。
2. mathematical invariants：causal isolation、manual/SDPA equivalence、finite gradients。
3. integration tests：tiny-corpus overfit、checkpoint roundtrip。
4. conformance tests：章节包装层必须复用 canonical implementation。
5. smoke tests：每个 lecture script 独立执行。
6. CI：ruff + pytest + lecture smoke。

## Performance Path

P3 不把 optimized code 混入教学公式。manual 与 sdpa 使用同一个 CausalSelfAttention 接口，通过 backend 参数切换。tools/benchmark_attention.py 用相同权重对比 latency 和数值误差。

后续如增加 torch.compile、KV cache、mixed precision，应保持同样原则：reference implementation 保留，优化实现通过可比较测试接入。
