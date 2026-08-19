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
    # 以 GPT-2 (163M, 含未绑定输出头) 的量级为参考: 12 层, 768 维, 12 头
    cfg = dict(vocab_size=50257, embed_size=768)
    model = GPT(vocab_size=cfg["vocab_size"], embed_size=cfg["embed_size"],
                num_layers=12, heads=12, max_len=1024)
    total = sum(p.numel() for p in model.parameters())
    layers = len(model.blocks)

    print("===== GPT 架构概览 (GPT-2 约 163M 参数量级, 含未绑定的输出头) =====")
    print(f"词典大小 (vocab_size):        {cfg['vocab_size']}")
    print(f"嵌入维度 (embed_size):        {cfg['embed_size']}")
    print(f"解码器层数 (num_layers):      {layers}")
    print(f"注意力头数 (num_heads):       12")
    print(f"最大上下文长度 (max_len):     1024")
    print(f"总参数量 (total params):      {total:,}")
    print()
    print("结构:  token_embedding → 位置编码 → [Self-Attention → FFN]*12 → LayerNorm → Linear(→ vocab)")
