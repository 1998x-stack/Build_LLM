# 02_4.3_Multi-head_attention_mechanisms

"""
Lecture: /4_Coding_Attention_Mechanisms
Content: 02_4.3_Multi-head_attention_mechanisms
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        """
        多头注意力机制的初始化方法。

        参数:
        embed_size (int): 嵌入向量的维度。
        heads (int): 多头注意力机制的头数。
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        values (torch.Tensor): 值向量，形状为(batch_size, value_len, embed_size)。
        keys (torch.Tensor): 键向量，形状为(batch_size, key_len, embed_size)。
        query (torch.Tensor): 查询向量，形状为(batch_size, query_len, embed_size)。
        mask (torch.Tensor, optional): 掩码张量，形状为(batch_size, 1, 1, key_len)。

        返回:
        torch.Tensor: 多头注意力机制的输出，形状为(batch_size, query_len, embed_size)。
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将输入分割成多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 计算注意力权重
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 计算加权和值
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out

# 示例使用
if __name__ == "__main__":
    embed_size = 256
    heads = 8
    query_len = 10
    key_len = 10
    value_len = 10
    batch_size = 64

    values = torch.rand((batch_size, value_len, embed_size))
    keys = torch.rand((batch_size, key_len, embed_size))
    query = torch.rand((batch_size, query_len, embed_size))
    mask = torch.ones((batch_size, 1, 1, key_len))

    multi_head_attention = MultiHeadAttention(embed_size, heads)
    out = multi_head_attention(values, keys, query, mask)
    print(f"Output shape: {out.shape}")
