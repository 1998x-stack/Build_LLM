# 01_5.2_Implementing_GPT_model

"""
Lecture: /5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text
Content: 01_5.2_Implementing_GPT_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Tuple, List

class GPTSelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        """
        GPT模型中的自注意力机制。

        参数:
        embed_size (int): 嵌入向量的维度。
        heads (int): 多头注意力机制的头数。
        """
        super(GPTSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        values (torch.Tensor): 值向量，形状为(batch_size, value_len, embed_size)。
        keys (torch.Tensor): 键向量，形状为(batch_size, key_len, embed_size)。
        query (torch.Tensor): 查询向量，形状为(batch_size, query_len, embed_size)。
        mask (torch.Tensor, optional): 掩码张量，形状为(batch_size, 1, 1, key_len)。

        返回:
        torch.Tensor: 多头自注意力机制的输出，形状为(batch_size, query_len, embed_size)。
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out


class GPTBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int):
        """
        GPT模型中的单个解码器层。

        参数:
        embed_size (int): 嵌入向量的维度。
        heads (int): 多头注意力机制的头数。
        dropout (float): dropout概率。
        forward_expansion (int): 前馈神经网络的扩展维度。
        """
        super(GPTBlock, self).__init__()
        self.attention = GPTSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量，形状为(batch_size, seq_len, embed_size)。
        mask (torch.Tensor, optional): 掩码张量，形状为(batch_size, 1, 1, key_len)。

        返回:
        torch.Tensor: 解码器层的输出，形状为(batch_size, seq_len, embed_size)。
        """
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_layers: int, heads: int, device: str, forward_expansion: int, dropout: float, max_length: int):
        """
        GPT模型。

        参数:
        vocab_size (int): 词汇表大小。
        embed_size (int): 嵌入向量的维度。
        num_layers (int): 解码器层数。
        heads (int): 多头注意力机制的头数。
        device (str): 设备（'cpu'或'cuda'）。
        forward_expansion (int): 前馈神经网络的扩展维度。
        dropout (float): dropout概率。
        max_length (int): 输入序列的最大长度。
        """
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [GPTBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入张量，形状为(batch_size, seq_len)。
        mask (torch.Tensor, optional): 掩码张量，形状为(batch_size, 1, 1, key_len)。

        返回:
        torch.Tensor: 模型的输出，形状为(batch_size, seq_len, vocab_size)。
        """
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, mask)

        out = self.fc_out(out)

        return out

# 示例使用
if __name__ == "__main__":
    # 定义模型参数
    vocab_size = 10000
    embed_size = 256
    num_layers = 6
    heads = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward_expansion = 4
    dropout = 0.1
    max_length = 100

    # 初始化模型
    model = GPT(vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)

    # 打印模型结构
    print(model)