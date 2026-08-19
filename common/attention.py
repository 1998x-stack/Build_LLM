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
        """多头注意力前向计算。

        Args:
            v (torch.Tensor): 值序列, 形状 (N, L, embed_size)。
            k (torch.Tensor): 键序列, 形状 (N, L, embed_size)。
            q (torch.Tensor): 查询序列, 形状 (N, L, embed_size)。
            mask (Optional[torch.Tensor]): 形状 (L, L) 的 0/1 掩码; 为 0 的位置填充为极小值。

        Returns:
            torch.Tensor: 自注意力输出, 形状 (N, L, embed_size)。
        """
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
        """前向 GPT 块: 自注意力 + 前馈, 均带残差/LayerNorm/dropout。

        Args:
            x (torch.Tensor): 输入序列, 形状 (N, seq_len, embed_size)。
            mask (Optional[torch.Tensor]): 可选注意力掩码。

        Returns:
            torch.Tensor: 输出序列, 形状 (N, seq_len, embed_size)。
        """
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
        """前向 GPT 语言模型, 内置因果掩码。

        Args:
            x (torch.Tensor): token 索引, 形状 (N, seq_len)。

        Returns:
            torch.Tensor: 词表 logits, 形状 (N, seq_len, vocab_size)。
        """
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)   # 内置因果掩码
        for layer in self.layers:
            out = layer(out, mask)
        return self.fc_out(out)
