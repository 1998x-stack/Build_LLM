# 04_4.5_Implementing_self_attention_mechanisms

"""
Lecture: /4_Coding_Attention_Mechanisms
Content: 4.5 从零实现自注意力机制
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import Any


def compute_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                            causal: bool = True) -> np.ndarray:
    """
    用 NumPy 从零实现缩放点积自注意力 (scaled dot-product self-attention)。

    步骤: 计算得分 Q·Kᵀ/√d → (可选) 因果掩码 → softmax 归一化 → 加权求和。

    Args:
        q (np.ndarray): 查询矩阵, 形状 (seq_len, d)。
        k (np.ndarray): 键矩阵, 形状 (seq_len, d)。
        v (np.ndarray): 值矩阵, 形状 (seq_len, d)。
        causal (bool): 是否使用因果掩码 (每个 token 只能看到自己及之前的位置)。

    Returns:
        np.ndarray: 自注意力的输出, 形状 (seq_len, d)。
    """
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)                       # (seq_len, seq_len)
    if causal:
        seq_len = q.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))  # 数值稳定
    weights = exp / exp.sum(axis=-1, keepdims=True)
    return weights @ v


def compute_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            causal: bool = True) -> torch.Tensor:
    """
    等价的 PyTorch / einsum 实现，用于与 NumPy 版本做数值校验。

    Args:
        q (torch.Tensor): 查询矩阵, 形状 (seq_len, d)。
        k (torch.Tensor): 键矩阵, 形状 (seq_len, d)。
        v (torch.Tensor): 值矩阵, 形状 (seq_len, d)。
        causal (bool): 是否使用因果掩码。

    Returns:
        torch.Tensor: 自注意力输出, 形状 (seq_len, d)。
    """
    d_k = q.shape[-1]
    scores = torch.einsum("qd,kd->qk", q, k) / np.sqrt(d_k)
    if causal:
        seq_len = q.shape[0]
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return weights @ v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq_len, d = 6, 8
    X = rng.normal(size=(seq_len, d))

    out_np = compute_attention_numpy(X, X, X)
    out_pt = compute_attention_torch(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32),
    )
    diff = float(np.abs(out_np - out_pt.numpy()).max())
    print("NumPy 输出形状:", out_np.shape)
    print("因果掩码下第一行仅依赖自身，其余各 position 正常。")
    print(f"NumPy 与 Torch 最大数值差异: {diff:.3e}")