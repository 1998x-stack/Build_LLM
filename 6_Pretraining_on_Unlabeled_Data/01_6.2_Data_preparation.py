# 01_6.2_Data_preparation

"""
Lecture: /6_Pretraining_on_Unlabeled_Data
Content: 6.2 数据准备
"""

import re
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class GPTDataset(Dataset):
    """滑动窗口构造 (输入, 目标) 样本的语言模型数据集。

    每个样本: inputs = tokens[i : i+context], targets = tokens[i+1 : i+context+1]。
    """

    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.token_ids[index: index + self.context_length]
        targets = self.token_ids[index + 1: index + self.context_length + 1]
        return torch.tensor(inputs), torch.tensor(targets)


def build_vocab(text: str) -> Dict[str, int]:
    """从文本构建词汇表 (简单空格分词, 排序保证确定性)。"""
    tokens = sorted(set(re.findall(r"\S+", text.lower())))
    return {token: i for i, token in enumerate(tokens)}


if __name__ == "__main__":
    text = ("the cat sat on the mat. the dog sat too. "
            "the cat ran fast across the yard.")
    vocab = build_vocab(text)
    token_ids = [vocab[w] for w in re.findall(r"\S+", text.lower())]

    ctx_len = 4
    dataset = GPTDataset(token_ids, ctx_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    print("词汇表大小:", len(vocab))
    print("数据集样本数:", len(dataset))
    inputs, targets = next(iter(loader))
    print("输入批次形状:", tuple(inputs.shape), "目标批次形状:", tuple(targets.shape))
    print("示例输入 token id:", inputs[0].tolist(), "-> 目标:", targets[0].tolist())
