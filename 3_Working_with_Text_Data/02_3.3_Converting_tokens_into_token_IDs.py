# 02_3.3_Converting_tokens_into_token_IDs

"""
Lecture: /3_Working_with_Text_Data
Content: 02_3.3_Converting_tokens_into_token_IDs
"""

import re
from typing import List, Dict

class Tokenizer:
    def __init__(self, vocab: Dict[str, int]):
        """
        初始化标记器。

        参数:
        vocab (Dict[str, int]): 词汇表，从字符串标记到整数ID的映射。
        """
        self.str_to_int = vocab  # 词汇表，从字符串到整数的映射
        self.int_to_str = {i: s for s, i in vocab.items()}  # 逆词汇表，从整数到字符串的映射

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为标记ID。

        参数:
        text (str): 输入文本。

        返回:
        List[int]: 标记ID列表。
        """
        # 使用正则表达式拆分文本为标记
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 将标记转换为标记ID，处理未知标记
        ids = [self.str_to_int.get(s, self.str_to_int["<|unk|>"]) for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        将标记ID解码为文本。

        参数:
        ids (List[int]): 标记ID列表。

        返回:
        str: 解码后的文本。
        """
        # 将标记ID转换为字符串标记
        text = " ".join([self.int_to_str[i] for i in ids])
        # 修正标点符号前的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def build_vocab(preprocessed: List[str]) -> Dict[str, int]:
    """
    构建词汇表。

    参数:
    preprocessed (List[str]): 预处理后的标记列表。

    返回:
    Dict[str, int]: 词汇表，从字符串标记到整数ID的映射。
    """
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab

# 示例使用
if __name__ == "__main__":
    # 示例文本
    text = """I HAD always thought Jack Gisburn rather a cheap genius -- though a good fellow enough -- so it was no great surprise to me to hear that, in"""

    # 构建词汇表
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    vocab = build_vocab(preprocessed)

    # 初始化标记器
    tokenizer = Tokenizer(vocab)

    # 编码和解码示例
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
