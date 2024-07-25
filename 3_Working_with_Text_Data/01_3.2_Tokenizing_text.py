# 01_3.2_Tokenizing_text

"""
Lecture: /3_Working_with_Text_Data
Content: 01_3.2_Tokenizing_text
"""

import re
from typing import List, Dict

class TextTokenizer:
    """
    文本分词类，用于将输入文本分解为tokens

    Attributes:
        vocab (Dict[str, int]): 词汇表，将单词映射到索引
        str_to_int (Dict[str, int]): 单词到索引的映射
        int_to_str (Dict[int, str]): 索引到单词的映射
    """

    def __init__(self, vocab: Dict[str, int] = None):
        """
        初始化TextTokenizer类

        Args:
            vocab (Dict[str, int], optional): 初始化词汇表。默认值为None
        """
        if vocab is None:
            vocab = {"<|unk|>": 0}
        self.vocab = vocab
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def build_vocab(self, sentences: List[str]):
        """
        构建词汇表和索引映射

        Args:
            sentences (List[str]): 句子列表，每个句子是一个字符串
        """
        word_set = set()
        for sentence in sentences:
            words = re.split(r'\s+', sentence)
            word_set.update(words)

        self.str_to_int = {word: idx for idx, word in enumerate(word_set, start=len(self.vocab))}
        self.int_to_str = {idx: word for word, idx in self.str_to_int.items()}
        self.str_to_int.update(self.vocab)  # 保留原始词汇表
        self.int_to_str.update({0: "<|unk|>"})  # 确保<|unk|>在索引中
        print(f"词汇表构建完成，共包含{len(self.str_to_int)}个单词。")

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token IDs

        Args:
            text (str): 输入文本

        Returns:
            List[int]: token ID列表
        """
        tokens = re.split(r'(\s|,|\.)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        token_ids = [self.str_to_int.get(token, self.str_to_int['<|unk|>']) for token in tokens]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将token IDs解码为文本

        Args:
            token_ids (List[int]): token ID列表

        Returns:
            str: 解码后的文本
        """
        tokens = [self.int_to_str[token_id] for token_id in token_ids]
        text = ' '.join(tokens)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

    def most_common_tokens(self, top_n: int = 10) -> List[str]:
        """
        获取出现频率最高的前n个token

        Args:
            top_n (int): 返回的token数量

        Returns:
            List[str]: 最常见的token列表
        """
        token_counts = {token: self.vocab.count(token) for token in self.vocab}
        sorted_tokens = sorted(token_counts, key=token_counts.get, reverse=True)
        return sorted_tokens[:top_n]

# 示例用法
if __name__ == "__main__":
    sentences = [
        "Hello, world.",
        "Word embeddings are useful.",
        "Deep learning is a subset of machine learning.",
        "Machine learning is a field of artificial intelligence."
    ]

    tokenizer = TextTokenizer()
    tokenizer.build_vocab(sentences)

    text = "Machine learning is amazing."
    encoded = tokenizer.encode(text)
    print(f"文本 '{text}' 的编码结果是：{encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"编码结果解码后的文本是：'{decoded}'")

    common_tokens = tokenizer.most_common_tokens(top_n=5)
    print(f"最常见的五个token是：{common_tokens}")