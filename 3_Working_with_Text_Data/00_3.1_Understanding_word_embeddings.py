# 00_3.1_Understanding_word_embeddings

"""
Lecture: /3_Working_with_Text_Data
Content: 00_3.1_Understanding_word_embeddings
"""

import numpy as np
from typing import List, Dict

class WordEmbedding:
    """
    词嵌入类，用于将单词映射到连续向量空间

    Attributes:
        embedding_dim (int): 嵌入向量的维度
        vocab (Dict[str, int]): 词汇表，将单词映射到索引
        index_to_word (Dict[int, str]): 索引到单词的映射
        embedding_matrix (np.ndarray): 嵌入矩阵，每行对应一个单词的向量表示
    """

    def __init__(self, embedding_dim: int = 100):
        """
        初始化WordEmbedding类

        Args:
            embedding_dim (int): 嵌入向量的维度
        """
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.index_to_word = {}
        self.embedding_matrix = None

    def build_vocab(self, sentences: List[str]):
        """
        构建词汇表和索引映射

        Args:
            sentences (List[str]): 句子列表，每个句子是一个字符串
        """
        word_set = set()
        for sentence in sentences:
            words = sentence.split()
            word_set.update(words)

        self.vocab = {word: idx for idx, word in enumerate(word_set)}
        self.index_to_word = {idx: word for word, idx in self.vocab.items()}
        self.embedding_matrix = np.random.randn(len(self.vocab), self.embedding_dim)
        print(f"词汇表构建完成，共包含{len(self.vocab)}个单词。")

    def get_embedding(self, word: str) -> np.ndarray:
        """
        获取单词的嵌入向量

        Args:
            word (str): 单词

        Returns:
            np.ndarray: 单词的嵌入向量
        """
        idx = self.vocab.get(word)
        if idx is None:
            raise ValueError(f"单词 '{word}' 不在词汇表中。")
        return self.embedding_matrix[idx]

    def most_similar(self, word: str, top_n: int = 5) -> List[str]:
        """
        找到与给定单词最相似的top_n个单词

        Args:
            word (str): 单词
            top_n (int): 返回最相似单词的数量

        Returns:
            List[str]: 最相似的单词列表
        """
        if word not in self.vocab:
            raise ValueError(f"单词 '{word}' 不在词汇表中。")

        word_vec = self.get_embedding(word)
        similarities = self.embedding_matrix @ word_vec
        closest_idxs = np.argsort(similarities)[::-1][:top_n + 1]  # 排除自己
        similar_words = [self.index_to_word[idx] for idx in closest_idxs if self.index_to_word[idx] != word]
        return similar_words[:top_n]

    def save_embeddings(self, file_path: str):
        """
        保存嵌入矩阵到文件

        Args:
            file_path (str): 文件路径
        """
        np.save(file_path, self.embedding_matrix)
        print(f"嵌入矩阵已保存到 {file_path}")

    def load_embeddings(self, file_path: str):
        """
        从文件加载嵌入矩阵

        Args:
            file_path (str): 文件路径
        """
        self.embedding_matrix = np.load(file_path)
        print(f"嵌入矩阵已从 {file_path} 加载")

# 示例用法
if __name__ == "__main__":
    sentences = [
        "hello world",
        "word embeddings are useful",
        "deep learning is a subset of machine learning",
        "machine learning is a field of artificial intelligence"
    ]

    embedding = WordEmbedding(embedding_dim=50)
    embedding.build_vocab(sentences)

    word = "machine"
    print(f"单词 '{word}' 的嵌入向量是：\n{embedding.get_embedding(word)}")

    similar_words = embedding.most_similar(word, top_n=3)
    print(f"与单词 '{word}' 最相似的三个单词是：{similar_words}")

    # 保存和加载嵌入
    embedding.save_embeddings("/mnt/data/word_embeddings.npy")
    embedding.load_embeddings("/mnt/data/word_embeddings.npy")
