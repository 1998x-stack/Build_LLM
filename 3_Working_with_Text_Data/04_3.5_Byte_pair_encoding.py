# 04_3.5_Byte_pair_encoding

"""
Lecture: /3_Working_with_Text_Data
Content: 04_3.5_Byte_pair_encoding
"""

import re
from typing import List, Tuple, Dict

class BytePairEncoding:
    def __init__(self, vocab_size: int):
        """
        初始化BPE编码器。

        参数:
        vocab_size (int): 词汇表的目标大小。
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_codes = {}
    
    def _get_stats(self, corpus: List[str]) -> Dict[Tuple[str, str], int]:
        """
        统计字符对的频率。

        参数:
        corpus (List[str]): 文本语料库，已拆分为字符对的形式。

        返回:
        Dict[Tuple[str, str], int]: 字符对及其频率的字典。
        """
        pairs = {}
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], corpus: List[str]) -> List[str]:
        """
        合并字符对。

        参数:
        pair (Tuple[str, str]): 要合并的字符对。
        corpus (List[str]): 文本语料库，已拆分为字符对的形式。

        返回:
        List[str]: 更新后的文本语料库。
        """
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        return [pattern.sub(''.join(pair), word) for word in corpus]
    
    def fit(self, corpus: List[str]) -> None:
        """
        训练BPE模型。

        参数:
        corpus (List[str]): 文本语料库。
        """
        corpus = [' '.join(word) + ' </w>' for word in corpus]
        self.vocab = {word: corpus.count(word) for word in corpus}
        
        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            corpus = self._merge_vocab(best_pair, corpus)
            self.bpe_codes[best_pair] = len(self.bpe_codes)
            self.vocab = {word: corpus.count(word) for word in set(corpus)}

    def encode(self, text: str) -> List[str]:
        """
        使用BPE编码文本。

        参数:
        text (str): 要编码的文本。

        返回:
        List[str]: 编码后的子词列表。
        """
        word = ' '.join(text) + ' </w>'
        symbols = word.split()
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            pairs = [(pair, self.bpe_codes[pair]) for pair in pairs if pair in self.bpe_codes]
            if not pairs:
                break
            best_pair = min(pairs, key=lambda x: x[1])[0]
            symbols = self._merge_vocab(best_pair, [' '.join(symbols)])[0].split()
        return symbols
    
    def decode(self, tokens: List[str]) -> str:
        """
        解码BPE子词列表为文本。

        参数:
        tokens (List[str]): BPE子词列表。

        返回:
        str: 解码后的文本。
        """
        return ''.join(tokens).replace(' </w>', '')

# 示例使用
if __name__ == "__main__":
    # 示例文本语料库
    corpus = ["low", "lowest", "newer", "wider"]
    
    # 初始化BPE编码器
    bpe = BytePairEncoding(vocab_size=50)
    
    # 训练BPE模型
    bpe.fit(corpus)
    
    # 编码示例
    encoded = bpe.encode("lower")
    print("Encoded:", encoded)
    
    # 解码示例
    decoded = bpe.decode(encoded)
    print("Decoded:", decoded)