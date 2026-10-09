import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t32 = load(
    "3_Working_with_Text_Data/01_3.2_Tokenizing_text.py",
    "t32",
)
t34 = load(
    "3_Working_with_Text_Data/03_3.4_Adding_special_context_tokens.py",
    "t34",
)


def test_32_encode_decode_roundtrip():
    tokenizer = t32.TextTokenizer()
    tokenizer.build_vocab(
        ["apple banana cherry", "apple is sweet"]
    )
    assert (
        tokenizer.decode(tokenizer.encode("apple banana"))
        == "apple banana"
    )


def test_32_most_common_tokens():
    tokenizer = t32.TextTokenizer()
    tokenizer.build_vocab(
        ["apple banana", "apple apple banana"]
    )
    assert tokenizer.most_common_tokens(1) == ["apple"]


def test_34_unknown_maps_to_unk():
    vocab = t34.build_vocab(["hello", "world"])
    tokenizer = t34.TokenizerWithSpecialTokens(vocab)
    assert tokenizer.encode("missingword") == [
        vocab["<|unk|>"]
    ]


def test_34_known_decode_matches_encoded_text():
    text = (
        "hello do you like tea <|endoftext|> "
        "in the sunlit terraces"
    )
    vocab = t34.build_vocab(
        [
            "hello",
            "do",
            "you",
            "like",
            "tea",
            "in",
            "the",
            "sunlit",
            "terraces",
        ]
    )
    tokenizer = t34.TokenizerWithSpecialTokens(vocab)
    assert tokenizer.decode(tokenizer.encode(text)) == text
