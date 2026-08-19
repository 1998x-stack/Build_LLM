import os, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

t32 = load("3_Working_with_Text_Data/01_3.2_Tokenizing_text.py", "t32")
t34 = load("3_Working_with_Text_Data/03_3.4_Adding_special_context_tokens.py", "t34")

def test_32_encode_decode_roundtrip():
    tok = t32.TextTokenizer()
    tok.build_vocab(["apple banana cherry", "apple is sweet"])
    assert tok.decode(tok.encode("apple banana")) == "apple banana"

def test_32_most_common_tokens():
    tok = t32.TextTokenizer()
    tok.build_vocab(["apple banana", "apple apple banana"])
    assert tok.most_common_tokens(1) == ["apple"]

def test_34_unknown_maps_to_unk():
    vocab = t34.build_vocab(["hello", "world"])
    tok = t34.TokenizerWithSpecialTokens(vocab)
    assert tok.encode("missingword") == [vocab["<|unk|>"]]

def test_34_known_decode_matches_encoded_text():
    # Lowercase known tokens only: capitalized/punctuated forms aren't in this
    # vocab and would map to <|unk|>, so use a lowercase clean roundtrip to
    # verify token identity across encode/decode.
    text = "hello do you like tea <|endoftext|> in the sunlit terraces"
    vocab = t34.build_vocab(["hello", "do", "you", "like", "tea", "in", "the", "sunlit", "terraces"])
    tok = t34.TokenizerWithSpecialTokens(vocab)
    assert tok.decode(tok.encode(text)) == text