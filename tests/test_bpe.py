import os, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

bpe = load("3_Working_with_Text_Data/04_3.5_Byte_pair_encoding.py", "bpe")

def test_bpe_fit_encode_decode():
    m = bpe.BytePairEncoding(vocab_size=50)
    m.fit(["low", "lowest", "newer", "wider"])
    assert m.decode(m.encode("lower")) == "lower"

def test_bpe_roundtrip_clean():
    m = bpe.BytePairEncoding(vocab_size=50)
    m.fit(["low", "lowest", "newer", "wider"])
    assert m.decode(m.encode("lowest")) == "lowest"