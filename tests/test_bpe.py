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


bpe = load(
    "3_Working_with_Text_Data/04_3.5_Byte_pair_encoding.py",
    "bpe",
)


def test_bpe_fit_encode_decode():
    model = bpe.BytePairEncoding(vocab_size=50)
    model.fit(["low", "lowest", "newer", "wider"])
    assert model.decode(model.encode("lower")) == "lower"


def test_bpe_roundtrip_clean():
    model = bpe.BytePairEncoding(vocab_size=50)
    model.fit(["low", "lowest", "newer", "wider"])
    assert model.decode(model.encode("lowest")) == "lowest"
