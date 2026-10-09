import importlib.util
from pathlib import Path

from common.attention import GPT as CanonicalLegacyGPT
from common.attention import MultiHeadAttention as CanonicalMHA

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chapter_43_reuses_canonical_attention():
    chapter = _load(
        "4_Coding_Attention_Mechanisms/02_4.3_Multi_head_attention_mechanisms.py",
        "chapter43",
    )
    assert chapter.MultiHeadAttention is CanonicalMHA


def test_chapter_52_reuses_canonical_gpt():
    chapter = _load(
        "5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.py",
        "chapter52",
    )
    assert chapter.GPT is CanonicalLegacyGPT
