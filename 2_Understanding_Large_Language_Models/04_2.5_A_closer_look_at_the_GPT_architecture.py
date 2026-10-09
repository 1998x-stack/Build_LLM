# 04_2.5_A_closer_look_at_the_GPT_architecture

"""Lecture 2.5: inspect a GPT-style architecture without allocating a huge model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_llm.config import GPTConfig
from build_llm.model import GPTModel


def estimate_parameters(config: GPTConfig) -> int:
    """Estimate parameters for the exact modules used by GPTModel."""
    c = config.d_model
    r = config.mlp_ratio
    embeddings = config.vocab_size * c + config.context_length * c
    qkv = 3 * c * c + (3 * c if config.qkv_bias else 0)
    attn_out = c * c + c
    norms = 4 * c
    mlp = c * (r * c) + (r * c) + (r * c) * c + c
    block = qkv + attn_out + norms + mlp
    final_norm = 2 * c
    lm_head = 0 if config.tie_embeddings else c * config.vocab_size
    return embeddings + config.n_layers * block + final_norm + lm_head


if __name__ == "__main__":
    reference = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        n_heads=12,
        n_layers=12,
        dropout=0.1,
        mlp_ratio=4,
        qkv_bias=True,
        tie_embeddings=True,
    )
    print("===== GPT-style reference architecture =====")
    print("vocab_size:", reference.vocab_size)
    print("context_length:", reference.context_length)
    print("d_model:", reference.d_model)
    print("n_heads:", reference.n_heads)
    print("n_layers:", reference.n_layers)
    print("estimated parameters:", f"{estimate_parameters(reference):,}")
    print("block: LN -> causal MHA -> residual -> LN -> GELU MLP -> residual")

    tiny = GPTModel(
        GPTConfig(
            vocab_size=128,
            context_length=16,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
    )
    print("tiny runnable model parameters:", f"{tiny.num_parameters():,}")
