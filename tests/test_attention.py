import torch
import torch.nn.functional as F

from build_llm.nn.attention import (
    CausalSelfAttention,
    MultiHeadAttention,
)


def test_mha_uses_full_width_qkv_projections():
    module = MultiHeadAttention(64, 4)
    assert module.head_dim == 16
    assert module.q_proj.in_features == 64
    assert module.q_proj.out_features == 64
    assert module.k_proj.in_features == 64
    assert module.v_proj.in_features == 64


def test_mha_output_shape_and_mask():
    module = MultiHeadAttention(32, 4)
    values = torch.rand(2, 7, 32)
    keys = torch.rand(2, 7, 32)
    query = torch.rand(2, 5, 32)
    mask = torch.ones(2, 1, 5, 7)
    out = module(values, keys, query, mask)
    assert out.shape == (2, 5, 32)


def test_causal_attention_manual_matches_sdpa():
    if not hasattr(F, "scaled_dot_product_attention"):
        return
    manual = CausalSelfAttention(
        32, 4, 8, dropout=0.0, backend="manual"
    )
    sdpa = CausalSelfAttention(
        32, 4, 8, dropout=0.0, backend="sdpa"
    )
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    x = torch.randn(2, 6, 32)
    assert torch.allclose(
        manual(x), sdpa(x), atol=1e-5, rtol=1e-4
    )


def test_causal_attention_rejects_overlong_sequence():
    module = CausalSelfAttention(32, 4, 4)
    x = torch.randn(1, 5, 32)
    try:
        module(x)
    except ValueError as exc:
        assert "context_length" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
