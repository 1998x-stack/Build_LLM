import torch
import torch.nn as nn

from build_llm.config import GPTConfig
from build_llm.model import GPTModel
from common.attention import GPT


def _config():
    return GPTConfig(
        vocab_size=16,
        context_length=20,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )


def test_gpt_forward_shape_and_final_norm():
    model = GPTModel(_config())
    out = model(torch.randint(0, 16, (2, 6)))
    assert out.shape == (2, 6, 16)
    assert isinstance(model.final_norm, nn.LayerNorm)


def test_gpt_uses_pre_norm_gelu_blocks():
    model = GPTModel(_config())
    block = model.blocks[0]
    assert isinstance(block.norm1, nn.LayerNorm)
    assert any(
        isinstance(module, nn.GELU)
        for module in block.feed_forward.modules()
    )


def test_gpt_causal_pos0_unaffected_by_future():
    torch.manual_seed(0)
    model = GPTModel(_config())
    model.eval()
    x1 = torch.tensor([[3, 4, 5, 6, 7, 8]])
    x2 = torch.tensor([[3, 9, 9, 9, 9, 9]])
    assert torch.allclose(
        model(x1)[0, 0],
        model(x2)[0, 0],
        atol=1e-6,
    )


def test_backward_gradients_are_finite():
    model = GPTModel(_config())
    x = torch.randint(0, 16, (2, 8))
    y = torch.randint(0, 16, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 16), y.reshape(-1)
    )
    loss.backward()
    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_legacy_gpt_constructor_is_device_agnostic():
    model = GPT(
        16, 32, 2, 4, "cpu", 4, 0.0, 20
    )
    assert not hasattr(model, "device")
    model = model.to("cpu")
    assert model(torch.randint(0, 16, (1, 4))).shape == (
        1,
        4,
        16,
    )
