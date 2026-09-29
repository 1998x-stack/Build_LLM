import pytest
import torch

from build_llm.config import GPTConfig
from build_llm.generation import GenerationConfig, generate
from build_llm.model import GPTModel


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_forward_and_generation_follow_model_device(device):
    model = GPTModel(
        GPTConfig(
            vocab_size=12,
            context_length=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
        )
    ).to(device)
    tokens = torch.randint(0, 12, (1, 4), device=device)
    logits = model(tokens)
    assert logits.device.type == torch.device(device).type
    generated = generate(
        model,
        torch.tensor([[1, 2]]),
        GenerationConfig(max_new_tokens=2),
    )
    assert generated.device.type == torch.device(device).type
