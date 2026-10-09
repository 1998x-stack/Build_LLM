import math
import tempfile

import torch
from torch.utils.data import DataLoader

from build_llm.config import GPTConfig
from build_llm.data import NextTokenDataset
from build_llm.evaluation import perplexity
from build_llm.generation import GenerationConfig, generate
from build_llm.model import GPTModel
from build_llm.training import (
    TrainConfig,
    Trainer,
    load_checkpoint,
    save_checkpoint,
)


def _tiny_model():
    return GPTModel(
        GPTConfig(
            vocab_size=5,
            context_length=4,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
        )
    )


def _loader():
    ids = [
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
    ]
    return DataLoader(
        NextTokenDataset(ids, context_length=4),
        batch_size=4,
        shuffle=False,
    )


def test_dataset_shift():
    x, y = NextTokenDataset(
        [0, 1, 2, 3, 4, 5], context_length=4
    )[0]
    assert torch.equal(x[1:], y[:-1])


def test_tiny_training_reduces_loss():
    model = _tiny_model()
    trainer = Trainer(
        model,
        TrainConfig(
            epochs=5,
            learning_rate=0.03,
            weight_decay=0.0,
            device="cpu",
            seed=0,
        ),
    )
    history = trainer.fit(_loader())
    assert (
        history[-1]["train_loss"]
        < history[0]["train_loss"]
    )
    ppl = perplexity(model, _loader())
    assert ppl > 0 and math.isfinite(ppl)


def test_greedy_generation_is_deterministic_and_crops_context():
    model = _tiny_model()
    seed = torch.tensor([[1, 2, 3]])
    cfg = GenerationConfig(
        max_new_tokens=10, temperature=0.0
    )
    out1 = generate(model, seed, cfg)
    out2 = generate(model, seed, cfg)
    assert out1.shape == (1, 13)
    assert torch.equal(out1, out2)
    assert out1.device == next(model.parameters()).device


def test_checkpoint_roundtrip():
    model = _tiny_model()
    original = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        save_checkpoint(
            handle.name, model, epoch=3
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        metadata = load_checkpoint(
            handle.name, model
        )
    assert metadata["epoch"] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
