import importlib.util
import math
import os

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORPUS = "the cat sat on the mat. the dog sat too. the cat ran fast."


def _fixture():
    vocab = {
        token: index
        for index, token in enumerate(
            sorted(set(CORPUS.lower().split()))
        )
    }
    ids = [vocab[word] for word in CORPUS.split()]
    context = 4
    module = load(
        "6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py",
        "m62",
    )
    dataset = module.GPTDataset(ids, context)
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False
    )
    return vocab, ids, context, loader


def test_62_batch_shapes_and_shift():
    _, _, context, loader = _fixture()
    x, y = next(iter(loader))
    assert list(x.shape) == [2, context]
    assert list(y.shape) == [2, context]
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_63_train_reduces_loss():
    module = load(
        "6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py",
        "m63",
    )
    vocab, _, context, loader = _fixture()
    model = module.GPT(
        vocab_size=len(vocab),
        embed_size=16,
        num_layers=1,
        heads=2,
        forward_expansion=2,
        dropout=0.0,
        max_length=context,
    )
    history = module.train(
        model, loader, n_epochs=4, lr=0.03
    )
    assert history[-1] < history[0]


def test_64_perplexity_finite_and_generate():
    module = load(
        "6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py",
        "m64",
    )
    vocab, _, context, loader = _fixture()
    model = module.GPT(
        vocab_size=len(vocab),
        embed_size=16,
        num_layers=1,
        heads=2,
        forward_expansion=2,
        dropout=0.0,
        max_length=context,
    )
    ids_to_token = {
        index: token for token, index in vocab.items()
    }
    value = module.perplexity(model, loader)
    assert value > 0 and math.isfinite(value)
    sample = module.generate(
        model, [0], ids_to_token, n_new=3
    )
    assert len(sample) == 4
    assert all(token in ids_to_token.values() for token in sample)
