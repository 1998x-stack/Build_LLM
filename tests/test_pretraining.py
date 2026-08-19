import os, math, importlib.util, torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

CORPUS = "the cat sat on the mat. the dog sat too. the cat ran fast."

def _fixture():
    vocab = {t: i for i, t in enumerate(sorted(set(CORPUS.lower().split())))}
    ids = [vocab[w] for w in CORPUS.split()]
    ctx = 4
    m62 = load("6_Pretraining_on_Unlabeled_Data/01_6.2_Data_preparation.py", "m62")
    ds = m62.GPTDataset(ids, ctx)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    return vocab, ids, ctx, loader

def test_62_batch_shapes_and_shift():
    vocab, ids, ctx, loader = _fixture()
    x, y = next(iter(loader))
    assert list(x.shape) == [2, ctx] and list(y.shape) == [2, ctx]
    for i in range(x.size(0)):
        assert torch.equal(y[i, :-1], x[i, 1:])   # y = x shifted left by 1 (last target = next id)

def test_63_train_reduces_loss():
    m63 = load("6_Pretraining_on_Unlabeled_Data/02_6.3_Pretraining_process.py", "m63")
    vocab, ids, ctx, loader = _fixture()
    model = m63.GPT(vocab_size=len(vocab), embed_size=16, num_layers=1, heads=2,
                    device="cpu", forward_expansion=2, dropout=0.0, max_length=ctx)
    history = m63.train(model, loader, n_epochs=3, lr=0.05)
    assert history[-1] < history[0]

def test_64_perplexity_finite_and_generate():
    m64 = load("6_Pretraining_on_Unlabeled_Data/03_6.4_Evaluating_pretrained_model.py", "m64")
    vocab, ids, ctx, loader = _fixture()
    model = m64.GPT(vocab_size=len(vocab), embed_size=16, num_layers=1, heads=2,
                    device="cpu", forward_expansion=2, dropout=0.0, max_length=ctx)
    ids2tok = {i: t for t, i in vocab.items()}
    ppl = m64.perplexity(model, loader)
    assert ppl > 0 and (not math.isinf(ppl)) and not math.isnan(ppl)
    sample = m64.generate(model, [0], ids2tok, n_new=3)
    assert len(sample) > 0 and all(t in ids2tok.values() for t in sample)
