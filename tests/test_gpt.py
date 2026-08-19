import os, torch, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(rel, name):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
from common.attention import GPT, MultiHeadAttention, GPTBlock

def test_gpt_forward_shape():
    model = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    out = model(torch.randint(0, 16, (2, 6)))
    assert tuple(out.shape) == (2, 6, 16)

def test_gpt_causal_pos0_unaffected_by_future():
    m = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    x1 = torch.tensor([[3, 4, 5, 6, 7, 8]])
    x2 = torch.tensor([[3, 9, 9, 9, 9, 9]])   # same pos-0, different future
    assert torch.allclose(m(x1)[0, 0], m(x2)[0, 0], atol=1e-5)

def test_gpt_copy_consistent_with_52():
    m52 = load("5_Implementing_a_GPT_model_from_Scratch_To_Generate_Text/01_5.2_Implementing_GPT_model.py", "g52")
    g = GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    g52 = m52.GPT(16, 32, 2, 4, "cpu", 4, 0.0, 20)
    g.load_state_dict(g52.state_dict())
    x = torch.randint(0, 16, (2, 6))
    mask = torch.tril(torch.ones(6, 6))          # 5.2 requires the mask passed explicitly
    assert torch.equal(g(x), g52(x, mask))       # identical & causal
