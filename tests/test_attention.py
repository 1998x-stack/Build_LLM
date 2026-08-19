import torch
from common.attention import MultiHeadAttention

def test_mha_output_shape():
    m = MultiHeadAttention(64, 4)
    out = m(torch.rand(2, 6, 64), torch.rand(2, 6, 64), torch.rand(2, 6, 64))
    assert tuple(out.shape) == (2, 6, 64)

def test_mha_projects_per_head_dim():
    m = MultiHeadAttention(64, 4)
    assert m.head_dim == 16
    assert m.values.in_features == 16 and m.queries.out_features == 16