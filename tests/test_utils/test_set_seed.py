"""#83: set_seed makes torch draws reproducible."""
import torch
from thinkrl.utils import set_seed


def test_set_seed_reproducible():
    set_seed(1234)
    a = torch.rand(5)
    set_seed(1234)
    b = torch.rand(5)
    assert torch.equal(a, b)
