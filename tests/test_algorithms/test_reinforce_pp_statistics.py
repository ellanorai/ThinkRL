"""Regression test: advantage normalization must not depend on world size."""

import torch
import torch.nn as nn

from thinkrl.algorithms.reinforce_pp import REINFORCEPPAlgorithm, REINFORCEPPConfig


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


def _algorithm():
    return REINFORCEPPAlgorithm(policy_model=_TinyLM(), ref_model=_TinyLM(), config=REINFORCEPPConfig())


def _population_std(tensor):
    """What the distributed branch computes: sqrt(E[x^2] - E[x]^2)."""
    return torch.sqrt(torch.clamp((tensor**2).mean() - tensor.mean() ** 2, min=1e-8))


def test_single_process_std_matches_the_distributed_formula():
    algorithm = _algorithm()
    values = torch.tensor([1.0, 2.0, 3.0, 10.0])

    _, std = algorithm._global_statistics(values)

    torch.testing.assert_close(std, _population_std(values))


def test_small_batches_do_not_drift_from_the_distributed_formula():
    """The n vs n-1 gap is widest exactly where per-device batches usually sit."""
    algorithm = _algorithm()

    for n in (4, 8, 16):
        values = torch.arange(float(n))
        _, std = algorithm._global_statistics(values)
        torch.testing.assert_close(std, _population_std(values), rtol=1e-6, atol=1e-6)


def test_masking_still_selects_valid_elements():
    algorithm = _algorithm()
    values = torch.tensor([1.0, 2.0, 3.0, 999.0])
    mask = torch.tensor([1, 1, 1, 0])

    mean, std = algorithm._global_statistics(values, mask)

    torch.testing.assert_close(mean, torch.tensor(2.0))
    torch.testing.assert_close(std, _population_std(values[:3]))


def test_degenerate_inputs_are_still_handled():
    algorithm = _algorithm()

    assert algorithm._global_statistics(torch.tensor([]))[1].item() == 1.0
    assert algorithm._global_statistics(torch.tensor([5.0]))[1].item() == 1.0
