"""Regression test: PRIME must apply Equation 5 for every gamma, not only gamma == 1.0."""

import torch
import torch.nn as nn

from thinkrl.algorithms.prime import PRIMEAlgorithm, PRIMEConfig


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


def _algorithm(gamma):
    return PRIMEAlgorithm(policy_model=_TinyLM(), ref_model=_TinyLM(), config=PRIMEConfig(gamma=gamma))


PROCESS = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
OUTCOME = torch.tensor([1.0, 0.0])
MASK = torch.ones(2, 3)


def _advantages(gamma):
    return _algorithm(gamma).compute_advantages(PROCESS, OUTCOME, MASK, group_size=2)


def test_undiscounted_case_is_the_reverse_cumulative_sum():
    # Per-token RLOO residual is 1.0 on the first row; outcome residual adds a further 1.0.
    torch.testing.assert_close(_advantages(1.0)[0], torch.tensor([4.0, 3.0, 2.0]))


def test_discounted_case_applies_the_discount_rather_than_skipping_the_sum():
    gamma = 0.9
    advantages = _advantages(gamma)[0]

    # A_t = d_t + gamma * A_{t+1} over a residual of 1.0 per token, plus the outcome term.
    process = torch.tensor([1.0 + gamma * (1.0 + gamma * 1.0), 1.0 + gamma * 1.0, 1.0])
    torch.testing.assert_close(advantages, process + 1.0)


def test_discount_is_not_silently_dropped():
    """The regression: gamma != 1.0 used to return the raw per-token residual."""
    raw_residual = torch.tensor([2.0, 2.0, 2.0])
    assert not torch.allclose(_advantages(0.9)[0], raw_residual)
    assert not torch.allclose(_advantages(0.99)[0], raw_residual)


def test_gamma_approaching_one_approaches_the_undiscounted_result():
    torch.testing.assert_close(_advantages(0.999)[0], _advantages(1.0)[0], rtol=0, atol=0.01)
