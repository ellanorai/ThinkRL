"""Regression test: PRIMEConfig rejects values it cannot honour instead of silently degrading."""

import pytest
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


def test_defaults_still_construct():
    assert PRIMEConfig().advantage_estimator == "rloo"


def test_unimplemented_advantage_estimator_is_rejected():
    with pytest.raises(NotImplementedError, match="rloo"):
        PRIMEConfig(advantage_estimator="reinforce")


def test_unknown_advantage_estimator_is_rejected():
    with pytest.raises(ValueError, match="advantage_estimator"):
        PRIMEConfig(advantage_estimator="gae")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_generations_per_prompt": 1},
        {"n_epochs": 0},
        {"batch_size": 0},
        {"clip_epsilon": 0.0},
        {"clip_epsilon": 1.0},
        {"gamma": 0.0},
        {"gamma": 1.5},
        {"temperature": 0.0},
    ],
)
def test_out_of_range_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PRIMEConfig(**kwargs)


def test_group_size_of_one_raises_instead_of_producing_nan():
    algorithm = PRIMEAlgorithm(policy_model=_TinyLM(), ref_model=_TinyLM(), config=PRIMEConfig())

    with pytest.raises(ValueError, match="group_size"):
        algorithm.compute_advantages(torch.ones(1, 3), torch.ones(1), torch.ones(1, 3), group_size=1)


def test_indivisible_batch_raises():
    algorithm = PRIMEAlgorithm(policy_model=_TinyLM(), ref_model=_TinyLM(), config=PRIMEConfig())

    with pytest.raises(ValueError, match="divisible"):
        algorithm.compute_advantages(torch.ones(3, 3), torch.ones(3), torch.ones(3, 3), group_size=2)
