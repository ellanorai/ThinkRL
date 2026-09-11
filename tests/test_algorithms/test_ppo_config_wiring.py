"""Regression test: PPOConfig hyperparameters must reach the GAE computation."""

import torch
import torch.nn as nn

from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.vhead = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.emb(input_ids)
        return {"logits": self.head(hidden), "values": self.vhead(hidden).squeeze(-1)}


def test_gamma_and_gae_lambda_reach_the_algorithm():
    config = PPOConfig(gamma=0.5, gae_lambda=0.1)
    algorithm = PPOAlgorithm(policy_model=_TinyLM(), config=config)

    assert algorithm.gamma == config.gamma
    assert algorithm.lambda_ == config.gae_lambda


def test_normalize_advantages_reaches_the_algorithm():
    assert PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(normalize_advantages=False)).normalize_advantages is False
    assert PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(normalize_advantages=True)).normalize_advantages is True


def test_configured_discount_changes_the_advantages():
    """The wiring is only meaningful if the values actually alter the GAE result."""
    rewards = torch.tensor([[0.0, 0.0, 10.0]])
    values = torch.tensor([[1.0, 1.0, 1.0]])

    default = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig())
    steep = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(gamma=0.5, gae_lambda=0.1))

    assert not torch.allclose(
        default.compute_gae_advantages(rewards, values, normalize=False),
        steep.compute_gae_advantages(rewards, values, normalize=False),
    )
