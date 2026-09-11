"""Regression test: DAPOConfig.dynamic_sampling must actually change what training sees."""

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


def _config(**kwargs):
    defaults = dict(group_size=2, n_epochs=1, min_batch_size=2, use_overlong_punishment=False)
    return DAPOConfig(**{**defaults, **kwargs})


def _batch(rewards):
    n = len(rewards)
    return {
        "input_ids": torch.randint(0, 16, (n, 3)),
        "attention_mask": torch.ones(n, 3, dtype=torch.long),
        "labels": torch.randint(0, 16, (n, 3)),
        "rewards": torch.tensor(rewards),
    }


def test_disabled_by_default_leaves_the_batch_alone():
    algorithm = DAPOAlgorithm(policy_model=_TinyLM(), config=_config())
    assert algorithm.config.dynamic_sampling is False

    metrics = algorithm.train_on_rollout(_batch([1.0, 0.0]))
    assert metrics


def test_enabled_without_a_sampler_raises_instead_of_ignoring_the_flag():
    algorithm = DAPOAlgorithm(policy_model=_TinyLM(), config=_config(dynamic_sampling=True))

    with pytest.raises(ValueError, match="sample_fn"):
        algorithm.train_on_rollout(_batch([1.0, 0.0]))


def test_enabled_resamples_until_a_group_has_reward_variance():
    algorithm = DAPOAlgorithm(policy_model=_TinyLM(), config=_config(dynamic_sampling=True))

    # First two draws are degenerate (both members of the group score the same), so they
    # carry no gradient signal and must be discarded rather than trained on.
    draws = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    calls = []

    def sample_fn():
        rewards = draws[len(calls)] if len(calls) < len(draws) else draws[-1]
        calls.append(rewards)
        batch = _batch(rewards)
        return {k: v for k, v in batch.items() if k != "rewards"}, batch["rewards"]

    metrics = algorithm.train_on_rollout(_batch([0.0, 0.0]), sample_fn=sample_fn)

    assert len(calls) == 3, "degenerate draws were not discarded"
    assert metrics


def test_exhausting_the_attempt_budget_raises():
    algorithm = DAPOAlgorithm(
        policy_model=_TinyLM(), config=_config(dynamic_sampling=True, max_sampling_attempts=2)
    )

    def always_degenerate():
        batch = _batch([1.0, 1.0])
        return {k: v for k, v in batch.items() if k != "rewards"}, batch["rewards"]

    with pytest.raises(RuntimeError, match="non-zero reward variance"):
        algorithm.train_on_rollout(_batch([0.0, 0.0]), sample_fn=always_degenerate)
