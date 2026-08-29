"""Regression test: PPO value targets must be real discounted returns, not whitened ones.

These drive `train_on_rollout` itself and inspect the batch it hands to `training_step`,
so they fail if the returns computation regresses regardless of how it is written.
"""

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


def _capture_training_batches(algorithm, batch):
    """Run one rollout and return every batch train_on_rollout passed to training_step."""
    seen = []
    original = algorithm.training_step

    def spy(training_batch):
        seen.append({k: v.detach().clone() for k, v in training_batch.items()})
        return original(training_batch)

    algorithm.training_step = spy
    algorithm.train_on_rollout(batch)
    return seen


def _rollout_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "rewards": torch.tensor([10.0, -4.0]),
    }


def test_non_terminal_targets_stay_on_the_reward_scale():
    """A whitened target collapses every position toward the value estimate; a real one does not."""
    torch.manual_seed(0)
    algorithm = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(n_epochs=1, batch_size=2))

    seen = _capture_training_batches(algorithm, _rollout_batch())
    returns = torch.cat([s["returns"] for s in seen])

    positive = returns[returns[:, -1] > 0]
    assert positive.numel(), "expected one sequence with a positive terminal reward"
    # gamma=0.99, lambda=0.95 over three steps keeps the earlier targets close to 10.0,
    # where the whitened form put them near 1.8.
    assert positive.min().item() > 8.0


def test_terminal_target_equals_the_terminal_reward():
    """With gamma applied to a zero bootstrap, the last token's target is the reward itself."""
    torch.manual_seed(0)
    algorithm = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(n_epochs=1, batch_size=2))

    seen = _capture_training_batches(algorithm, _rollout_batch())
    terminal = torch.cat([s["returns"] for s in seen])[:, -1]

    assert sorted(round(v, 4) for v in terminal.tolist()) == [-4.0, 10.0]


def test_normalize_advantages_does_not_touch_the_critic_target():
    torch.manual_seed(0)
    on = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(n_epochs=1, batch_size=2, normalize_advantages=True))
    torch.manual_seed(0)
    off = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(n_epochs=1, batch_size=2, normalize_advantages=False))

    on_terminal = sorted(torch.cat([s["returns"] for s in _capture_training_batches(on, _rollout_batch())])[:, -1].tolist())
    off_terminal = sorted(torch.cat([s["returns"] for s in _capture_training_batches(off, _rollout_batch())])[:, -1].tolist())

    assert on_terminal == off_terminal == [-4.0, 10.0]
