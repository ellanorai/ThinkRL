"""Regression test: padding must not leak into advantages or returns."""

import torch
import torch.nn as nn

from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig
from thinkrl.utils.metrics import compute_advantages, compute_returns


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.vhead = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.emb(input_ids)
        return {"logits": self.head(hidden), "values": self.vhead(hidden).squeeze(-1)}


# One sequence of true length 2 padded to width 3.
REWARDS = torch.tensor([[0.0, 5.0, 0.0]])
MASK = torch.tensor([[1.0, 1.0, 0.0]])


def test_pad_slot_values_do_not_reach_real_tokens():
    junk = compute_advantages(REWARDS, torch.tensor([[0.0, 0.0, 7.0]]), normalize=False, action_mask=MASK)
    zero = compute_advantages(REWARDS, torch.tensor([[0.0, 0.0, 0.0]]), normalize=False, action_mask=MASK)

    torch.testing.assert_close(junk, zero)


def test_pad_slot_values_leak_without_the_mask():
    """Pins why the argument exists: unmasked, the same inputs disagree."""
    junk = compute_advantages(REWARDS, torch.tensor([[0.0, 0.0, 7.0]]), normalize=False)
    zero = compute_advantages(REWARDS, torch.tensor([[0.0, 0.0, 0.0]]), normalize=False)

    assert not torch.allclose(junk[:, :2], zero[:, :2])


def test_masked_positions_are_zero():
    advantages = compute_advantages(REWARDS, torch.tensor([[1.0, 1.0, 9.0]]), normalize=False, action_mask=MASK)

    assert advantages[0, 2].item() == 0.0


def test_normalization_statistics_exclude_padding():
    values = torch.tensor([[1.0, 1.0, 50.0], [1.0, 1.0, 50.0]])
    rewards = torch.tensor([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    normalized = compute_advantages(rewards, values, normalize=True, action_mask=mask)
    valid = normalized[mask.bool()]

    torch.testing.assert_close(valid.mean(), torch.tensor(0.0), rtol=0, atol=1e-5)
    assert normalized[:, 2].abs().sum().item() == 0.0


def test_returns_ignore_padding():
    masked = compute_returns(torch.tensor([[1.0, 1.0, 99.0]]), gamma=1.0, action_mask=MASK)

    torch.testing.assert_close(masked, torch.tensor([[2.0, 1.0, 0.0]]))


def test_unmasked_behaviour_is_unchanged():
    """Callers that pass no mask must get exactly what they got before."""
    rewards = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, -4.0]])
    values = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    advantages = compute_advantages(rewards, values, normalize=False)

    # At the terminal step the bootstrap value is zero, so the residual is r - V exactly.
    torch.testing.assert_close(advantages[0, 2], torch.tensor(9.0))
    torch.testing.assert_close(advantages[1, 2], torch.tensor(-6.0))


def test_ppo_rollout_passes_the_attention_mask():
    torch.manual_seed(0)
    algorithm = PPOAlgorithm(policy_model=_TinyLM(), config=PPOConfig(n_epochs=1, batch_size=2))

    seen = []
    original = algorithm.compute_gae_advantages

    def spy(*args, **kwargs):
        seen.append(kwargs.get("action_mask"))
        return original(*args, **kwargs)

    algorithm.compute_gae_advantages = spy
    algorithm.train_on_rollout(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, -100]]),
            "rewards": torch.tensor([10.0, -4.0]),
        }
    )

    assert seen and seen[0] is not None, "train_on_rollout computed GAE without a mask"
