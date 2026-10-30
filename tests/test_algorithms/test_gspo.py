"""GSPO (#195): Qwen3's RL algorithm, a sequence-level variant of GRPO.

GRPO clips a per-token ratio and averages the clipped surrogate over tokens.
GSPO instead computes one length-normalized ratio per sequence, clips that,
and averages over sequences -- every response in the group counts equally
regardless of length. These tests pin that difference with an independently
computed expectation (not just "it runs"), since getting a published
algorithm's math wrong is a worse failure than a crash.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import thinkrl.algorithms as algorithms
from thinkrl.algorithms.gspo import GSPOAlgorithm, GSPOConfig, create_gspo
from thinkrl.models.loss import GSPOLoss


class SimplePolicy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        logits = self.head(embeds)
        return {"logits": logits}


@pytest.fixture
def policy_model():
    return SimplePolicy()


@pytest.fixture
def gspo_config():
    return GSPOConfig(group_size=4, learning_rate=1e-5)


@pytest.fixture
def gspo_algo(policy_model, gspo_config):
    return GSPOAlgorithm(policy_model, config=gspo_config)


def test_gspo_config_defaults():
    config = GSPOConfig()
    assert config.beta == 0.0
    assert config.epsilon_low == 3e-4
    assert config.epsilon_high == 4e-4
    assert config.group_size == 64  # inherited from GRPOConfig


def test_gspo_registered_as_a_real_non_stub_algorithm():
    assert algorithms.ALGORITHMS["gspo"] is GSPOAlgorithm
    assert algorithms.CONFIGS["gspo"] is GSPOConfig
    assert not algorithms.is_stub(GSPOAlgorithm)


def test_create_gspo_accepts_a_prebuilt_config():
    config = GSPOConfig(learning_rate=2e-6)
    algorithm = create_gspo(policy_model=SimplePolicy(), config=config)

    assert isinstance(algorithm, GSPOAlgorithm)
    assert algorithm.config is config


# --- GSPOLoss: the sequence-level ratio itself ---


def _expected_surrogate(log_probs, old_log_probs, advantages, mask, eps_low, eps_high):
    """Independently reproduces the paper's formula, so the test isn't just
    checking GSPOLoss against a transcription of its own source."""
    seq_len = mask.sum(dim=1)
    log_ratio = ((log_probs - old_log_probs) * mask).sum(dim=1) / seq_len
    s = torch.exp(log_ratio)
    surr1 = s * advantages
    surr2 = torch.clamp(s, 1.0 - eps_low, 1.0 + eps_high) * advantages
    return s, torch.min(surr1, surr2)


def test_gspo_loss_matches_the_sequence_level_formula():
    loss_fn = GSPOLoss(epsilon_low=3e-4, epsilon_high=4e-4)

    log_probs = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.1]])
    old_log_probs = torch.zeros(2, 3)
    advantages = torch.tensor([2.0, -3.0])
    mask = torch.ones(2, 3)

    loss, metrics = loss_fn(log_probs, old_log_probs, advantages, mask)

    expected_s, expected_surrogate = _expected_surrogate(log_probs, old_log_probs, advantages, mask, 3e-4, 4e-4)
    expected_loss = -expected_surrogate.mean()

    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert torch.allclose(metrics["sequence_ratio_mean"], expected_s.mean(), atol=1e-6)
    # Both sequences' ratios fall outside [1-3e-4, 1+4e-4] in this example.
    assert metrics["clip_frac"].item() == 1.0


def test_gspo_loss_ignores_padding_tokens():
    """A padded tail must not dilute the per-sequence length normalization."""
    loss_fn = GSPOLoss()

    log_probs = torch.tensor([[0.3, 0.3, 999.0]])  # position 2 is padding, must be ignored
    old_log_probs = torch.tensor([[0.0, 0.0, 0.0]])
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    _, metrics = loss_fn(log_probs, old_log_probs, advantages, mask)

    expected_s = torch.exp(torch.tensor((0.3 + 0.3) / 2))
    assert torch.allclose(metrics["sequence_ratio_mean"], expected_s, atol=1e-6)


def test_gspo_loss_is_unaffected_by_sequence_length_given_equal_per_token_ratio():
    """The length normalization is the point: a longer response with the same
    per-token log-ratio must produce the same sequence ratio as a shorter one."""
    loss_fn = GSPOLoss()

    short = torch.full((1, 3), 0.05)
    long = torch.full((1, 10), 0.05)
    old_short, old_long = torch.zeros(1, 3), torch.zeros(1, 10)
    mask_short, mask_long = torch.ones(1, 3), torch.ones(1, 10)
    advantages = torch.tensor([1.0])

    _, metrics_short = loss_fn(short, old_short, advantages, mask_short)
    _, metrics_long = loss_fn(long, old_long, advantages, mask_long)

    assert torch.allclose(metrics_short["sequence_ratio_mean"], metrics_long["sequence_ratio_mean"], atol=1e-6)


# --- GSPOAlgorithm: reuses GRPO's advantage + rollout loop, overrides compute_loss ---


def test_compute_advantages_is_reused_unchanged_from_grpo(gspo_algo):
    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
    adv = gspo_algo.compute_advantages(rewards)

    assert adv.shape == (4,)
    assert torch.isclose(adv.sum(), torch.tensor(0.0), atol=1e-5)


def test_compute_loss_structure_and_backward(gspo_algo):
    batch_size, seq_len, vocab_size = 4, 5, 10
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "rewards": torch.randn(batch_size),
        "old_log_probs": torch.randn(batch_size, seq_len),
    }

    loss_dict = gspo_algo.compute_loss(batch)

    assert "loss" in loss_dict
    assert "advantage_mean" in loss_dict
    assert "clip_fraction" in loss_dict
    assert "sequence_ratio_mean" in loss_dict
    assert "kl_mean" not in loss_dict  # GSPO applies no explicit KL term

    loss = loss_dict["loss"]
    assert loss.requires_grad

    gspo_algo.optimizer.zero_grad()
    loss.backward()
    for param in gspo_algo.policy_model.parameters():
        assert param.grad is not None


def test_train_on_rollout_loop_is_reused_from_grpo(gspo_algo):
    """GSPOAlgorithm overrides only compute_loss; the multi-epoch loop (#194's
    framing for PAPO) is GRPOAlgorithm.train_on_rollout, unchanged."""
    gspo_algo.config.n_epochs = 2

    batch = {
        "input_ids": torch.randint(0, 10, (4, 5), dtype=torch.long),
        "attention_mask": torch.ones((4, 5)),
        "labels": torch.randint(0, 10, (4, 5), dtype=torch.long),
        "rewards": torch.randn(4),
    }

    with patch.object(gspo_algo, "compute_rollout_log_probs") as mock_old_log:
        mock_old_log.return_value = torch.zeros(4, 5)
        with patch.object(gspo_algo, "training_step") as mock_step:
            mock_step.side_effect = lambda batch, old_log_probs: {"loss": 0.5}

            metrics = gspo_algo.train_on_rollout(batch)

            assert mock_step.call_count == 2
            assert len(metrics) == 2


def test_beta_above_zero_logs_a_warning_since_no_kl_term_is_applied():
    config = GSPOConfig(beta=0.1)
    with patch("thinkrl.algorithms.gspo.logger") as mock_logger:
        GSPOAlgorithm(SimplePolicy(), config=config)
        mock_logger.warning.assert_called_once()
