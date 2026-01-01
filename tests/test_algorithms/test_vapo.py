"""
Test Suite for ThinkRL VAPO Algorithm
=====================================

Tests for the VAPO implementation focusing on:
- Length-Adaptive GAE (dynamic lambda based on response length)
- Decoupled GAE (different lambdas for policy and critic)
- Asymmetric Clipping (Clip-Higher)
- Token-level Loss aggregation
- Positive Example LM Loss

Author: EllanorAI
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Explicitly import VAPO components
from thinkrl.algorithms.vapo import (
    VAPOAlgorithm,
    VAPOConfig,
)


# --- Helper Classes ---


class SimplePolicyUnified(nn.Module):
    """Simple policy model that also outputs values (Unified architecture)."""

    def __init__(self, vocab_size=20, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        logits = self.policy_head(embeds)
        values = self.value_head(embeds).squeeze(-1)
        return {"logits": logits, "values": values}


class SimplePolicySeparate(nn.Module):
    """Simple policy model (Actor only)."""

    def __init__(self, vocab_size=20, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        logits = self.policy_head(embeds)
        return {"logits": logits}


class SimpleValueModel(nn.Module):
    """Simple value model (Critic only)."""

    def __init__(self, vocab_size=20, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        # Output [B, T, 1] -> [B, T]
        return self.value_head(embeds).squeeze(-1)


# --- Fixtures ---


@pytest.fixture
def vapo_config():
    return VAPOConfig(
        learning_rate=1e-5,
        value_lr=2e-5,
        batch_size=4,
        n_epochs=2,
        epsilon_low=0.2,
        epsilon_high=0.28,
        adaptive_gae_alpha=0.05,
        gamma=1.0,
    )


@pytest.fixture
def unified_model():
    return SimplePolicyUnified()


@pytest.fixture
def separate_models():
    return SimplePolicySeparate(), SimpleValueModel()


@pytest.fixture
def rollout_batch():
    """Create a sample rollout batch."""
    batch_size = 4
    seq_len = 10
    vocab_size = 20

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        # Dense rewards for testing [B, T] or scalar [B]
        "rewards": torch.randn(batch_size),  # Scalar rewards typically
    }


# --- VAPOConfig Tests ---


def test_vapo_config_defaults():
    config = VAPOConfig()
    assert config.epsilon_low == 0.2
    assert config.epsilon_high == 0.28
    assert config.adaptive_gae_alpha == 0.05
    assert config.lambda_value == 1.0


def test_vapo_config_validation():
    # epsilon_high must be >= epsilon_low
    with pytest.raises(ValueError):
        VAPOConfig(epsilon_low=0.3, epsilon_high=0.2)

    with pytest.raises(ValueError):
        VAPOConfig(adaptive_gae_alpha=0.0)


# --- VAPOAlgorithm Logic Tests ---


def test_init_separate_optimizers(separate_models, vapo_config):
    """Test that VAPO initializes separate optimizers correctly."""
    policy, value = separate_models
    algo = VAPOAlgorithm(policy_model=policy, value_model=value, config=vapo_config)

    assert algo.optimizer is not None
    assert algo.value_optimizer is not None
    assert algo.optimizer is not algo.value_optimizer

    # Check LR settings
    assert algo.optimizer.param_groups[0]["lr"] == vapo_config.learning_rate
    assert algo.value_optimizer.param_groups[0]["lr"] == vapo_config.value_lr


def test_compute_adaptive_gae_math(unified_model, vapo_config):
    """Test the math of adaptive GAE with varying lambdas."""
    algo = VAPOAlgorithm(policy_model=unified_model, config=vapo_config)

    # Setup simple scenario
    # T=2, Gamma=1.0
    # Sample 0: Lambda=1.0 (Monte Carlo)
    # Sample 1: Lambda=0.0 (One-step TD)

    rewards = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # [B=2, T=2]
    values = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # Zero values for simplicity
    lambdas = torch.tensor([1.0, 0.0])  # [B=2]

    # Manual Calculation:
    # deltas = r + gamma*V_next - V = r (since V=0) = [[1, 1], [1, 1]]

    # Sample 0 (Lambda=1.0):
    # A_1 = delta_1 = 1.0
    # A_0 = delta_0 + gamma*lambda*A_1 = 1.0 + 1.0*1.0*1.0 = 2.0
    # Expected: [2.0, 1.0]

    # Sample 1 (Lambda=0.0):
    # A_1 = delta_1 = 1.0
    # A_0 = delta_0 + gamma*lambda*A_1 = 1.0 + 1.0*0.0*1.0 = 1.0
    # Expected: [1.0, 1.0]

    advantages = algo.compute_adaptive_gae(rewards, values, lambdas)

    assert torch.allclose(advantages[0], torch.tensor([2.0, 1.0]))
    assert torch.allclose(advantages[1], torch.tensor([1.0, 1.0]))


def test_decoupled_gae_logic(unified_model, vapo_config, rollout_batch):
    """
    Test that train_on_rollout calls compute_adaptive_gae with
    different lambdas for policy and value updates.
    """
    algo = VAPOAlgorithm(policy_model=unified_model, config=vapo_config)

    # Mock compute_adaptive_gae to capture calls
    with patch.object(algo, "compute_adaptive_gae", return_value=torch.zeros((4, 10))) as mock_gae:
        # Run one pass
        algo.train_on_rollout(rollout_batch)

        assert mock_gae.call_count == 2

        # Call 1: Critic Updates (should be lambda_value = 1.0)
        args_critic, _ = mock_gae.call_args_list[0]
        lambdas_critic = args_critic[2]
        assert torch.all(lambdas_critic == 1.0)

        # Call 2: Policy Updates (should be adaptive < 1.0 for long sequences)
        args_policy, _ = mock_gae.call_args_list[1]
        lambdas_policy = args_policy[2]

        # With standard params (alpha=0.05, length=10), lambda != 1.0
        assert not torch.allclose(lambdas_policy, lambdas_critic)


def test_asymmetric_clipping(unified_model, vapo_config):
    """Test that loss computation uses asymmetric clipping bounds."""
    # Set distinct bounds
    vapo_config.epsilon_low = 0.1
    vapo_config.epsilon_high = 0.5
    algo = VAPOAlgorithm(policy_model=unified_model, config=vapo_config)

    batch_size = 1
    seq_len = 1

    # Mock data to force clipping check
    # We want ratio > 1.1 (should NOT clip if < 1.5)
    # We want ratio < 0.9 (should clip)

    batch = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "old_log_probs": torch.zeros(batch_size, seq_len),
        "old_values": torch.zeros(batch_size, seq_len),
        "advantages": torch.ones(batch_size, seq_len),  # Positive advantage
        "returns": torch.zeros(batch_size, seq_len),
        "raw_rewards": torch.zeros(batch_size),
    }

    # Scenario 1: Ratio = 1.4 (High range).
    # Standard PPO (sym 0.1) would clip at 1.1.
    # VAPO (high 0.5) should NOT clip at 1.4.

    with patch.object(algo, "get_log_probs", return_value=torch.tensor([[0.336]])):
        # log(1.4) approx 0.336. old_log_probs=0. Ratio approx 1.4.
        with patch.object(algo, "forward_value", return_value=torch.zeros(batch_size, seq_len)):
            loss_dict = algo.compute_loss(batch)

            # If not clipped, policy loss = -ratio * adv = -1.4 * 1 = -1.4
            # If clipped (sym 0.1), policy loss = -1.1 * 1 = -1.1

            # Since 1.4 < 1.0 + 0.5, VAPO does NOT clip.
            # Loss should be roughly -1.4
            assert loss_dict["policy_loss"].item() < -1.2


def test_lm_loss_activation(unified_model, vapo_config):
    """Test that Positive Example LM Loss activates for high rewards."""
    algo = VAPOAlgorithm(policy_model=unified_model, config=vapo_config)

    # Sample 0: Reward 1.0 (Correct) -> Should trigger LM loss
    # Sample 1: Reward 0.0 (Incorrect) -> Should NOT trigger LM loss

    batch_size = 2
    seq_len = 5

    batch = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "old_log_probs": torch.zeros(batch_size, seq_len),
        "old_values": torch.zeros(batch_size, seq_len),
        "advantages": torch.zeros(batch_size, seq_len),
        "returns": torch.zeros(batch_size, seq_len),
        "raw_rewards": torch.tensor([1.0, 0.0]),  # One correct, one wrong
    }

    # Mock output to return some log probs
    # We set log_probs to 0.0 -> NLL = 0.0 ideally, but let's set to -1.0 to see positive loss

    with patch.object(algo, "get_log_probs", return_value=torch.ones(batch_size, seq_len) * -1.0):
        with patch.object(algo, "forward_value", return_value=torch.zeros(batch_size, seq_len)):
            loss_dict = algo.compute_loss(batch)

            assert "lm_loss" in loss_dict
            assert loss_dict["lm_loss"] > 0

            # Check calculation:
            # Sample 0 is correct. NLL = -(-1.0) = 1.0. Avg over 5 tokens = 1.0.
            # Sample 1 is ignored.
            assert torch.isclose(loss_dict["lm_loss"], torch.tensor(1.0))


def test_token_level_loss_aggregation(unified_model, vapo_config):
    """Verify that loss is normalized by total tokens, not batch size."""
    algo = VAPOAlgorithm(policy_model=unified_model, config=vapo_config)

    # Batch of 2 sequences
    # Seq 1: Length 2 (Mask 1 1 0 0)
    # Seq 2: Length 4 (Mask 1 1 1 1)

    batch_size = 2
    seq_len = 4

    batch = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[1, 1, -100, -100], [1, 1, 1, 1]], dtype=torch.long),
        "old_log_probs": torch.zeros(batch_size, seq_len),
        "old_values": torch.zeros(batch_size, seq_len),
        "advantages": torch.ones(batch_size, seq_len),  # Adv = 1 everywhere
        "returns": torch.zeros(batch_size, seq_len),
        "raw_rewards": torch.zeros(batch_size),
    }

    # Mock log probs = 0 (Ratio = 1) -> Loss per token = -1
    # Total loss sum = -2 (seq1) + -4 (seq2) = -6
    # Token-level avg = -6 / 6 tokens = -1.0

    with patch.object(algo, "get_log_probs", return_value=torch.zeros(batch_size, seq_len)):
        with patch.object(algo, "forward_value", return_value=torch.zeros(batch_size, seq_len)):
            loss_dict = algo.compute_loss(batch)
            assert torch.isclose(loss_dict["policy_loss"], torch.tensor(-1.0))
