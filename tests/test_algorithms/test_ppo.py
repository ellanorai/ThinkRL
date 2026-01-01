"""
Test Suite for ThinkRL PPO Algorithm (Infra-Grade)
==================================================

Tests for the refactored PPO implementation focusing on:
- RLHF/Token-level PPO logic
- Unified vs Separate Policy/Value models
- PPO Loss correctness (clipping, entropy, value loss)
- Training loop mechanics (train_on_rollout)
- Gradient flow

Author: EllanorAI
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Explicitly import create_ppo to avoid NameError
from thinkrl.algorithms.ppo import (
    PPOAlgorithm,
    PPOConfig,
    create_ppo,
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
def ppo_config():
    return PPOConfig(
        learning_rate=1e-4,
        batch_size=4,
        n_epochs=2,
        policy_clip=0.2,
        value_clip=0.2,
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
    batch_size = 8
    seq_len = 10
    vocab_size = 20

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        # FIX: Explicitly use torch.long for masks to avoid indexing errors
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        # Dense rewards for testing [B, T]
        "rewards": torch.randn(batch_size, seq_len),
    }


# --- PPOConfig Tests ---


def test_ppo_config_defaults():
    config = PPOConfig()
    assert config.learning_rate == 3e-4
    assert config.gamma == 0.99
    assert config.n_epochs == 4
    assert config.batch_size == 64


def test_ppo_config_validation():
    with pytest.raises(AssertionError):
        PPOConfig(n_epochs=0)
    with pytest.raises(AssertionError):
        PPOConfig(policy_clip=1.5)


# --- PPOAlgorithm Tests ---


def test_init_unified_model(unified_model, ppo_config):
    """Test initialization with a unified actor-critic model."""
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)

    assert algo.value_model is None
    assert algo.value_optimizer is None
    # Base class handles the main optimizer
    assert algo.optimizer is not None


def test_init_separate_models(separate_models, ppo_config):
    """Test initialization with separate actor and critic models."""
    policy, value = separate_models
    algo = PPOAlgorithm(policy_model=policy, value_model=value, config=ppo_config)

    assert algo.value_model is value
    assert algo.value_optimizer is not None
    assert algo.value_optimizer is not algo.optimizer


def test_forward_value_unified(unified_model, ppo_config):
    """Test value estimation using unified model."""
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)
    input_ids = torch.randint(0, 20, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)

    values = algo.forward_value(input_ids, mask)

    assert values.shape == (2, 5)
    assert values.requires_grad


def test_forward_value_separate(separate_models, ppo_config):
    """Test value estimation using separate critic model."""
    policy, value = separate_models
    algo = PPOAlgorithm(policy_model=policy, value_model=value, config=ppo_config)
    input_ids = torch.randint(0, 20, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)

    values = algo.forward_value(input_ids, mask)

    assert values.shape == (2, 5)
    assert values.requires_grad


def test_forward_value_missing_head(separate_models, ppo_config):
    """Test error when unified model is expected but 'values' missing."""
    policy, _ = separate_models  # separate policy has no value head
    algo = PPOAlgorithm(policy_model=policy, config=ppo_config)

    input_ids = torch.randint(0, 20, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)

    with pytest.raises(ValueError, match="did not return 'values'"):
        algo.forward_value(input_ids, mask)


def test_train_on_rollout_execution(unified_model, ppo_config, rollout_batch):
    """Test the full training loop execution (integration test)."""
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)

    metrics = algo.train_on_rollout(rollout_batch)

    # Check output structure
    assert isinstance(metrics, list)
    assert len(metrics) == ppo_config.n_epochs
    assert "loss" in metrics[0]
    assert "epoch" in metrics[0]
    assert metrics[0]["epoch"] == 0


def test_train_on_rollout_separate_models(separate_models, ppo_config, rollout_batch):
    """Test training loop with separate actor/critic."""
    policy, value = separate_models
    algo = PPOAlgorithm(policy_model=policy, value_model=value, config=ppo_config)

    metrics = algo.train_on_rollout(rollout_batch)

    assert len(metrics) == ppo_config.n_epochs
    assert "value_loss" in metrics[0]


def test_compute_loss_components(unified_model, ppo_config):
    """Test individual loss components computation."""
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)

    batch_size, seq_len = 4, 10

    # Mock batch data similar to what train_on_rollout produces
    batch = {
        "input_ids": torch.randint(0, 20, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 20, (batch_size, seq_len)),
        "old_log_probs": torch.randn(batch_size, seq_len),
        "old_values": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size, seq_len),
        "returns": torch.randn(batch_size, seq_len),
    }

    loss_dict = algo.compute_loss(batch)

    assert "loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy_loss" in loss_dict

    # Check gradients
    loss_dict["loss"].backward()
    for param in unified_model.parameters():
        assert param.grad is not None


def test_value_clipping(unified_model, ppo_config):
    """Test that value clipping logic is active."""
    ppo_config.value_clip = 0.1
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)

    # Create batch where new values diverge significantly from old values
    batch = {
        "input_ids": torch.randint(0, 20, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.randint(0, 20, (2, 5)),
        "old_log_probs": torch.randn(2, 5),
        "old_values": torch.zeros(2, 5),  # Old values are 0
        "advantages": torch.zeros(2, 5),
        "returns": torch.ones(2, 5) * 10,  # Returns are far away (10)
    }

    # Force model to output something large
    with patch.object(algo, "forward_value", return_value=torch.ones(2, 5) * 5):
        with patch.object(algo, "get_log_probs", return_value=torch.randn(2, 5)):
            loss_dict = algo.compute_loss(batch)

            # Since clipping is active and diff is large, value loss should be non-zero
            assert loss_dict["value_loss"] > 0


def test_training_step_gradients(separate_models, ppo_config):
    """Test that training step advances separate optimizers."""
    policy, value = separate_models
    algo = PPOAlgorithm(policy_model=policy, value_model=value, config=ppo_config)

    # Create mock batch
    batch = {
        "input_ids": torch.randint(0, 20, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.randint(0, 20, (2, 5)),
        "old_log_probs": torch.randn(2, 5),
        "old_values": torch.randn(2, 5),
        "advantages": torch.randn(2, 5),
        "returns": torch.randn(2, 5),
    }

    # Take a step
    # Assign to _ to avoid unused variable warning (metrics)
    # This call uses 'algo' and 'batch', preventing F841 errors for them
    _ = algo.training_step(batch)

    # Check if params updated (naively check if grad was computed)
    has_policy_grad = any(p.grad is not None for p in policy.parameters())
    has_value_grad = any(p.grad is not None for p in value.parameters())

    assert has_policy_grad
    assert has_value_grad


def test_create_ppo_factory(unified_model):
    """Test factory function."""
    algo = create_ppo(unified_model, learning_rate=1e-5, n_epochs=3)

    assert isinstance(algo, PPOAlgorithm)
    assert algo.config.learning_rate == 1e-5
    assert algo.config.n_epochs == 3


def test_sparse_rewards_handling(unified_model, ppo_config):
    """Test train_on_rollout with sequence-level scalar rewards."""
    algo = PPOAlgorithm(policy_model=unified_model, config=ppo_config)

    batch_size = 4
    seq_len = 5

    batch = {
        "input_ids": torch.randint(0, 20, (batch_size, seq_len)),
        # FIX: dtype=torch.long is required for proper indexing in sparse reward assignment
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        # Rewards are scalars [B]
        "rewards": torch.randn(batch_size),
    }

    # This should run without error and internally map scalars to dense rewards
    metrics = algo.train_on_rollout(batch)
    assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
