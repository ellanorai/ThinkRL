"""
Tests for REINFORCE Algorithm
=============================

Comprehensive tests for Monte Carlo policy gradient implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.reinforce import (
    REINFORCEAlgorithm,
    REINFORCEConfig,
    create_reinforce,
)


class SimplePolicyModel(nn.Module):
    """Simple policy model for testing."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        logits = self.fc(x)
        return {"logits": logits}


class TestREINFORCEConfig:
    """Tests for REINFORCEConfig dataclass."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = REINFORCEConfig()

        assert config.learning_rate == 1e-5
        assert config.baseline_type == "moving_average"
        assert config.baseline_momentum == 0.99
        assert config.entropy_coeff == 0.01
        assert config.kl_coeff == 0.0
        assert config.gamma == 1.0

    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = REINFORCEConfig(
            learning_rate=1e-4,
            baseline_type="batch_mean",
            entropy_coeff=0.02,
        )

        assert config.learning_rate == 1e-4
        assert config.baseline_type == "batch_mean"
        assert config.entropy_coeff == 0.02

    def test_valid_baseline_types(self):
        """Test all valid baseline types."""
        for baseline_type in ["none", "moving_average", "batch_mean"]:
            config = REINFORCEConfig(baseline_type=baseline_type)
            assert config.baseline_type == baseline_type

    def test_invalid_baseline_type(self):
        """Test invalid baseline type raises error."""
        with pytest.raises(AssertionError):
            REINFORCEConfig(baseline_type="invalid")

    def test_invalid_baseline_momentum(self):
        """Test invalid baseline momentum raises error."""
        with pytest.raises(AssertionError):
            REINFORCEConfig(baseline_momentum=1.5)

        with pytest.raises(AssertionError):
            REINFORCEConfig(baseline_momentum=-0.1)


class TestREINFORCEAlgorithm:
    """Tests for REINFORCEAlgorithm class."""

    @pytest.fixture
    def policy_model(self):
        """Create simple policy model."""
        return SimplePolicyModel()

    @pytest.fixture
    def algorithm(self, policy_model):
        """Create REINFORCE algorithm."""
        return REINFORCEAlgorithm(
            policy_model=policy_model,
            config=REINFORCEConfig(learning_rate=1e-4),
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        batch_size = 4
        seq_len = 16

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        labels[:, :8] = -100  # Mask prompt tokens
        rewards = torch.randn(batch_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rewards": rewards,
        }

    def test_initialization(self, policy_model):
        """Test algorithm initialization."""
        algo = REINFORCEAlgorithm(policy_model=policy_model)

        assert algo.policy_model is policy_model
        assert algo.config is not None
        assert algo._baseline_mean == 0.0
        assert algo._baseline_initialized is False

    def test_initialization_with_config(self, policy_model):
        """Test algorithm initialization with config."""
        config = REINFORCEConfig(
            baseline_type="batch_mean",
            entropy_coeff=0.05,
        )
        algo = REINFORCEAlgorithm(
            policy_model=policy_model,
            config=config,
        )

        assert algo.config.baseline_type == "batch_mean"
        assert algo.config.entropy_coeff == 0.05

    def test_initialization_creates_optimizer(self, policy_model):
        """Test algorithm creates optimizer if not provided."""
        algo = REINFORCEAlgorithm(policy_model=policy_model)

        assert algo.optimizer is not None


class TestComputeReturns:
    """Tests for compute_returns method."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm for testing."""
        model = SimplePolicyModel()
        return REINFORCEAlgorithm(policy_model=model)

    def test_sequence_level_rewards(self, algorithm):
        """Test returns computation for sequence-level rewards."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        returns = algorithm.compute_returns(rewards)

        assert returns.shape == rewards.shape
        assert torch.allclose(returns, rewards)

    def test_token_level_rewards(self, algorithm):
        """Test returns computation for token-level rewards."""
        # With gamma=1.0, returns should be cumulative sum from end
        algorithm.config.gamma = 1.0
        rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
            ]
        )

        returns = algorithm.compute_returns(rewards)

        assert returns.shape == rewards.shape
        # Last token should equal reward
        assert torch.allclose(returns[:, -1], rewards[:, -1])

    def test_discounted_returns(self, algorithm):
        """Test discounted returns computation."""
        algorithm.config.gamma = 0.9
        rewards = torch.tensor(
            [
                [1.0, 1.0, 1.0],
            ]
        )

        returns = algorithm.compute_returns(rewards)

        # Returns should be discounted
        expected_last = 1.0
        expected_mid = 1.0 + 0.9 * 1.0
        expected_first = 1.0 + 0.9 * expected_mid

        assert torch.isclose(returns[0, 2], torch.tensor(expected_last))
        assert torch.isclose(returns[0, 1], torch.tensor(expected_mid))
        assert torch.isclose(returns[0, 0], torch.tensor(expected_first))


class TestComputeBaseline:
    """Tests for compute_baseline method."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm for testing."""
        model = SimplePolicyModel()
        return REINFORCEAlgorithm(policy_model=model)

    def test_no_baseline(self, algorithm):
        """Test no baseline returns zero."""
        algorithm.config.baseline_type = "none"
        returns = torch.tensor([1.0, 2.0, 3.0])

        baseline = algorithm.compute_baseline(returns)

        assert baseline.item() == 0.0

    def test_batch_mean_baseline(self, algorithm):
        """Test batch mean baseline."""
        algorithm.config.baseline_type = "batch_mean"
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])

        baseline = algorithm.compute_baseline(returns)

        expected = returns.mean()
        assert torch.isclose(baseline, expected)

    def test_moving_average_baseline_init(self, algorithm):
        """Test moving average baseline initialization."""
        algorithm.config.baseline_type = "moving_average"
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])

        baseline = algorithm.compute_baseline(returns)

        # First call should initialize with batch mean
        assert algorithm._baseline_initialized is True
        assert torch.isclose(baseline, returns.mean())

    def test_moving_average_baseline_update(self, algorithm):
        """Test moving average baseline updates."""
        algorithm.config.baseline_type = "moving_average"
        algorithm.config.baseline_momentum = 0.9

        returns1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        returns2 = torch.tensor([2.0, 2.0, 2.0, 2.0])

        algorithm.compute_baseline(returns1)
        baseline2 = algorithm.compute_baseline(returns2)

        # Should be EMA: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected = 0.9 * 1.0 + 0.1 * 2.0
        assert torch.isclose(baseline2, torch.tensor(expected))


class TestComputeLoss:
    """Tests for compute_loss method."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm for testing."""
        model = SimplePolicyModel()
        return REINFORCEAlgorithm(
            policy_model=model,
            config=REINFORCEConfig(entropy_coeff=0.0, kl_coeff=0.0),
        )

    @pytest.fixture
    def batch(self):
        """Create sample batch."""
        return {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "labels": torch.randint(0, 100, (2, 8)),
            "rewards": torch.tensor([1.0, -1.0]),
        }

    def test_compute_loss_returns_dict(self, algorithm, batch):
        """Test compute_loss returns dictionary."""
        result = algorithm.compute_loss(batch)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "policy_loss" in result

    def test_compute_loss_contains_metrics(self, algorithm, batch):
        """Test compute_loss contains expected metrics."""
        result = algorithm.compute_loss(batch)

        expected_keys = [
            "loss",
            "policy_loss",
            "entropy",
            "entropy_loss",
            "kl_div",
            "kl_loss",
            "reward_mean",
            "reward_std",
            "baseline",
            "advantage_mean",
        ]
        for key in expected_keys:
            assert key in result

    def test_compute_loss_gradients_flow(self, algorithm, batch):
        """Test gradients flow through loss."""
        result = algorithm.compute_loss(batch)
        loss = result["loss"]

        assert loss.requires_grad
        loss.backward()

        # Check gradients exist
        for param in algorithm.policy_model.parameters():
            assert param.grad is not None

    def test_compute_loss_with_entropy(self, algorithm, batch):
        """Test loss includes entropy when configured."""
        algorithm.config.entropy_coeff = 0.01
        result = algorithm.compute_loss(batch)

        assert result["entropy"].item() > 0
        assert result["entropy_loss"].item() != 0

    def test_compute_loss_masks_prompt(self, algorithm, batch):
        """Test loss correctly masks prompt tokens."""
        batch["labels"][:, :4] = -100  # Mask first 4 tokens

        result = algorithm.compute_loss(batch)

        # Loss should still compute
        assert not torch.isnan(result["loss"])
        assert not torch.isinf(result["loss"])


class TestTrainingStep:
    """Tests for training_step method."""

    @pytest.fixture
    def algorithm(self):
        """Create algorithm for testing."""
        model = SimplePolicyModel()
        return REINFORCEAlgorithm(policy_model=model)

    @pytest.fixture
    def batch(self):
        """Create sample batch."""
        return {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "labels": torch.randint(0, 100, (2, 8)),
            "rewards": torch.tensor([1.0, -1.0]),
        }

    def test_training_step_returns_metrics(self, algorithm, batch):
        """Test training_step returns metrics dictionary."""
        metrics = algorithm.training_step(batch)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_training_step_updates_weights(self, algorithm, batch):
        """Test training_step updates model weights."""
        # Get initial weights
        initial_weights = {name: param.clone() for name, param in algorithm.policy_model.named_parameters()}

        algorithm.training_step(batch)

        # Check weights changed
        for name, param in algorithm.policy_model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param, initial_weights[name])

    def test_training_step_contains_grad_norm(self, algorithm, batch):
        """Test training_step returns gradient norm."""
        metrics = algorithm.training_step(batch)

        assert "grad_norm" in metrics
        assert metrics["grad_norm"] >= 0


class TestCreateReinforce:
    """Tests for create_reinforce factory function."""

    def test_creates_algorithm(self):
        """Test factory creates algorithm."""
        model = SimplePolicyModel()
        algo = create_reinforce(model)

        assert isinstance(algo, REINFORCEAlgorithm)

    def test_with_custom_config(self):
        """Test factory with custom parameters."""
        model = SimplePolicyModel()
        algo = create_reinforce(
            model,
            learning_rate=1e-3,
            baseline_type="batch_mean",
            entropy_coeff=0.05,
        )

        assert algo.config.learning_rate == 1e-3
        assert algo.config.baseline_type == "batch_mean"
        assert algo.config.entropy_coeff == 0.05

    def test_with_ref_model(self):
        """Test factory with reference model."""
        model = SimplePolicyModel()
        ref_model = SimplePolicyModel()

        algo = create_reinforce(model, ref_model=ref_model, kl_coeff=0.1)

        assert algo.ref_model is ref_model
        assert algo.config.kl_coeff == 0.1

    def test_with_custom_optimizer(self):
        """Test factory with custom optimizer."""
        model = SimplePolicyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        algo = create_reinforce(model, optimizer=optimizer)

        assert algo.optimizer is optimizer


class TestREINFORCEIntegration:
    """Integration tests for REINFORCE algorithm."""

    def test_full_training_loop(self):
        """Test full training loop."""
        model = SimplePolicyModel()
        algo = create_reinforce(model, learning_rate=1e-3)

        losses = []
        for _step in range(5):
            batch = {
                "input_ids": torch.randint(0, 100, (4, 16)),
                "attention_mask": torch.ones(4, 16),
                "labels": torch.randint(0, 100, (4, 16)),
                "rewards": torch.randn(4),
            }

            metrics = algo.training_step(batch)
            losses.append(metrics["loss"])

        # Training should complete without errors
        assert len(losses) == 5
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    def test_baseline_accumulation(self):
        """Test baseline accumulates over training."""
        model = SimplePolicyModel()
        algo = create_reinforce(
            model,
            baseline_type="moving_average",
            baseline_momentum=0.9,
        )

        baselines = []
        for step in range(10):
            batch = {
                "input_ids": torch.randint(0, 100, (4, 16)),
                "attention_mask": torch.ones(4, 16),
                "labels": torch.randint(0, 100, (4, 16)),
                "rewards": torch.ones(4) * (step + 1),  # Increasing rewards
            }

            algo.training_step(batch)
            baselines.append(algo._baseline_mean)

        # Baseline should increase with increasing rewards
        assert baselines[-1] > baselines[0]

    def test_with_variable_length_sequences(self):
        """Test with variable length sequences."""
        model = SimplePolicyModel()
        algo = create_reinforce(model)

        # Different sequence lengths
        for seq_len in [8, 16, 32]:
            batch = {
                "input_ids": torch.randint(0, 100, (2, seq_len)),
                "attention_mask": torch.ones(2, seq_len),
                "labels": torch.randint(0, 100, (2, seq_len)),
                "rewards": torch.randn(2),
            }

            metrics = algo.training_step(batch)
            assert not torch.isnan(torch.tensor(metrics["loss"]))
