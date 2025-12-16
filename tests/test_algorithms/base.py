"""
Test Suite for ThinkRL DAPO Algorithm
=====================================

Comprehensive tests for:
- DAPOConfig validation
- DynamicSamplingBuffer operations
- DAPOAlgorithm core functionality
- Loss computation and diagnostics
- Multi-epoch training
- Dynamic sampling workflow

Author: Archit Sood @ EllanorAI
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.dapo import (
    DAPOAlgorithm,
    DAPOConfig,
    DynamicSamplingBuffer,
    create_dapo,
)


# --- Helper Classes ---


class SimplePolicy(nn.Module):
    """Simple policy model for testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return {"logits": logits}


# --- Fixtures ---


@pytest.fixture
def simple_model():
    """Create a simple policy model."""
    return SimplePolicy(vocab_size=100, hidden_size=32)


@pytest.fixture
def default_config():
    """Create default DAPO config."""
    return DAPOConfig()


@pytest.fixture
def small_config():
    """Create config suitable for small test batches."""
    return DAPOConfig(
        group_size=4,
        min_batch_size=8,
        max_len=128,
        cache_len=32,
        n_epochs=2,
    )


@pytest.fixture
def dapo_algorithm(simple_model, small_config):
    """Create DAPO algorithm instance."""
    return DAPOAlgorithm(
        policy_model=simple_model,
        config=small_config,
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 8
    seq_len = 16

    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 100, (batch_size, seq_len)),
        "rewards": torch.randn(batch_size),
    }


# --- DAPOConfig Tests ---


class TestDAPOConfig:
    """Tests for DAPOConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DAPOConfig()

        assert config.learning_rate == 1e-6
        assert config.epsilon_low == 0.2
        assert config.epsilon_high == 0.28
        assert config.group_size == 16
        assert config.beta == 0.0
        assert config.n_epochs == 1
        assert config.dynamic_sampling is True
        assert config.use_overlong_punishment is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DAPOConfig(
            learning_rate=1e-5,
            epsilon_low=0.1,
            epsilon_high=0.3,
            group_size=8,
            n_epochs=3,
        )

        assert config.learning_rate == 1e-5
        assert config.epsilon_low == 0.1
        assert config.epsilon_high == 0.3
        assert config.group_size == 8
        assert config.n_epochs == 3

    def test_validation_epsilon_low_positive(self):
        """Test that epsilon_low must be positive."""
        with pytest.raises(AssertionError, match="epsilon_low must be positive"):
            DAPOConfig(epsilon_low=0)

        with pytest.raises(AssertionError, match="epsilon_low must be positive"):
            DAPOConfig(epsilon_low=-0.1)

    def test_validation_epsilon_high_ge_low(self):
        """Test that epsilon_high >= epsilon_low."""
        with pytest.raises(AssertionError, match="epsilon_high >= epsilon_low"):
            DAPOConfig(epsilon_low=0.3, epsilon_high=0.2)

    def test_validation_group_size(self):
        """Test that group_size must be >= 2."""
        with pytest.raises(AssertionError, match="group_size must be >= 2"):
            DAPOConfig(group_size=1)

    def test_validation_cache_len(self):
        """Test that cache_len must be positive."""
        with pytest.raises(AssertionError, match="cache_len must be positive"):
            DAPOConfig(cache_len=0)

    def test_validation_max_len_exceeds_cache(self):
        """Test that max_len must exceed cache_len."""
        with pytest.raises(AssertionError, match="max_len must exceed cache_len"):
            DAPOConfig(max_len=100, cache_len=100)

        with pytest.raises(AssertionError, match="max_len must exceed cache_len"):
            DAPOConfig(max_len=50, cache_len=100)

    def test_validation_n_epochs(self):
        """Test that n_epochs must be >= 1."""
        with pytest.raises(AssertionError, match="n_epochs must be >= 1"):
            DAPOConfig(n_epochs=0)


# --- DynamicSamplingBuffer Tests ---


class TestDynamicSamplingBuffer:
    """Tests for DynamicSamplingBuffer."""

    def test_init(self, small_config):
        """Test buffer initialization."""
        buffer = DynamicSamplingBuffer(small_config)

        assert buffer.config == small_config
        assert len(buffer.buffer) == 0
        assert buffer.total_sampled == 0
        assert buffer.total_filtered == 0

    def test_add_samples_valid(self, small_config):
        """Test adding samples with variance."""
        buffer = DynamicSamplingBuffer(small_config)

        # Create samples with variance in rewards
        samples = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16),
        }
        # Rewards with clear variance within each group
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        valid_count = buffer.add_samples(samples, rewards)

        assert valid_count == 8  # Both groups have variance
        assert len(buffer.buffer) == 2  # 2 groups of 4
        assert buffer.total_sampled == 2
        assert buffer.total_filtered == 0

    def test_add_samples_zero_variance(self, small_config):
        """Test filtering samples with zero variance."""
        buffer = DynamicSamplingBuffer(small_config)

        samples = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16),
        }
        # First group has variance, second group has no variance
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0])

        valid_count = buffer.add_samples(samples, rewards)

        assert valid_count == 4  # Only first group has variance
        assert len(buffer.buffer) == 1
        assert buffer.total_sampled == 2
        assert buffer.total_filtered == 1

    def test_add_samples_batch_size_error(self, small_config):
        """Test error when batch size not divisible by group size."""
        buffer = DynamicSamplingBuffer(small_config)

        samples = {"input_ids": torch.randint(0, 100, (7, 16))}
        rewards = torch.randn(7)

        with pytest.raises(ValueError, match="must be divisible by group_size"):
            buffer.add_samples(samples, rewards)

    def test_is_ready(self, small_config):
        """Test is_ready check."""
        buffer = DynamicSamplingBuffer(small_config)

        assert not buffer.is_ready()

        # Add enough samples
        samples = {"input_ids": torch.randint(0, 100, (8, 16))}
        rewards = torch.randn(8)
        buffer.add_samples(samples, rewards)

        assert buffer.is_ready()

    def test_get_batch(self, small_config):
        """Test getting a batch from buffer."""
        buffer = DynamicSamplingBuffer(small_config)

        # Add samples
        samples = {"input_ids": torch.randint(0, 100, (8, 16))}
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        buffer.add_samples(samples, rewards)

        # Get batch
        batch = buffer.get_batch(8)

        assert batch is not None
        assert "input_ids" in batch
        assert "rewards" in batch
        assert batch["input_ids"].shape[0] == 8
        assert len(buffer.buffer) == 0  # Buffer should be empty

    def test_get_batch_insufficient(self, small_config):
        """Test get_batch returns None when insufficient samples."""
        buffer = DynamicSamplingBuffer(small_config)

        # Add only 4 samples
        samples = {"input_ids": torch.randint(0, 100, (4, 16))}
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
        buffer.add_samples(samples, rewards)

        # Try to get 8 samples
        batch = buffer.get_batch(8)

        assert batch is None

    def test_get_stats(self, small_config):
        """Test getting buffer statistics."""
        buffer = DynamicSamplingBuffer(small_config)

        # Initial stats
        stats = buffer.get_stats()
        assert stats["filter_ratio"] == 0.0
        assert stats["buffer_size"] == 0

        # Add samples
        samples = {"input_ids": torch.randint(0, 100, (8, 16))}
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0])
        buffer.add_samples(samples, rewards)

        stats = buffer.get_stats()
        assert stats["filter_ratio"] == 0.5
        assert stats["buffer_size"] == 4
        assert stats["total_sampled"] == 2
        assert stats["total_filtered"] == 1

    def test_clear(self, small_config):
        """Test clearing the buffer."""
        buffer = DynamicSamplingBuffer(small_config)

        samples = {"input_ids": torch.randint(0, 100, (8, 16))}
        rewards = torch.randn(8)
        buffer.add_samples(samples, rewards)

        buffer.clear()

        assert len(buffer.buffer) == 0
        assert buffer.total_sampled == 0
        assert buffer.total_filtered == 0


# --- DAPOAlgorithm Tests ---


class TestDAPOAlgorithm:
    """Tests for DAPOAlgorithm class."""

    def test_init_default(self, simple_model):
        """Test initialization with defaults."""
        algo = DAPOAlgorithm(policy_model=simple_model)

        assert algo.policy_model is simple_model
        assert algo.config is not None
        assert algo.sampling_buffer is not None
        assert algo.optimizer is not None

    def test_init_with_config(self, simple_model, small_config):
        """Test initialization with custom config."""
        algo = DAPOAlgorithm(policy_model=simple_model, config=small_config)

        assert algo.config == small_config
        assert algo.config.group_size == 4

    def test_init_with_optimizer(self, simple_model):
        """Test initialization with custom optimizer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        algo = DAPOAlgorithm(policy_model=simple_model, optimizer=optimizer)

        assert algo.optimizer is optimizer

    def test_validate_model_error(self):
        """Test validation error for invalid model."""
        invalid_model = object()

        with pytest.raises(ValueError, match="must implement forward"):
            DAPOAlgorithm(policy_model=invalid_model)  # type: ignore[arg-type]

    def test_get_log_probs_dict_output(self, dapo_algorithm, sample_batch):
        """Test get_log_probs with dict output."""
        outputs = dapo_algorithm.policy_model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
        )

        log_probs = dapo_algorithm.get_log_probs(outputs, sample_batch["labels"])

        assert log_probs.shape == sample_batch["labels"].shape

    def test_get_log_probs_tensor_output(self, dapo_algorithm, sample_batch):
        """Test get_log_probs with raw tensor output."""
        outputs = dapo_algorithm.policy_model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
        )
        logits = outputs["logits"]

        log_probs = dapo_algorithm.get_log_probs(logits, sample_batch["labels"])

        assert log_probs.shape == sample_batch["labels"].shape

    def test_get_log_probs_masked_positions(self, dapo_algorithm):
        """Test that masked positions have zero log probs."""
        batch_size, seq_len = 4, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 100, (batch_size, seq_len))
        labels[:, -2:] = -100  # Mask last 2 positions

        outputs = dapo_algorithm.policy_model(input_ids=input_ids)
        log_probs = dapo_algorithm.get_log_probs(outputs, labels)

        # Check masked positions are zero
        assert torch.all(log_probs[:, -2:] == 0)

    def test_compute_advantages(self, dapo_algorithm):
        """Test advantage computation."""
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        advantages = dapo_algorithm.compute_advantages(rewards)

        assert advantages.shape == rewards.shape
        # Check normalization within groups
        grouped = advantages.view(2, 4)
        assert torch.allclose(grouped.mean(dim=1), torch.zeros(2), atol=1e-6)

    def test_compute_overlong_penalty_no_penalty(self, dapo_algorithm):
        """Test no penalty for short sequences."""
        cfg = dapo_algorithm.config
        seq_lengths = torch.tensor([cfg.max_len - cfg.cache_len - 100])

        penalties = dapo_algorithm.compute_overlong_penalty(seq_lengths)

        assert penalties[0] == 0.0

    def test_compute_overlong_penalty_buffer_zone(self, dapo_algorithm):
        """Test linear penalty in buffer zone."""
        cfg = dapo_algorithm.config
        # In the middle of buffer zone
        mid_point = cfg.max_len - cfg.cache_len // 2
        seq_lengths = torch.tensor([mid_point])

        penalties = dapo_algorithm.compute_overlong_penalty(seq_lengths)

        assert -1.0 < penalties[0] < 0.0

    def test_compute_overlong_penalty_full(self, dapo_algorithm):
        """Test full penalty beyond max_len."""
        cfg = dapo_algorithm.config
        seq_lengths = torch.tensor([cfg.max_len + 100])

        penalties = dapo_algorithm.compute_overlong_penalty(seq_lengths)

        assert penalties[0] == -1.0

    def test_compute_rollout_log_probs(self, dapo_algorithm, sample_batch):
        """Test rollout log probs computation."""
        log_probs = dapo_algorithm.compute_rollout_log_probs(sample_batch)

        assert log_probs.shape == sample_batch["labels"].shape
        assert not log_probs.requires_grad

    def test_compute_loss(self, dapo_algorithm, sample_batch):
        """Test loss computation."""
        loss_dict = dapo_algorithm.compute_loss(sample_batch)

        assert "loss" in loss_dict
        assert "policy_loss" in loss_dict
        assert "entropy_loss" in loss_dict
        assert "clip_frac" in loss_dict
        assert "approx_kl" in loss_dict
        assert "advantage_mean" in loss_dict
        assert "reward_mean" in loss_dict
        assert "ratio_mean" in loss_dict

        assert loss_dict["loss"].requires_grad

    def test_compute_loss_with_old_log_probs(self, dapo_algorithm, sample_batch):
        """Test loss computation with pre-computed old_log_probs."""
        old_log_probs = dapo_algorithm.compute_rollout_log_probs(sample_batch)
        loss_dict = dapo_algorithm.compute_loss(sample_batch, old_log_probs)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_compute_loss_with_entropy(self, simple_model):
        """Test loss computation with entropy bonus."""
        config = DAPOConfig(group_size=4, entropy_coeff=0.01)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        loss_dict = algo.compute_loss(batch)

        assert loss_dict["entropy_loss"] != 0.0

    def test_compute_loss_without_overlong(self, simple_model):
        """Test loss computation without overlong punishment."""
        config = DAPOConfig(group_size=4, use_overlong_punishment=False)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        # Should not raise error
        loss_dict = algo.compute_loss(batch)
        assert "loss" in loss_dict

    def test_compute_diagnostics_clip_fractions(self, dapo_algorithm):
        """Test diagnostic computation for clip fractions."""
        # Create data where clipping happens
        ratio = torch.tensor([[0.5, 1.0, 1.5], [1.0, 1.3, 1.6]])
        log_probs = torch.zeros_like(ratio)
        old_log_probs = torch.zeros_like(ratio)
        advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        token_mask = torch.ones_like(ratio, dtype=torch.bool)
        rewards = torch.tensor([1.0, -1.0])

        metrics = dapo_algorithm._compute_diagnostics(
            ratio=ratio,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            token_mask=token_mask,
            rewards=rewards,
        )

        assert "clip_frac" in metrics
        assert "clip_frac_high" in metrics
        assert "clip_frac_low" in metrics
        assert 0.0 <= metrics["clip_frac"] <= 1.0

    def test_training_step(self, dapo_algorithm, sample_batch):
        """Test single training step."""
        initial_params = [p.clone() for p in dapo_algorithm.policy_model.parameters()]

        metrics = dapo_algorithm.training_step(sample_batch)

        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert isinstance(metrics["loss"], float)

        # Check parameters were updated
        updated = False
        for p_init, p_new in zip(initial_params, dapo_algorithm.policy_model.parameters()):
            if not torch.allclose(p_init, p_new):
                updated = True
                break
        assert updated

    def test_training_step_with_old_log_probs(self, dapo_algorithm, sample_batch):
        """Test training step with pre-computed log probs."""
        old_log_probs = dapo_algorithm.compute_rollout_log_probs(sample_batch)

        metrics = dapo_algorithm.training_step(sample_batch, old_log_probs)

        assert "loss" in metrics

    def test_train_on_rollout_single_epoch(self, simple_model):
        """Test multi-epoch training with n_epochs=1."""
        config = DAPOConfig(group_size=4, n_epochs=1)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        all_metrics = algo.train_on_rollout(batch)

        assert len(all_metrics) == 1
        assert all_metrics[0]["epoch"] == 0

    def test_train_on_rollout_multi_epoch(self, simple_model):
        """Test multi-epoch training with n_epochs > 1."""
        config = DAPOConfig(group_size=4, n_epochs=3)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        all_metrics = algo.train_on_rollout(batch)

        assert len(all_metrics) == 3
        for i, metrics in enumerate(all_metrics):
            assert metrics["epoch"] == i

    def test_train_on_rollout_early_stop(self, simple_model):
        """Test early stopping when KL is too high."""
        config = DAPOConfig(group_size=4, n_epochs=10)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        # Mock high KL
        with patch.object(algo, "compute_loss") as mock_loss:
            mock_loss.return_value = {
                "loss": torch.tensor(1.0, requires_grad=True),
                "policy_loss": torch.tensor(1.0),
                "entropy_loss": torch.tensor(0.0),
                "approx_kl": torch.tensor(0.2),  # High KL
                "clip_frac": torch.tensor(0.0),
                "clip_frac_high": torch.tensor(0.0),
                "clip_frac_low": torch.tensor(0.0),
                "advantage_mean": torch.tensor(0.0),
                "advantage_std": torch.tensor(1.0),
                "reward_mean": torch.tensor(0.0),
                "reward_std": torch.tensor(1.0),
                "ratio_mean": torch.tensor(1.0),
                "ratio_std": torch.tensor(0.1),
                "ratio_max": torch.tensor(1.5),
                "ratio_min": torch.tensor(0.5),
            }

            all_metrics = algo.train_on_rollout(batch)

            # Should stop early due to high KL
            assert len(all_metrics) < 10

    def test_process_rollout_with_dynamic_sampling_success(self, simple_model):
        """Test dynamic sampling with successful collection."""
        config = DAPOConfig(
            group_size=4,
            min_batch_size=8,
            max_sampling_attempts=5,
        )
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        call_count = [0]

        def sample_fn():
            call_count[0] += 1
            samples = {
                "input_ids": torch.randint(0, 100, (8, 16)),
                "attention_mask": torch.ones(8, 16),
            }
            rewards = torch.randn(8)  # Has variance
            return samples, rewards

        batch = algo.process_rollout_with_dynamic_sampling(sample_fn, batch_size=8)

        assert batch is not None
        assert batch["input_ids"].shape[0] == 8

    def test_process_rollout_with_dynamic_sampling_failure(self, simple_model):
        """Test dynamic sampling when collection fails."""
        config = DAPOConfig(
            group_size=4,
            min_batch_size=16,
            max_sampling_attempts=2,
        )
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        def sample_fn():
            samples = {
                "input_ids": torch.randint(0, 100, (4, 16)),
                "attention_mask": torch.ones(4, 16),
            }
            # All same reward = zero variance
            rewards = torch.ones(4)
            return samples, rewards

        batch = algo.process_rollout_with_dynamic_sampling(sample_fn, batch_size=16)

        assert batch is None


# --- Factory Function Tests ---


class TestCreateDapo:
    """Tests for create_dapo factory function."""

    def test_create_with_defaults(self, simple_model):
        """Test factory with default parameters."""
        algo = create_dapo(simple_model)

        assert isinstance(algo, DAPOAlgorithm)
        assert algo.config.learning_rate == 1e-6
        assert algo.config.epsilon_low == 0.2
        assert algo.config.epsilon_high == 0.28

    def test_create_with_custom_params(self, simple_model):
        """Test factory with custom parameters."""
        algo = create_dapo(
            simple_model,
            learning_rate=1e-5,
            epsilon_low=0.15,
            epsilon_high=0.35,
            group_size=8,
            n_epochs=2,
        )

        assert algo.config.learning_rate == 1e-5
        assert algo.config.epsilon_low == 0.15
        assert algo.config.epsilon_high == 0.35
        assert algo.config.group_size == 8
        assert algo.config.n_epochs == 2

    def test_create_with_optimizer(self, simple_model):
        """Test factory with custom optimizer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        algo = create_dapo(simple_model, optimizer=optimizer)

        assert algo.optimizer is optimizer


# --- Integration Tests ---


class TestDAPOIntegration:
    """Integration tests for DAPO algorithm."""

    def test_full_training_loop(self, simple_model):
        """Test a complete training loop."""
        config = DAPOConfig(
            group_size=4,
            n_epochs=2,
            learning_rate=1e-4,
        )
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        # Multiple training iterations
        for i in range(3):
            batch = {
                "input_ids": torch.randint(0, 100, (8, 16)),
                "attention_mask": torch.ones(8, 16, dtype=torch.long),
                "labels": torch.randint(0, 100, (8, 16)),
                "rewards": torch.randn(8) + i,
            }

            metrics_list = algo.train_on_rollout(batch)

            assert len(metrics_list) > 0
            for metrics in metrics_list:
                assert "loss" in metrics
                assert not torch.isnan(torch.tensor(metrics["loss"]))

    def test_gradient_flow(self, simple_model):
        """Test that gradients flow correctly."""
        config = DAPOConfig(group_size=4)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        loss_dict = algo.compute_loss(batch)
        loss_dict["loss"].backward()

        # Check that gradients exist
        has_grad = False
        for p in simple_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad

    def test_numerical_stability(self, simple_model):
        """Test numerical stability with extreme values."""
        config = DAPOConfig(group_size=4)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.tensor([1e6, -1e6, 1e6, -1e6, 1e6, -1e6, 1e6, -1e6]),
        }

        loss_dict = algo.compute_loss(batch)

        assert not torch.isnan(loss_dict["loss"])
        assert not torch.isinf(loss_dict["loss"])

    def test_asymmetric_clipping_behavior(self, simple_model):
        """Test that asymmetric clipping works as expected."""
        config = DAPOConfig(
            group_size=4,
            epsilon_low=0.2,
            epsilon_high=0.3,  # Asymmetric
        )
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (8, 16)),
            "attention_mask": torch.ones(8, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (8, 16)),
            "rewards": torch.randn(8),
        }

        loss_dict = algo.compute_loss(batch)

        # Just verify it runs without error
        assert "clip_frac_high" in loss_dict
        assert "clip_frac_low" in loss_dict


# --- Edge Case Tests ---


class TestDAPOEdgeCases:
    """Edge case tests for DAPO algorithm."""

    def test_single_group(self, simple_model):
        """Test with minimum batch size (single group)."""
        config = DAPOConfig(group_size=4)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (4, 16)),
            "rewards": torch.randn(4),
        }

        loss_dict = algo.compute_loss(batch)
        assert "loss" in loss_dict

    def test_all_masked_tokens(self, simple_model):
        """Test with all tokens masked."""
        config = DAPOConfig(group_size=4)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "labels": torch.full((4, 16), -100),  # All masked
            "rewards": torch.randn(4),
        }

        # Should handle gracefully
        loss_dict = algo.compute_loss(batch)
        assert "loss" in loss_dict

    def test_zero_rewards(self, simple_model):
        """Test with all zero rewards."""
        config = DAPOConfig(group_size=4)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (4, 16)),
            "rewards": torch.zeros(4),
        }

        loss_dict = algo.compute_loss(batch)
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])

    def test_uniform_rewards_in_group(self, simple_model):
        """Test handling of uniform rewards within group."""
        config = DAPOConfig(group_size=4, advantage_eps=1e-8)
        algo = DAPOAlgorithm(policy_model=simple_model, config=config)

        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "labels": torch.randint(0, 100, (4, 16)),
            "rewards": torch.ones(4),  # All same
        }

        # Should handle via advantage_eps clamping
        loss_dict = algo.compute_loss(batch)
        assert not torch.isnan(loss_dict["loss"])
