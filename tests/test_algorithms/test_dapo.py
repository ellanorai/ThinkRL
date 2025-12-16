from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Assuming the DAPO implementation is in thinkrl.algorithms.dapo
from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig, DynamicSamplingBuffer


# --- Fixtures ---
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
def dapo_config():
    return DAPOConfig(group_size=4, min_batch_size=8, epsilon_low=0.2, epsilon_high=0.2, max_len=20, cache_len=10)


@pytest.fixture
def dapo_algo(policy_model, dapo_config):
    return DAPOAlgorithm(policy_model, config=dapo_config)


# --- Tests for Configuration ---


def test_dapo_config_defaults():
    config = DAPOConfig()
    assert config.learning_rate == 1e-6
    assert config.group_size == 16
    assert config.dynamic_sampling is True


def test_dapo_config_validation():
    with pytest.raises(AssertionError):
        DAPOConfig(epsilon_low=-0.1)

    with pytest.raises(AssertionError):
        DAPOConfig(group_size=1)  # Must be >= 2


# --- Tests for DynamicSamplingBuffer ---


def test_buffer_add_samples_filtering(dapo_config):
    buffer = DynamicSamplingBuffer(dapo_config)
    group_size = dapo_config.group_size  # 4

    # Create mock data for 2 groups (8 samples total)
    # Group 1: Rewards [1, 1, 1, 1] -> Variance 0 -> Should be filtered
    # Group 2: Rewards [1, 2, 3, 4] -> Variance > 0 -> Should be kept

    samples = {"input_ids": torch.randn(8, 5), "labels": torch.randn(8, 5)}
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0])

    valid_added = buffer.add_samples(samples, rewards)

    # Only group 2 should be added (4 samples)
    # FIX: Use group_size variable in assertion
    assert valid_added == group_size
    assert buffer.total_filtered == 1  # 1 group filtered
    assert len(buffer.buffer) == 1  # 1 group in buffer


def test_buffer_get_batch(dapo_config):
    buffer = DynamicSamplingBuffer(dapo_config)
    # Target batch size 8 (2 groups)

    # Add 3 valid groups
    samples = {"input_ids": torch.zeros(12, 5)}
    rewards = torch.tensor(
        [
            0.0,
            1.0,
            0.0,
            1.0,  # Group 1 (Valid)
            0.0,
            1.0,
            0.0,
            1.0,  # Group 2 (Valid)
            0.0,
            1.0,
            0.0,
            1.0,  # Group 3 (Valid)
        ]
    )

    buffer.add_samples(samples, rewards)

    # Request batch of 8
    batch = buffer.get_batch(8)

    assert batch is not None
    assert batch["input_ids"].shape[0] == 8
    assert batch["rewards"].shape[0] == 8
    # Buffer should have 1 group left
    assert len(buffer.buffer) == 1


def test_buffer_insufficient_samples(dapo_config):
    buffer = DynamicSamplingBuffer(dapo_config)

    # Add 1 valid group (4 samples)
    samples = {"input_ids": torch.zeros(4, 5)}
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    buffer.add_samples(samples, rewards)

    # Request batch of 8
    batch = buffer.get_batch(8)
    assert batch is None


# --- Tests for DAPO Algorithm Logic ---


def test_compute_advantages(dapo_algo):
    # Config group size is 4
    # Rewards: [10, 20, 30, 40]
    # Mean: 25, Std (unbiased): 12.91
    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])

    adv = dapo_algo.compute_advantages(rewards)

    assert adv.shape == (4,)
    # Sum of standardized values should be close to 0
    assert torch.isclose(adv.sum(), torch.tensor(0.0), atol=1e-5)
    # Variance should be close to 1
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-1)

    # Manual check
    mean = 25.0
    std = torch.tensor([10.0, 20.0, 30.0, 40.0]).std(unbiased=False)
    expected = (rewards - mean) / (std + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-5)


def test_overlong_penalty(dapo_algo):
    cfg = dapo_algo.config
    # max_len=20, cache_len=10 -> soft_start=10

    # Case 1: Short sequence (len 5) -> 0 penalty
    seq_short = torch.tensor([5])
    pen_short = dapo_algo.compute_overlong_penalty(seq_short)
    assert pen_short.item() == 0.0

    # Case 2: Buffer zone (len 15) -> linear penalty
    # FIX: Use cfg to derive the test input
    mid_len = cfg.cache_len + 5  # 10 + 5 = 15
    seq_mid = torch.tensor([mid_len])

    # Penalty = (10 - 15) / 10 = -0.5
    pen_mid = dapo_algo.compute_overlong_penalty(seq_mid)
    assert pen_mid.item() == -0.5

    # Case 3: Over max (len 25) -> -1.0 penalty
    # FIX: Use cfg to derive the test input
    long_len = cfg.max_len + 5
    seq_long = torch.tensor([long_len])
    pen_long = dapo_algo.compute_overlong_penalty(seq_long)
    assert pen_long.item() == -1.0


def test_compute_loss_structure(dapo_algo):
    """Test that compute_loss returns the expected dictionary structure and runs backward."""
    batch_size = 4
    seq_len = 5
    vocab_size = 10

    # Mock Batch
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rewards": torch.randn(batch_size),
        # Ensure old_log_probs is present as expected by the fixed DAPO implementation
        "old_log_probs": torch.randn(batch_size, seq_len),
    }

    # Run loss computation
    loss_dict = dapo_algo.compute_loss(batch)

    assert "loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "entropy_loss" in loss_dict
    assert "clip_frac" in loss_dict

    # Check gradients
    loss = loss_dict["loss"]
    dapo_algo.optimizer.zero_grad()
    loss.backward()

    # Ensure gradients are computed for policy model
    for param in dapo_algo.policy_model.parameters():
        assert param.grad is not None


def test_train_on_rollout_loop(dapo_algo):
    """Test the multi-epoch training loop."""
    dapo_algo.config.n_epochs = 2

    batch = {
        "input_ids": torch.randint(0, 10, (4, 5)),
        "attention_mask": torch.ones((4, 5)),
        "labels": torch.randint(0, 10, (4, 5)),
        "rewards": torch.randn(4),
    }

    # Mock compute_rollout_log_probs to avoid needing a forward pass inside the loop preparation
    with patch.object(dapo_algo, "compute_rollout_log_probs") as mock_old_log:
        mock_old_log.return_value = torch.zeros(4, 5)

        # Mock training_step to track calls
        with patch.object(dapo_algo, "training_step") as mock_step:
            # FIX: Use side_effect to return a NEW dict each time.
            # If we use return_value, the same dict object is modified in the loop (overwriting 'epoch'),
            # causing assertions on stored metrics to fail.
            mock_step.side_effect = lambda batch, old_log_probs: {"loss": 0.5, "approx_kl": 0.01}

            metrics = dapo_algo.train_on_rollout(batch)

            # Should call compute_old_log_probs once
            mock_old_log.assert_called_once()

            # Should call training_step twice (n_epochs=2)
            assert mock_step.call_count == 2
            assert len(metrics) == 2

            # Verify epochs are correctly recorded
            assert metrics[0]["epoch"] == 0
            assert metrics[1]["epoch"] == 1


def test_process_rollout_dynamic_sampling(dapo_algo):
    """Test the dynamic sampling retry loop."""
    batch_size = 8

    # Mock sample_fn to return data
    # We need it to run at least twice to fill buffer if we return small chunks

    mock_samples = {"input_ids": torch.zeros(4, 5)}  # 1 group of 4
    mock_rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Valid variance

    mock_sample_fn = MagicMock(return_value=(mock_samples, mock_rewards))

    # It takes 2 calls to get 8 samples (4 per call)
    batch = dapo_algo.process_rollout_with_dynamic_sampling(mock_sample_fn, batch_size)

    assert batch is not None
    assert mock_sample_fn.call_count == 2
    assert batch["input_ids"].shape[0] == 8
