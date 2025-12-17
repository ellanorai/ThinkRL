from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig


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
def ref_model():
    return SimplePolicy()


@pytest.fixture
def grpo_config():
    return GRPOConfig(group_size=4, learning_rate=1e-5, beta=0.04)


@pytest.fixture
def grpo_algo(policy_model, ref_model, grpo_config):
    return GRPOAlgorithm(policy_model, ref_model=ref_model, config=grpo_config)


# --- Tests for Configuration ---


def test_grpo_config_defaults():
    config = GRPOConfig()
    assert config.learning_rate == 1e-6
    assert config.group_size == 64
    assert config.beta == 0.04


def test_grpo_config_validation():
    with pytest.raises(AssertionError):
        GRPOConfig(group_size=1)  # Must be > 1
    with pytest.raises(AssertionError):
        GRPOConfig(beta=-0.1)  # Must be non-negative


# --- Tests for GRPO Algorithm Logic ---


def test_compute_advantages(grpo_algo):
    # Config group size is 4
    # Rewards: [10, 20, 30, 40]
    # Mean: 25.0
    # Std (Population/Unbiased=False): sqrt(((15^2 + 5^2 + 5^2 + 15^2) / 4)) = sqrt(125) approx 11.1803
    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])

    adv = grpo_algo.compute_advantages(rewards)

    assert adv.shape == (4,)
    # Sum of standardized values should be close to 0
    assert torch.isclose(adv.sum(), torch.tensor(0.0), atol=1e-5)

    # Manual check
    mean = 25.0
    std = torch.tensor([10.0, 20.0, 30.0, 40.0]).std(unbiased=False)
    expected = (rewards - mean) / (std + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-5)


def test_compute_advantages_batch_mismatch(grpo_algo):
    # Batch size 6 is not divisible by group size 4
    rewards = torch.randn(6)
    with pytest.raises(ValueError, match="divisible by group_size"):
        grpo_algo.compute_advantages(rewards)


def test_compute_loss_structure(grpo_algo):
    """Test that compute_loss returns the expected dictionary structure and runs backward."""
    batch_size = 4
    seq_len = 5
    vocab_size = 10

    # Ensure model parameters require grad
    for param in grpo_algo.policy_model.parameters():
        assert param.requires_grad

    # Mock Batch - Use Long tensors for IDs
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "rewards": torch.randn(batch_size),
        "old_log_probs": torch.randn(batch_size, seq_len),
    }

    # Run loss computation
    loss_dict = grpo_algo.compute_loss(batch)

    assert "loss" in loss_dict
    assert "kl_mean" in loss_dict
    assert "advantage_mean" in loss_dict
    assert "clip_fraction" in loss_dict

    # Check gradients
    loss = loss_dict["loss"]
    assert loss.requires_grad, "Loss should require gradients. Check if model outputs are detached."

    grpo_algo.optimizer.zero_grad()
    loss.backward()

    # Ensure gradients are computed for policy model
    for param in grpo_algo.policy_model.parameters():
        assert param.grad is not None


def test_kl_penalty_computation(grpo_algo):
    """Verify that KL penalty is actually affecting the loss."""
    # Set high beta so loss is dominated by KL
    grpo_algo.config.beta = 100.0

    batch_size = 4
    seq_len = 5
    vocab_size = 10

    # Zero rewards -> Advantages = 0 -> Surrogate = 0
    # Loss should be purely beta * KL
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
        "rewards": torch.zeros(batch_size),
        # Log ratio 0 => ratio 1
        "old_log_probs": torch.zeros(batch_size, seq_len),
    }

    loss_dict = grpo_algo.compute_loss(batch)

    # KL should be positive (models initialized randomly)
    kl_val = loss_dict["kl_mean"].item()
    assert kl_val >= 0.0, "KL Divergence should be non-negative"

    # Loss = -surrogate + beta * KL.
    # With 0 rewards, A=0, so surrogate=0.
    # Loss approx beta * KL
    expected_loss = grpo_algo.config.beta * loss_dict["kl_mean"]

    assert torch.isclose(loss_dict["loss"], expected_loss, atol=1e-4)


def test_train_on_rollout_loop(grpo_algo):
    """Test the multi-epoch training loop."""
    grpo_algo.config.n_epochs = 2

    batch = {
        "input_ids": torch.randint(0, 10, (4, 5), dtype=torch.long),
        "attention_mask": torch.ones((4, 5)),
        "labels": torch.randint(0, 10, (4, 5), dtype=torch.long),
        "rewards": torch.randn(4),
    }

    # Mock compute_rollout_log_probs
    with patch.object(grpo_algo, "compute_rollout_log_probs") as mock_old_log:
        mock_old_log.return_value = torch.zeros(4, 5)

        # Mock training_step
        with patch.object(grpo_algo, "training_step") as mock_step:
            # Return new dict to avoid mutation issues
            mock_step.side_effect = lambda batch, old_log_probs: {"loss": 0.5}

            metrics = grpo_algo.train_on_rollout(batch)

            mock_old_log.assert_called_once()
            assert mock_step.call_count == 2
            assert len(metrics) == 2
            assert metrics[0]["epoch"] == 0
            assert metrics[1]["epoch"] == 1


def test_no_ref_model_warning():
    """Test that initializing without ref_model but beta > 0 logs a warning."""
    policy = SimplePolicy()
    config = GRPOConfig(beta=0.1)

    with patch("thinkrl.algorithms.grpo.logger") as mock_logger:
        GRPOAlgorithm(policy, ref_model=None, config=config)
        mock_logger.warning.assert_called_with(
            "GRPO initialized without ref_model but beta > 0. KL penalty will be 0."
        )
