"""
Test Suite for ThinkRL IPO Algorithm
====================================

Comprehensive tests for:
- IPOConfig validation
- IPOAlgorithm core functionality
- Loss computation (Standard, Hinge)
- Reference model handling
- Length normalization behavior
- Training step execution and gradient accumulation

Author: EllanorAI
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.ipo import (
    IPOAlgorithm,
    IPOConfig,
    create_ipo,
)


# --- Helper Classes ---


class SimplePolicy(nn.Module):
    """Simple policy model for testing."""

    def __init__(self, vocab_size=20, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        logits = self.head(embeds)
        return {"logits": logits}


# --- Fixtures ---


@pytest.fixture
def policy_model():
    return SimplePolicy()


@pytest.fixture
def ref_model():
    return SimplePolicy()


@pytest.fixture
def ipo_config():
    return IPOConfig(
        learning_rate=1e-5,
        beta=0.1,
        tau=0.05,
        loss_type="ipo",
    )


@pytest.fixture
def ipo_algo(policy_model, ref_model, ipo_config):
    return IPOAlgorithm(policy_model=policy_model, ref_model=ref_model, config=ipo_config)


@pytest.fixture
def preference_batch():
    """Create a sample preference batch."""
    batch_size = 4
    seq_len = 8
    vocab_size = 20

    return {
        "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
    }


# --- IPOConfig Tests ---


def test_ipo_config_defaults():
    config = IPOConfig()
    assert config.learning_rate == 1e-6
    assert config.beta == 0.1
    assert config.tau == 0.05
    assert config.loss_type == "ipo"
    assert config.length_normalization is False
    assert config.gradient_accumulation_steps == 1


# --- IPOAlgorithm Tests ---


def test_initialization_freezes_ref_model(policy_model, ref_model):
    """Test that reference model parameters are frozen."""
    # Ensure ref model initially has grad
    for p in ref_model.parameters():
        p.requires_grad = True

    _ = IPOAlgorithm(policy_model, ref_model)

    # Check that it's now frozen
    for p in ref_model.parameters():
        assert not p.requires_grad
    assert not ref_model.training


def test_initialization_missing_ref_model(policy_model):
    """Test error raised when ref_model is missing."""
    with pytest.raises(ValueError, match="requires a reference model"):
        IPOAlgorithm(policy_model, ref_model=None)  # type: ignore


def test_get_batch_log_probs_standard(ipo_algo):
    """Test standard summation of log probabilities (no length norm)."""
    ipo_algo.config.length_normalization = False

    batch_size = 2
    seq_len = 5
    vocab_size = 20

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mask last token
    labels[:, -1] = -100

    log_probs = ipo_algo.get_batch_log_probs(logits, labels)

    assert log_probs.shape == (batch_size,)
    # Verify it's not a mean (magnitude should be roughly seq_len * entropy)
    # Just check it's not NaN
    assert not torch.isnan(log_probs).any()


def test_get_batch_log_probs_length_norm(ipo_algo):
    """Test length normalized log probabilities."""
    ipo_algo.config.length_normalization = True

    batch_size = 1
    seq_len = 5
    vocab_size = 20

    logits = torch.zeros(batch_size, seq_len, vocab_size)  # Uniform probs
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # All valid tokens
    labels[0, :] = 1

    # Log prob of uniform is -log(vocab_size)
    expected_token_log_prob = -torch.log(torch.tensor(float(vocab_size)))

    # Case 1: Full length (5 tokens)
    log_probs_full = ipo_algo.get_batch_log_probs(logits, labels)
    # With norm, should equal the per-token probability
    assert torch.isclose(log_probs_full, expected_token_log_prob, atol=1e-5)

    # Case 2: Masked length (3 tokens valid)
    labels[0, 3:] = -100
    log_probs_masked = ipo_algo.get_batch_log_probs(logits, labels)

    # With norm, should STILL equal the per-token probability (average)
    assert torch.isclose(log_probs_masked, expected_token_log_prob, atol=1e-5)


def test_compute_loss_ipo_structure(ipo_algo, preference_batch):
    """Test IPO loss structure and metrics."""
    ipo_algo.config.loss_type = "ipo"
    loss_dict = ipo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert "rewards/chosen" in loss_dict
    assert "rewards/rejected" in loss_dict
    assert "log_probs/kl_approx" in loss_dict

    assert loss_dict["loss"].ndim == 0
    assert not torch.isnan(loss_dict["loss"])


def test_compute_loss_hinge_structure(ipo_algo, preference_batch):
    """Test Hinge IPO loss structure."""
    ipo_algo.config.loss_type = "ipo_hinge"
    loss_dict = ipo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"])


def test_compute_log_ratios(ipo_algo, preference_batch):
    """Test that log ratios are computed correctly."""
    chosen_ratios, rejected_ratios = ipo_algo.compute_log_ratios(preference_batch)

    batch_size = preference_batch["chosen_input_ids"].shape[0]
    assert chosen_ratios.shape == (batch_size,)
    assert rejected_ratios.shape == (batch_size,)


def test_training_step(ipo_algo, preference_batch):
    """Test single training step updates parameters."""
    initial_params = [p.clone() for p in ipo_algo.policy_model.parameters()]

    metrics = ipo_algo.training_step(preference_batch)

    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert isinstance(metrics["loss"], float)

    # Check that parameters were updated
    updated = False
    for p_init, p_new in zip(initial_params, ipo_algo.policy_model.parameters()):
        if not torch.allclose(p_init, p_new):
            updated = True
            break
    assert updated


def test_gradient_accumulation(ipo_algo, preference_batch):
    """Test explicit gradient accumulation logic."""
    ipo_algo.config.gradient_accumulation_steps = 2

    # Mock optimizer to track steps
    optimizer_mock = MagicMock(spec=torch.optim.Optimizer)
    ipo_algo.optimizer = optimizer_mock

    # Step 1: Should calculate loss but NOT step optimizer
    ipo_algo.training_step(preference_batch)
    assert ipo_algo.accum_steps == 1
    optimizer_mock.step.assert_not_called()
    optimizer_mock.zero_grad.assert_not_called()

    # Step 2: Should step optimizer and reset counter
    ipo_algo.training_step(preference_batch)
    assert ipo_algo.accum_steps == 0
    optimizer_mock.step.assert_called_once()
    optimizer_mock.zero_grad.assert_called_once()


def test_create_ipo_factory(policy_model, ref_model):
    """Test factory function arguments."""
    ipo = create_ipo(policy_model, ref_model, learning_rate=3e-4, beta=0.1, tau=0.1, length_normalization=True)

    assert isinstance(ipo, IPOAlgorithm)
    assert ipo.config.learning_rate == 3e-4
    assert ipo.config.beta == 0.1
    assert ipo.config.tau == 0.1
    assert ipo.config.length_normalization is True
    assert ipo.ref_model is ref_model

    # Check if ref model is frozen
    assert not next(ipo.ref_model.parameters()).requires_grad
