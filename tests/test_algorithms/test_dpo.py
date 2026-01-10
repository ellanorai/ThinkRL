"""
Test Suite for ThinkRL DPO Algorithm
====================================

Comprehensive tests for:
- DPOConfig validation
- DPOAlgorithm core functionality
- Loss computation (Sigmoid, Hinge, IPO)
- Reference model handling
- Training step execution

Author: EllanorAI
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.dpo import (
    DPOAlgorithm,
    DPOConfig,
    create_dpo,
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
def dpo_config():
    return DPOConfig(
        learning_rate=1e-5,
        beta=0.1,
        loss_type="sigmoid",
        # batch_size removed as it is not part of DPOConfig definition
    )


@pytest.fixture
def dpo_algo(policy_model, ref_model, dpo_config):
    return DPOAlgorithm(policy_model=policy_model, ref_model=ref_model, config=dpo_config)


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


# --- DPOConfig Tests ---


def test_dpo_config_defaults():
    config = DPOConfig()
    assert config.learning_rate == 1e-6
    assert config.beta == 0.1
    assert config.loss_type == "sigmoid"
    assert config.label_smoothing == 0.0


def test_dpo_config_validation():
    with pytest.raises(AssertionError, match="Beta.*must be positive"):
        DPOConfig(beta=-0.1)

    with pytest.raises(AssertionError):
        DPOConfig(loss_type="invalid_type")


# --- DPOAlgorithm Tests ---


def test_initialization_freezes_ref_model(policy_model, ref_model):
    """Test that reference model parameters are frozen."""
    # Ensure ref model initially has grad
    for p in ref_model.parameters():
        p.requires_grad = True

    _ = DPOAlgorithm(policy_model, ref_model)

    # Check that it's now frozen
    for p in ref_model.parameters():
        assert not p.requires_grad
    assert not ref_model.training


def test_initialization_missing_ref_model(policy_model):
    """Test error raised when ref_model is missing."""
    # We explicitly check for this in the DPO implementation
    with pytest.raises(ValueError, match="requires a reference model"):
        DPOAlgorithm(policy_model, ref_model=None)  # type: ignore


def test_get_batch_log_probs(dpo_algo):
    """Test computation of sequence log probabilities."""
    batch_size = 2
    seq_len = 5
    vocab_size = 20

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mask some labels (simulate padding or non-response tokens)
    labels[:, -1] = -100

    log_probs = dpo_algo.get_batch_log_probs(logits, labels)

    assert log_probs.shape == (batch_size,)
    # Should be sum of log probs, checking it's scalar per sequence
    assert not torch.isnan(log_probs).any()


def test_compute_loss_sigmoid(dpo_algo, preference_batch):
    """Test Sigmoid loss computation."""
    dpo_algo.config.loss_type = "sigmoid"
    loss_dict = dpo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert "chosen_reward" in loss_dict
    assert "rejected_reward" in loss_dict
    assert "dpo_accuracy" in loss_dict
    assert "reward_margin" in loss_dict

    # Check shape/type
    assert loss_dict["loss"].ndim == 0
    assert not torch.isnan(loss_dict["loss"])


def test_compute_loss_hinge(dpo_algo, preference_batch):
    """Test Hinge loss computation."""
    dpo_algo.config.loss_type = "hinge"
    loss_dict = dpo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"])


def test_compute_loss_ipo(dpo_algo, preference_batch):
    """Test IPO loss computation."""
    dpo_algo.config.loss_type = "ipo"
    loss_dict = dpo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"])


def test_compute_loss_label_smoothing(dpo_algo, preference_batch):
    """Test loss computation with label smoothing."""
    dpo_algo.config.label_smoothing = 0.1
    loss_dict = dpo_algo.compute_loss(preference_batch)

    assert "loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"])


def test_training_step(dpo_algo, preference_batch):
    """Test single training step execution."""
    initial_params = [p.clone() for p in dpo_algo.policy_model.parameters()]

    metrics = dpo_algo.training_step(preference_batch)

    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert isinstance(metrics["loss"], float)

    # Check that parameters were updated
    updated = False
    for p_init, p_new in zip(initial_params, dpo_algo.policy_model.parameters()):
        if not torch.allclose(p_init, p_new):
            updated = True
            break
    assert updated


def test_create_dpo_factory(policy_model, ref_model):
    """Test factory function."""
    dpo = create_dpo(policy_model, ref_model, learning_rate=3e-4, beta=0.05)

    assert isinstance(dpo, DPOAlgorithm)
    assert dpo.config.learning_rate == 3e-4
    assert dpo.config.beta == 0.05
    assert dpo.ref_model is ref_model

    # Explicitly check for None to satisfy type checkers (Pylance)
    assert dpo.ref_model is not None
    # Check if ref model is frozen
    assert not next(dpo.ref_model.parameters()).requires_grad


def test_gradient_accumulation_config_pass():
    """Test that gradient accumulation config parameter is handled."""
    config = DPOConfig(gradient_accumulation_steps=4)
    assert config.gradient_accumulation_steps == 4
