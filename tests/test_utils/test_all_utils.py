"""
Test Suite for ThinkRL Utils
=============================

Comprehensive tests for all utility modules:
- logging
- metrics
- checkpoint
- data
- tokenizers

Author: Test Suite for ThinkRL
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np

# Import all utils modules
from thinkrl.utils.logging import (
    setup_logger,
    get_logger,
    configure_logging_for_distributed,
    ColoredFormatter,
)

from thinkrl.utils.metrics import (
    MetricsTracker,
    compute_reward,
    compute_kl_divergence,
    compute_advantages,
    compute_returns,
    compute_policy_entropy,
    compute_accuracy,
    compute_perplexity,
    compute_clip_fraction,
    compute_explained_variance,
    aggregate_metrics,
    compute_statistical_metrics,
)

from thinkrl.utils.checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    save_config,
    load_config,
)

from thinkrl.utils.data import (
    BatchEncoding,
    pad_sequences,
    create_attention_mask,
    create_position_ids,
    create_causal_mask,
    collate_batch,
    preprocess_text,
    truncate_sequence,
    create_labels_for_clm,
    compute_sequence_lengths,
    to_device,
    prepare_batch_for_training,
)

# Tokenizer tests are optional (requires transformers)
try:
    from thinkrl.utils.tokenizers import (
        get_tokenizer,
        tokenize_text,
        decode_tokens,
        get_special_tokens,
        count_tokens,
    )

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel()


@pytest.fixture
def sample_batch():
    """Create sample batch data."""
    return {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
        "labels": torch.randint(0, 100, (4, 10)),
    }


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Test logging utilities."""

    def test_setup_logger_basic(self, temp_dir):
        """Test basic logger setup."""
        logger = setup_logger(name="test_logger", level=logging.INFO, log_dir=temp_dir)

        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")

        # Check log file was created
        log_files = list(temp_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("thinkrl.test")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s", use_colors=True)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_distributed_logging(self, temp_dir):
        """Test distributed logging configuration."""
        logger = configure_logging_for_distributed(
            rank=0, world_size=2, log_dir=temp_dir
        )

        assert logger is not None
        logger.info("Rank 0 message")


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Test metrics utilities."""

    def test_metrics_tracker(self):
        """Test MetricsTracker class."""
        tracker = MetricsTracker()

        # Update metrics
        tracker.update("loss", 0.5)
        tracker.update("loss", 0.4)
        tracker.update("accuracy", 0.95)

        # Get current values
        current = tracker.get_current()
        assert current["loss"] == 0.4
        assert current["accuracy"] == 0.95

        # Get averages
        avg = tracker.get_average()
        assert avg["loss"] == 0.45  # (0.5 + 0.4) / 2

        # Reset
        tracker.reset("loss")
        assert "loss" not in tracker.get_current()

    def test_compute_reward(self):
        """Test reward computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Without normalization
        result = compute_reward(rewards, normalize=False)
        assert torch.allclose(result, rewards)

        # With normalization
        normalized = compute_reward(rewards, normalize=True)
        assert torch.abs(normalized.mean()) < 1e-6
        assert torch.abs(normalized.std() - 1.0) < 1e-6

    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        log_probs_policy = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        log_probs_ref = torch.log(torch.tensor([0.1, 0.4, 0.5]))

        kl_div = compute_kl_divergence(log_probs_policy, log_probs_ref)
        assert isinstance(kl_div, torch.Tensor)
        assert kl_div.dim() == 0  # Scalar

    def test_compute_advantages(self):
        """Test GAE advantage computation."""
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)

        advantages = compute_advantages(
            rewards=rewards, values=values, gamma=0.99, lambda_=0.95
        )

        assert advantages.shape == rewards.shape
        # Check normalization
        assert torch.abs(advantages.mean()) < 0.1

    def test_compute_returns(self):
        """Test returns computation."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        returns = compute_returns(rewards, gamma=0.9)

        assert returns.shape == rewards.shape
        # First return should be highest (includes all future rewards)
        assert returns[0, 0] > rewards[0, 0]

    def test_compute_policy_entropy(self):
        """Test policy entropy computation."""
        logits = torch.randn(4, 10, 100)  # batch, seq, vocab
        entropy = compute_policy_entropy(logits)

        assert isinstance(entropy, torch.Tensor)
        assert entropy >= 0  # Entropy is non-negative

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        predictions = torch.tensor([[1, 2, 3], [1, 1, 1]])
        targets = torch.tensor([[1, 2, 0], [1, 1, 1]])

        accuracy = compute_accuracy(predictions, targets)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 5 / 6  # 5 correct out of 6

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        loss = 2.5
        ppl = compute_perplexity(loss)

        assert ppl > 0
        assert np.abs(ppl - np.exp(2.5)) < 1e-6

    def test_compute_clip_fraction(self):
        """Test clip fraction computation."""
        ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
        clip_frac = compute_clip_fraction(ratio, epsilon=0.2)

        assert 0.0 <= clip_frac <= 1.0
        # Ratios 0.5 and 2.0 are outside [0.8, 1.2]
        assert clip_frac == 0.5  # 2/4

    def test_compute_explained_variance(self):
        """Test explained variance computation."""
        predictions = torch.randn(100)
        targets = torch.randn(100)

        ev = compute_explained_variance(predictions, targets)
        assert ev <= 1.0

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        metrics_list = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.6, "accuracy": 0.85},
            {"loss": 0.4, "accuracy": 0.95},
        ]

        avg_metrics = aggregate_metrics(metrics_list)
        assert "loss" in avg_metrics
        assert "accuracy" in avg_metrics
        assert avg_metrics["loss"] == 0.5  # (0.5 + 0.6 + 0.4) / 3

    def test_compute_statistical_metrics(self):
        """Test statistical metrics computation."""
        values = torch.randn(1000)
        stats = compute_statistical_metrics(values)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats


# ============================================================================
# Checkpoint Tests
# ============================================================================


class TestCheckpoint:
    """Test checkpoint utilities."""

    def test_save_and_load_checkpoint(self, temp_dir, simple_model):
        """Test basic checkpoint save/load."""
        checkpoint_path = temp_dir / "checkpoint.pt"

        # Save checkpoint
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=simple_model,
            epoch=10,
            metrics={"loss": 0.42},
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        new_model = type(simple_model)()
        checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, model=new_model)

        assert checkpoint["epoch"] == 10
        assert checkpoint["metrics"]["loss"] == 0.42

        # Check model weights are same
        for p1, p2 in zip(simple_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_manager(self, temp_dir, simple_model):
        """Test CheckpointManager class."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir, max_checkpoints=3, metric_name="loss", mode="min"
        )

        # Save multiple checkpoints
        for i in range(5):
            manager.save_checkpoint(
                model=simple_model,
                epoch=i,
                step=i * 100,
                metrics={"loss": 1.0 / (i + 1)},  # Decreasing loss
            )

        # Should keep only 3 best
        assert len(manager.checkpoints) <= 3

        # Best checkpoint should have lowest loss
        assert manager.best_checkpoint is not None
        assert manager.best_checkpoint["metrics"]["loss"] == 0.2  # 1/5

        # Load best checkpoint
        new_model = type(simple_model)()
        metadata = manager.load_best_checkpoint(new_model)
        assert metadata is not None

    def test_save_and_load_config(self, temp_dir):
        """Test config save/load."""
        config = {"model": "gpt2", "learning_rate": 1e-4, "batch_size": 32}

        config_path = temp_dir / "config.json"
        save_config(config, config_path)

        assert config_path.exists()

        loaded_config = load_config(config_path)
        assert loaded_config == config


# ============================================================================
# Data Utilities Tests
# ============================================================================


class TestDataUtils:
    """Test data utilities."""

    def test_pad_sequences(self):
        """Test sequence padding."""
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3, 4, 5]),
        ]

        padded = pad_sequences(sequences, padding_value=0)

        assert padded.shape == (3, 5)  # 3 sequences, max length 5
        assert padded[1, 2] == 0  # Padding
        assert padded[0, 0] == 1  # Original value

    def test_create_attention_mask(self):
        """Test attention mask creation."""
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_attention_mask(input_ids, padding_value=0)

        expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        assert torch.all(mask == expected)

    def test_create_position_ids(self):
        """Test position IDs creation."""
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        position_ids = create_position_ids(attention_mask)

        assert position_ids.shape == attention_mask.shape
        assert position_ids[0, 0] == 0
        assert position_ids[0, 1] == 1
        assert position_ids[0, 3] == 0  # Padding position

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        mask = create_causal_mask(4)

        assert mask.shape == (4, 4)
        assert mask[0, 0] == True  # Can attend to self
        assert mask[0, 1] == False  # Cannot attend to future
        assert mask[3, 0] == True  # Can attend to past

    def test_collate_batch(self):
        """Test batch collation."""
        batch = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [1, 2], "labels": [2, 3]},
        ]

        collated = collate_batch(batch, padding_value=0)

        assert "input_ids" in collated
        assert "labels" in collated
        assert collated["input_ids"].shape == (2, 3)  # 2 samples, max len 3

    def test_batch_encoding(self):
        """Test BatchEncoding class."""
        encoding = BatchEncoding(
            input_ids=torch.randint(0, 100, (4, 10)),
            attention_mask=torch.ones(4, 10),
            labels=torch.randint(0, 100, (4, 10)),
        )

        # Test dictionary-like access
        assert encoding["input_ids"] is not None
        assert "labels" in encoding.keys()

        # Test device movement
        if torch.cuda.is_available():
            encoding_cuda = encoding.to("cuda")
            assert encoding_cuda.input_ids.is_cuda

    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  Hello   World!  "
        cleaned = preprocess_text(text, lowercase=True, strip=True)

        assert cleaned == "hello world!"

    def test_truncate_sequence(self):
        """Test sequence truncation."""
        seq = [1, 2, 3, 4, 5]

        # Right truncation
        truncated_right = truncate_sequence(seq, max_length=3, truncation_side="right")
        assert truncated_right == [1, 2, 3]

        # Left truncation
        truncated_left = truncate_sequence(seq, max_length=3, truncation_side="left")
        assert truncated_left == [3, 4, 5]

    def test_create_labels_for_clm(self):
        """Test CLM labels creation."""
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
        labels = create_labels_for_clm(input_ids, ignore_index=-100)

        # Labels should be shifted input_ids
        assert labels[0, 0] == 2
        assert labels[0, 1] == 3
        assert labels[0, -1] == -100  # Last position
        assert labels[0, -2] == -100  # Padding masked

    def test_compute_sequence_lengths(self):
        """Test sequence length computation."""
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        lengths = compute_sequence_lengths(input_ids, padding_value=0)

        assert lengths[0] == 3
        assert lengths[1] == 2

    def test_to_device(self):
        """Test device movement."""
        batch = {"input_ids": torch.randn(4, 128), "labels": torch.randn(4)}

        # Move to CPU (always available)
        batch_cpu = to_device(batch, "cpu")
        assert batch_cpu["input_ids"].device.type == "cpu"

    def test_prepare_batch_for_training(self):
        """Test batch preparation."""
        batch = {"input_ids": [[1, 2, 3, 4]]}
        prepared = prepare_batch_for_training(batch, device="cpu", create_labels=True)

        assert "input_ids" in prepared
        assert "attention_mask" in prepared
        assert "labels" in prepared
        assert isinstance(prepared["input_ids"], torch.Tensor)


# ============================================================================
# Tokenizer Tests (Optional)
# ============================================================================


@pytest.mark.skipif(not TOKENIZERS_AVAILABLE, reason="transformers not installed")
class TestTokenizers:
    """Test tokenizer utilities (requires transformers)."""

    @pytest.fixture
    def tokenizer(self):
        """Get a test tokenizer."""
        return get_tokenizer("gpt2")

    def test_get_tokenizer(self, tokenizer):
        """Test tokenizer loading."""
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_tokenize_text(self, tokenizer):
        """Test text tokenization."""
        text = "Hello, world!"
        encoded = tokenize_text(
            text, tokenizer, max_length=512, padding="max_length", return_tensors="pt"
        )

        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert encoded["input_ids"].shape[1] == 512

    def test_decode_tokens(self, tokenizer):
        """Test token decoding."""
        token_ids = [15496, 11, 995, 0]
        text = decode_tokens(token_ids, tokenizer)

        assert isinstance(text, str)
        assert len(text) > 0

    def test_get_special_tokens(self, tokenizer):
        """Test special tokens extraction."""
        special_tokens = get_special_tokens(tokenizer)

        assert "pad_token" in special_tokens
        assert "eos_token" in special_tokens
        assert special_tokens["pad_token_id"] is not None

    def test_count_tokens(self, tokenizer):
        """Test token counting."""
        text = "Hello, world!"
        count = count_tokens(text, tokenizer)

        assert isinstance(count, int)
        assert count > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple utils."""

    def test_training_workflow(self, temp_dir, simple_model, sample_batch):
        """Test a typical training workflow using multiple utils."""
        # Setup logger
        logger = setup_logger("training", log_dir=temp_dir)
        logger.info("Starting training")

        # Setup checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints", max_checkpoints=3
        )

        # Setup metrics tracker
        tracker = MetricsTracker()

        # Simulate training loop
        for epoch in range(3):
            # Prepare batch
            batch = prepare_batch_for_training(
                sample_batch, device="cpu", create_labels=True
            )

            # Simulate forward pass
            outputs = simple_model(batch["input_ids"].float())
            loss = outputs.mean()

            # Track metrics
            tracker.update("loss", loss.item())

            # Save checkpoint
            manager.save_checkpoint(
                model=simple_model, epoch=epoch, metrics={"loss": loss.item()}
            )

            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")

        # Verify everything worked
        assert manager.best_checkpoint is not None
        assert tracker.get_average("loss") is not None
        assert len(list(temp_dir.glob("*.log"))) > 0


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance-critical operations."""

    def test_batch_padding_performance(self):
        """Test padding performance on large batches."""
        sequences = [
            torch.randint(0, 100, (torch.randint(10, 100, (1,)).item(),))
            for _ in range(1000)
        ]

        import time

        start = time.time()
        padded = pad_sequences(sequences, padding_value=0)
        elapsed = time.time() - start

        assert padded.shape[0] == 1000
        assert elapsed < 1.0  # Should be fast

    def test_metrics_computation_performance(self):
        """Test metrics computation on large tensors."""
        rewards = torch.randn(1000, 100)
        values = torch.randn(1000, 100)

        import time

        start = time.time()
        advantages = compute_advantages(rewards, values)
        elapsed = time.time() - start

        assert advantages.shape == rewards.shape
        assert elapsed < 2.0  # Should be reasonably fast
