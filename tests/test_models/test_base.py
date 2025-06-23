"""
Tests for base model classes and protocols.

This module tests the foundational model interfaces and base classes
used throughout ThinkRL. It ensures that model protocols are correctly
defined and that base implementations work as expected.
"""

from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

# Import test utilities
from tests.test_models import (
    TEST_DEVICES,
    MockModel,
    MockValueModel,
    ModelTestConfig,
    assert_model_output,
    create_dummy_batch,
    create_mock_tokenizer,
)

# Import the actual base classes we're testing
# Note: These imports will need to be updated based on actual implementation
try:
    from thinkrl.models.base import BaseModel, ModelConfig, ModelProtocol
except ImportError:
    # Create mock implementations for testing if not yet implemented
    class ModelProtocol:
        def forward(self, *args, **kwargs):
            pass

        def parameters(self):
            pass

        def train(self, mode=True):
            pass

        def eval(self):
            pass

    class ModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BaseModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config


class TestModelProtocol:
    """Test the ModelProtocol interface."""

    def test_mock_model_satisfies_protocol(self):
        """Test that MockModel satisfies the ModelProtocol interface."""
        model = MockModel()

        # Check that all required methods exist
        assert hasattr(model, "forward"), "Model should have forward method"
        assert hasattr(model, "parameters"), "Model should have parameters method"
        assert hasattr(model, "train"), "Model should have train method"
        assert hasattr(model, "eval"), "Model should have eval method"

        # Check that methods are callable
        assert callable(model.forward), "forward should be callable"
        assert callable(model.parameters), "parameters should be callable"
        assert callable(model.train), "train should be callable"
        assert callable(model.eval), "eval should be callable"

    def test_protocol_methods_work(self):
        """Test that protocol methods actually work."""
        model = MockModel()

        # Test parameters
        params = list(model.parameters())
        assert len(params) > 0, "Model should have parameters"
        assert all(
            isinstance(p, torch.Tensor) for p in params
        ), "All parameters should be tensors"

        # Test train/eval modes
        model.train()
        assert model.training is True, "Model should be in training mode"

        model.eval()
        assert model.training is False, "Model should be in eval mode"

        # Test forward pass
        batch = create_dummy_batch(batch_size=2, seq_len=16)
        output = model.forward(**batch)
        assert isinstance(output, dict), "Forward should return dictionary"


class TestMockModel:
    """Test the MockModel implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ModelTestConfig()

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return MockModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

    def test_model_initialization(self, config):
        """Test model initialization with different configurations."""
        model = MockModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        assert model.vocab_size == config.vocab_size
        assert model.hidden_size == config.hidden_size
        assert model.num_layers == config.num_layers
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.lm_head, nn.Linear)

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_forward_pass(self, model, device):
        """Test forward pass on different devices."""
        model = model.to(device)
        batch = create_dummy_batch(
            batch_size=4,
            seq_len=32,
            device=device,
        )

        output = model(**batch)

        # Validate output structure
        assert_model_output(
            output=output,
            expected_batch_size=4,
            expected_seq_len=32,
            expected_vocab_size=model.vocab_size,
            should_have_loss=False,
        )

        # Check device placement
        assert output["logits"].device.type == device.split(":")[0]
        assert output["hidden_states"].device.type == device.split(":")[0]

    def test_forward_with_labels(self, model):
        """Test forward pass with labels for loss computation."""
        batch = create_dummy_batch(
            batch_size=4,
            seq_len=32,
            include_labels=True,
        )

        output = model(**batch)

        # Should have loss when labels provided
        assert_model_output(
            output=output,
            expected_batch_size=4,
            expected_seq_len=32,
            expected_vocab_size=model.vocab_size,
            should_have_loss=True,
        )

        # Loss should be positive (CrossEntropyLoss)
        assert output["loss"].item() > 0, "Loss should be positive"

    def test_generation(self, model):
        """Test text generation functionality."""
        input_ids = torch.randint(0, model.vocab_size, (2, 10))

        generated = model.generate(
            input_ids=input_ids,
            max_length=20,
            temperature=1.0,
        )

        # Check output shape
        assert generated.shape == (
            2,
            20,
        ), f"Generated shape {generated.shape} != (2, 20)"

        # Check that original input is preserved
        assert torch.equal(
            generated[:, :10], input_ids
        ), "Original input should be preserved in generation"

        # Check that new tokens were generated
        assert not torch.equal(generated, input_ids), "New tokens should be generated"

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 1),
            (1, 100),
            (8, 50),
            (16, 25),
        ],
    )
    def test_different_input_sizes(self, model, batch_size, seq_len):
        """Test model with different input sizes."""
        batch = create_dummy_batch(
            batch_size=batch_size,
            seq_len=seq_len,
        )

        output = model(**batch)

        assert_model_output(
            output=output,
            expected_batch_size=batch_size,
            expected_seq_len=seq_len,
            expected_vocab_size=model.vocab_size,
        )

    def test_attention_mask_handling(self, model):
        """Test that attention mask is properly handled."""
        batch_size, seq_len = 4, 32

        # Create batch with custom attention mask
        batch = create_dummy_batch(batch_size=batch_size, seq_len=seq_len)

        # Mask out last half of sequence
        batch["attention_mask"][:, seq_len // 2 :] = 0

        output = model(**batch)

        # Should still produce valid output
        assert_model_output(
            output=output,
            expected_batch_size=batch_size,
            expected_seq_len=seq_len,
            expected_vocab_size=model.vocab_size,
        )

    def test_gradient_flow(self, model):
        """Test that gradients flow properly through the model."""
        batch = create_dummy_batch(
            batch_size=2,
            seq_len=16,
            include_labels=True,
        )

        # Forward pass
        output = model(**batch)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for parameter {name}"

    def test_train_eval_modes(self, model):
        """Test train/eval mode switching."""
        # Test training mode
        model.train()
        assert model.training is True

        # All submodules should be in training mode
        for module in model.modules():
            if hasattr(module, "training"):
                assert (
                    module.training is True
                ), f"Module {type(module)} not in training mode"

        # Test eval mode
        model.eval()
        assert model.training is False

        # All submodules should be in eval mode
        for module in model.modules():
            if hasattr(module, "training"):
                assert (
                    module.training is False
                ), f"Module {type(module)} not in eval mode"


class TestMockValueModel:
    """Test the MockValueModel implementation."""

    @pytest.fixture
    def value_model(self):
        """Create test value model."""
        return MockValueModel(hidden_size=512)

    def test_value_model_initialization(self, value_model):
        """Test value model initialization."""
        assert value_model.hidden_size == 512
        assert isinstance(value_model.value_head, nn.Linear)
        assert value_model.value_head.in_features == 512
        assert value_model.value_head.out_features == 1

    def test_value_model_forward(self, value_model):
        """Test value model forward pass."""
        batch_size, seq_len, hidden_size = 4, 32, 512

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = value_model(hidden_states=hidden_states)

        assert "values" in output
        assert output["values"].shape == (batch_size, seq_len)
        assert output["values"].dtype in [torch.float32, torch.float16]

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_value_model_device(self, value_model, device):
        """Test value model on different devices."""
        value_model = value_model.to(device)

        hidden_states = torch.randn(2, 16, 512, device=device)

        output = value_model(hidden_states=hidden_states)

        assert output["values"].device.type == device.split(":")[0]


class TestModelUtilities:
    """Test model utility functions."""

    def test_create_dummy_batch(self):
        """Test dummy batch creation."""
        batch = create_dummy_batch(
            batch_size=4,
            seq_len=32,
            vocab_size=1000,
            include_labels=True,
            include_rewards=True,
        )

        # Check required keys
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "rewards" in batch

        # Check shapes
        assert batch["input_ids"].shape == (4, 32)
        assert batch["attention_mask"].shape == (4, 32)
        assert batch["labels"].shape == (4, 32)
        assert batch["rewards"].shape == (4, 32)

        # Check data types
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["rewards"].dtype == torch.float32

        # Check value ranges
        assert torch.all(batch["input_ids"] >= 0)
        assert torch.all(batch["input_ids"] < 1000)
        # Check attention mask values are 0 or 1
        assert torch.all(
            (batch["attention_mask"] == 0) | (batch["attention_mask"] == 1)
        )

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_create_dummy_batch_device(self, device):
        """Test dummy batch creation on different devices."""
        batch = create_dummy_batch(
            batch_size=2,
            seq_len=16,
            device=device,
        )

        assert batch["input_ids"].device.type == device.split(":")[0]
        assert batch["attention_mask"].device.type == device.split(":")[0]

    def test_assert_model_output_valid(self):
        """Test assert_model_output with valid output."""
        output = {
            "logits": torch.randn(4, 32, 1000),
            "hidden_states": torch.randn(4, 32, 512),
            "loss": torch.tensor(1.5),
        }

        # Should not raise any assertions
        assert_model_output(
            output=output,
            expected_batch_size=4,
            expected_seq_len=32,
            expected_vocab_size=1000,
            should_have_loss=True,
        )

    def test_assert_model_output_invalid_shape(self):
        """Test assert_model_output with invalid shapes."""
        output = {
            "logits": torch.randn(4, 32, 500),  # Wrong vocab size
            "hidden_states": torch.randn(4, 32, 512),
        }

        with pytest.raises(AssertionError, match="Logits shape"):
            assert_model_output(
                output=output,
                expected_batch_size=4,
                expected_seq_len=32,
                expected_vocab_size=1000,
            )

    def test_assert_model_output_missing_keys(self):
        """Test assert_model_output with missing required keys."""
        output = {
            "logits": torch.randn(4, 32, 1000),
            # Missing hidden_states
        }

        with pytest.raises(AssertionError, match="hidden_states"):
            assert_model_output(
                output=output,
                expected_batch_size=4,
                expected_seq_len=32,
                expected_vocab_size=1000,
            )

    def test_create_mock_tokenizer(self):
        """Test mock tokenizer creation and functionality."""
        tokenizer = create_mock_tokenizer()

        # Test basic properties
        assert hasattr(tokenizer, "vocab_size")
        assert hasattr(tokenizer, "pad_token_id")
        assert hasattr(tokenizer, "eos_token_id")
        assert hasattr(tokenizer, "bos_token_id")

        # Test encoding/decoding
        text = "Hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert isinstance(encoded, list)
        assert isinstance(decoded, str)

        # Test tokenizer call
        result = tokenizer(["Hello", "World"], max_length=32)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2  # batch size
        assert result["input_ids"].shape[1] == 32  # max length


class TestModelIntegration:
    """Integration tests for model components."""

    def test_model_with_algorithms(self):
        """Test that mock models work with algorithm interfaces."""
        # This is a placeholder for testing model integration with algorithms
        # Will be expanded when algorithm classes are implemented

        model = MockModel()
        batch = create_dummy_batch(batch_size=2, seq_len=16)

        # Test that model satisfies basic algorithm requirements
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")
        assert hasattr(model, "train")
        assert hasattr(model, "eval")

        # Test forward pass works
        output = model(**batch)
        assert isinstance(output, dict)
        assert "logits" in output

    @pytest.mark.slow
    def test_model_memory_usage(self):
        """Test model memory usage (marked as slow test)."""
        model = MockModel(
            vocab_size=10000,
            hidden_size=1024,
            num_layers=8,
        )

        # Test with larger batch
        batch = create_dummy_batch(
            batch_size=16,
            seq_len=128,
            vocab_size=10000,
        )

        # Should handle larger inputs without issues
        output = model(**batch)

        assert_model_output(
            output=output,
            expected_batch_size=16,
            expected_seq_len=128,
            expected_vocab_size=10000,
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_gpu_memory(self):
        """Test model GPU memory usage."""
        model = MockModel().cuda()

        batch = create_dummy_batch(
            batch_size=8,
            seq_len=64,
            device="cuda",
        )

        # Clear cache before test
        torch.cuda.empty_cache()

        # Test forward pass
        output = model(**batch)

        # Check GPU memory usage
        memory_used = torch.cuda.memory_allocated()
        assert memory_used > 0, "Should use GPU memory"

        # Cleanup
        del model, batch, output
        torch.cuda.empty_cache()
