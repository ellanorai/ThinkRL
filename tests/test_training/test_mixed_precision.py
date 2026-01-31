"""
Test Suite for ThinkRL Mixed Precision Training
================================================

Tests for:
- thinkrl.training.mixed_precision.MixedPrecisionTrainer

Specifically tests Bug #1 fix: Double unscale_() crash prevention
"""

import pytest
import torch
import torch.nn as nn


# CUDA availability check
CUDA_AVAILABLE = torch.cuda.is_available()

try:
    from thinkrl.training.mixed_precision import MixedPrecisionTrainer, PrecisionType

    _MIXED_PRECISION_AVAILABLE = True
except ImportError:
    _MIXED_PRECISION_AVAILABLE = False
    MixedPrecisionTrainer = None
    PrecisionType = None


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    model = nn.Linear(10, 10)
    model = model.cuda()
    return model


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for the model."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.mark.skipif(not _MIXED_PRECISION_AVAILABLE, reason="MixedPrecisionTrainer not available")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16 tests")
class TestMixedPrecisionTrainer:
    """Tests for MixedPrecisionTrainer."""

    def test_fp32_initialization(self):
        """Test FP32 mode initialization (no scaler)."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP32)
        assert trainer.scaler is None
        assert trainer._grads_unscaled is False

    def test_bf16_initialization(self):
        """Test BF16 mode initialization (no scaler)."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.BF16)
        assert trainer.scaler is None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16")
    def test_fp16_initialization(self):
        """Test FP16 mode creates scaler and tracking flag."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP16)
        assert trainer.scaler is not None
        assert trainer._grads_unscaled is False

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16")
    def test_clip_gradients_sets_unscaled_flag(self, simple_model, optimizer):
        """Test that clip_gradients sets _grads_unscaled to True for FP16."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP16, max_grad_norm=1.0)

        # Simulate a backward pass
        x = torch.randn(2, 10, device="cuda")
        y = simple_model(x)
        loss = y.sum()

        with trainer.autocast():
            loss = simple_model(x).sum()

        trainer.backward(loss)

        # Before clip, flag should be False
        assert trainer._grads_unscaled is False

        # Call clip_gradients
        trainer.clip_gradients(simple_model, optimizer)

        # After clip, flag should be True
        assert trainer._grads_unscaled is True

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16")
    def test_step_resets_unscaled_flag(self, simple_model, optimizer):
        """Test that step() resets _grads_unscaled to False after FP16 step."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP16, max_grad_norm=1.0)

        x = torch.randn(2, 10, device="cuda")

        with trainer.autocast():
            loss = simple_model(x).sum()

        trainer.backward(loss)

        # Manually set flag to simulate clip_gradients was called
        trainer._grads_unscaled = True

        # Call step
        trainer.step(optimizer, model=None)  # Don't clip again

        # After step, flag should be reset to False
        assert trainer._grads_unscaled is False

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16")
    def test_no_double_unscale_when_clipping_twice(self, simple_model, optimizer):
        """Test Bug #1 fix: calling clip_gradients twice doesn't double unscale."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP16, max_grad_norm=1.0)

        x = torch.randn(2, 10, device="cuda")

        with trainer.autocast():
            loss = simple_model(x).sum()

        trainer.backward(loss)

        # First clip should unscale
        trainer.clip_gradients(simple_model, optimizer)
        assert trainer._grads_unscaled is True

        # Second clip should NOT unscale again (would crash before fix)
        trainer.clip_gradients(simple_model, optimizer)
        assert trainer._grads_unscaled is True  # Still True, no crash

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for FP16")
    def test_step_with_model_clips_and_steps_correctly(self, simple_model, optimizer):
        """Test that step(model=model) clips gradients and steps correctly."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP16, max_grad_norm=1.0)

        x = torch.randn(2, 10, device="cuda")

        with trainer.autocast():
            loss = simple_model(x).sum()

        trainer.backward(loss)

        # This should NOT crash (was crashing before Bug #1 fix)
        trainer.step(optimizer, model=simple_model)

        # After step, flag should be reset
        assert trainer._grads_unscaled is False


class TestMixedPrecisionTrainerCPU:
    """CPU-only tests for MixedPrecisionTrainer."""

    @pytest.mark.skipif(not _MIXED_PRECISION_AVAILABLE, reason="MixedPrecisionTrainer not available")
    def test_fp32_cpu_training(self):
        """Test FP32 training on CPU works."""
        trainer = MixedPrecisionTrainer(precision=PrecisionType.FP32, max_grad_norm=1.0)

        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 10)

        with trainer.autocast():
            loss = model(x).sum()

        trainer.backward(loss)
        trainer.step(optimizer, model=model)

        # Should complete without error
        assert True

    @pytest.mark.skipif(not _MIXED_PRECISION_AVAILABLE, reason="MixedPrecisionTrainer not available")
    def test_autocast_context_manager(self):
        """Test autocast context manager returns proper context."""
        trainer_fp32 = MixedPrecisionTrainer(precision=PrecisionType.FP32)
        trainer_bf16 = MixedPrecisionTrainer(precision=PrecisionType.BF16)

        # Should not raise
        with trainer_fp32.autocast():
            pass

        with trainer_bf16.autocast():
            pass
