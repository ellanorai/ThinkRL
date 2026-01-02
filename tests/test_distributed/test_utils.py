"""
Tests for Distributed Utilities
===============================

Tests for distributed training helper functions.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from thinkrl.distributed.utils import (
    is_deepspeed_available,
    reduce_tensor,
    broadcast_tensor,
    all_gather_tensors,
    get_world_info,
    is_main_process,
    barrier,
    print_rank_0,
    gather_scalar,
    compute_global_metrics,
)


class TestIsDeepSpeedAvailable:
    """Tests for DeepSpeed availability check."""

    def test_returns_boolean(self):
        """Test that is_deepspeed_available returns a boolean."""
        result = is_deepspeed_available()
        assert isinstance(result, bool)


class TestReduceTensor:
    """Tests for reduce_tensor function."""

    def test_reduce_without_distributed(self):
        """Test reduce when distributed is not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor)

        assert torch.allclose(result, tensor)

    def test_reduce_preserves_shape(self):
        """Test that reduce preserves tensor shape."""
        tensor = torch.randn(4, 8)
        result = reduce_tensor(tensor)

        assert result.shape == tensor.shape

    def test_reduce_mean_operation(self):
        """Test mean reduction operation."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor, reduce_op="mean")

        assert torch.allclose(result, tensor)

    def test_reduce_sum_operation(self):
        """Test sum reduction operation."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor, reduce_op="sum")

        assert torch.allclose(result, tensor)

    def test_reduce_invalid_operation(self):
        """Test invalid reduction operation."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown reduce_op"):
            reduce_tensor(tensor, reduce_op="invalid")


class TestBroadcastTensor:
    """Tests for broadcast_tensor function."""

    def test_broadcast_without_distributed(self):
        """Test broadcast when distributed is not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = broadcast_tensor(tensor)

        assert torch.allclose(result, tensor)

    def test_broadcast_preserves_shape(self):
        """Test that broadcast preserves tensor shape."""
        tensor = torch.randn(4, 8)
        result = broadcast_tensor(tensor)

        assert result.shape == tensor.shape


class TestAllGatherTensors:
    """Tests for all_gather_tensors function."""

    def test_gather_without_distributed(self):
        """Test gather when distributed is not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_gather_tensors(tensor)

        assert len(result) == 1
        assert torch.allclose(result[0], tensor)


class TestGetWorldInfo:
    """Tests for get_world_info function."""

    def test_returns_dict(self):
        """Test that get_world_info returns a dictionary."""
        result = get_world_info()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        result = get_world_info()

        assert "rank" in result
        assert "local_rank" in result
        assert "world_size" in result

    def test_single_process_values(self):
        """Test values when not distributed."""
        result = get_world_info()

        assert result["rank"] == 0
        assert result["local_rank"] == 0
        assert result["world_size"] == 1


class TestIsMainProcess:
    """Tests for is_main_process function."""

    def test_returns_true_without_distributed(self):
        """Test returns True when not distributed."""
        assert is_main_process() is True


class TestBarrier:
    """Tests for barrier function."""

    def test_barrier_without_distributed(self):
        """Test barrier when distributed is not initialized."""
        # Should not raise
        barrier()


class TestPrintRank0:
    """Tests for print_rank_0 function."""

    def test_prints_without_distributed(self, capsys):
        """Test print_rank_0 prints when not distributed."""
        print_rank_0("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out


class TestGatherScalar:
    """Tests for gather_scalar function."""

    def test_gather_float(self):
        """Test gathering a float value."""
        result = gather_scalar(1.5)

        assert len(result) == 1
        assert result[0] == 1.5

    def test_gather_int(self):
        """Test gathering an int value."""
        result = gather_scalar(10)

        assert len(result) == 1
        assert result[0] == 10.0


class TestComputeGlobalMetrics:
    """Tests for compute_global_metrics function."""

    def test_returns_same_metrics_without_distributed(self):
        """Test returns same metrics when not distributed."""
        metrics = {"loss": 0.5, "accuracy": 0.9}
        result = compute_global_metrics(metrics)

        assert result["loss"] == 0.5
        assert result["accuracy"] == 0.9

    def test_handles_non_numeric_values(self):
        """Test handling of non-numeric values."""
        metrics = {"loss": 0.5, "name": "test"}
        result = compute_global_metrics(metrics)

        assert result["loss"] == 0.5
        assert result["name"] == "test"

    def test_handles_int_values(self):
        """Test handling of integer values."""
        metrics = {"step": 100, "loss": 0.5}
        result = compute_global_metrics(metrics)

        assert result["step"] == 100
        assert result["loss"] == 0.5
