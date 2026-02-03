"""
Test Suite for ThinkRL Distributed Utilities
=============================================

Tests for:
- thinkrl.utils.distributed_util
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from thinkrl.utils.distributed_util import (
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    broadcast_object,
    cleanup_distributed,
    gather_object,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    reduce_mean,
    stateless_init_process_group,
    torch_dist_barrier_and_cuda_sync,
)


class TestBasicDistributedInfo:
    """Tests for basic distributed info functions."""

    def test_get_rank_not_initialized(self):
        """Test get_rank when dist is not initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                assert get_rank() == 0

    def test_get_rank_initialized(self):
        """Test get_rank when dist is initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.get_rank", return_value=3):
                    assert get_rank() == 3

    def test_get_world_size_not_initialized(self):
        """Test get_world_size when dist is not initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                assert get_world_size() == 1

    def test_get_world_size_initialized(self):
        """Test get_world_size when dist is initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.get_world_size", return_value=4):
                    assert get_world_size() == 4

    def test_get_local_rank_default(self):
        """Test get_local_rank with no env var."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_local_rank() == 0

    def test_get_local_rank_from_env(self):
        """Test get_local_rank from environment."""
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            assert get_local_rank() == 2

    def test_is_main_process_true(self):
        """Test is_main_process returns True for rank 0."""
        with patch("thinkrl.utils.distributed_util.get_rank", return_value=0):
            assert is_main_process() is True

    def test_is_main_process_false(self):
        """Test is_main_process returns False for non-zero rank."""
        with patch("thinkrl.utils.distributed_util.get_rank", return_value=1):
            assert is_main_process() is False

    def test_is_distributed_true(self):
        """Test is_distributed when initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                assert is_distributed() is True

    def test_is_distributed_false_not_available(self):
        """Test is_distributed when not available."""
        with patch("torch.distributed.is_available", return_value=False):
            assert is_distributed() is False

    def test_is_distributed_false_not_initialized(self):
        """Test is_distributed when not initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                assert is_distributed() is False


class TestInitDistributed:
    """Tests for init_distributed function."""

    def test_init_already_initialized(self):
        """Test init when already initialized returns True."""
        with patch("torch.distributed.is_initialized", return_value=True):
            result = init_distributed()
            assert result is True

    def test_init_single_process(self):
        """Test init with world_size=1 skips initialization."""
        with patch("torch.distributed.is_initialized", return_value=False):
            result = init_distributed(world_size=1, rank=0)
            assert result is False

    def test_init_from_env_single_process(self):
        """Test init reads from env and skips for single process."""
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
                result = init_distributed()
                assert result is False

    def test_init_success(self):
        """Test successful initialization."""
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.init_process_group") as mock_init:
                with patch("torch.cuda.is_available", return_value=False):
                    with patch("thinkrl.utils.distributed_util.get_rank", return_value=0):
                        with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                            with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
                                result = init_distributed(backend="gloo")

                                assert result is True
                                mock_init.assert_called_once()

    def test_init_with_init_method(self):
        """Test initialization with custom init_method."""
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.init_process_group") as mock_init:
                with patch("torch.cuda.is_available", return_value=False):
                    with patch("thinkrl.utils.distributed_util.get_rank", return_value=0):
                        with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                            result = init_distributed(
                                backend="gloo",
                                init_method="tcp://localhost:29500",
                                world_size=2,
                                rank=0,
                            )

                            assert result is True
                            mock_init.assert_called_once_with(
                                backend="gloo",
                                init_method="tcp://localhost:29500",
                                world_size=2,
                                rank=0,
                            )

    def test_init_failure(self):
        """Test initialization failure returns False."""
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.init_process_group", side_effect=Exception("Init failed")):
                with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
                    result = init_distributed()
                    assert result is False


class TestSynchronization:
    """Tests for synchronization functions."""

    def test_barrier_not_distributed(self):
        """Test barrier is no-op when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            # Should not raise
            barrier()

    def test_barrier_distributed(self):
        """Test barrier calls dist.barrier when distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("torch.distributed.barrier") as mock_barrier:
                barrier()
                mock_barrier.assert_called_once()

    def test_torch_dist_barrier_and_cuda_sync(self):
        """Test combined barrier and CUDA sync."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.barrier") as mock_barrier:
                    with patch("torch.cuda.is_available", return_value=True):
                        with patch("torch.cuda.synchronize") as mock_sync:
                            torch_dist_barrier_and_cuda_sync()

                            mock_barrier.assert_called_once()
                            mock_sync.assert_called_once()

    def test_torch_dist_barrier_no_cuda(self):
        """Test barrier without CUDA."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=True):
                with patch("torch.distributed.barrier") as mock_barrier:
                    with patch("torch.cuda.is_available", return_value=False):
                        torch_dist_barrier_and_cuda_sync()
                        mock_barrier.assert_called_once()


class TestCollectiveOperations:
    """Tests for collective operations."""

    def test_all_reduce_not_distributed(self):
        """Test all_reduce returns tensor unchanged when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = all_reduce(tensor)
            assert torch.equal(result, tensor)

    def test_all_reduce_distributed(self):
        """Test all_reduce calls dist.all_reduce when distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_reduce:
                tensor = torch.tensor([1.0, 2.0, 3.0])
                result = all_reduce(tensor)

                mock_reduce.assert_called_once()
                assert result is tensor

    def test_all_reduce_async(self):
        """Test all_reduce with async_op=True."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            mock_work = MagicMock()
            with patch("torch.distributed.all_reduce", return_value=mock_work) as mock_reduce:
                tensor = torch.tensor([1.0, 2.0, 3.0])
                result = all_reduce(tensor, async_op=True)

                mock_reduce.assert_called_once()
                assert result is mock_work

    def test_all_gather_not_distributed(self):
        """Test all_gather returns list with tensor when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            tensor = torch.tensor([1.0, 2.0])
            result = all_gather(tensor)

            assert len(result) == 1
            assert torch.equal(result[0], tensor)

    def test_all_gather_distributed(self):
        """Test all_gather calls dist.all_gather when distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                with patch("torch.distributed.all_gather") as mock_gather:
                    tensor = torch.tensor([1.0, 2.0])
                    result = all_gather(tensor)

                    mock_gather.assert_called_once()
                    assert len(result) == 2

    def test_broadcast_not_distributed(self):
        """Test broadcast returns tensor unchanged when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            tensor = torch.tensor([1.0, 2.0])
            result = broadcast(tensor, src=0)
            assert torch.equal(result, tensor)

    def test_broadcast_distributed(self):
        """Test broadcast calls dist.broadcast when distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("torch.distributed.broadcast") as mock_broadcast:
                tensor = torch.tensor([1.0, 2.0])
                result = broadcast(tensor, src=0)

                mock_broadcast.assert_called_once_with(tensor, src=0)
                assert result is tensor

    def test_reduce_mean_not_distributed(self):
        """Test reduce_mean returns tensor unchanged when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = reduce_mean(tensor)
            assert torch.equal(result, tensor)

    def test_reduce_mean_distributed(self):
        """Test reduce_mean computes mean across processes."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                with patch("torch.distributed.all_reduce") as mock_reduce:
                    tensor = torch.tensor([2.0, 4.0])
                    result = reduce_mean(tensor)

                    mock_reduce.assert_called_once()
                    # Result should be tensor / 2
                    assert torch.allclose(result, torch.tensor([1.0, 2.0]))


class TestObjectCommunication:
    """Tests for object-based communication."""

    def test_gather_object_not_distributed(self):
        """Test gather_object returns list with object when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            obj = {"key": "value"}
            result = gather_object(obj, dst=0)

            assert result == [obj]

    def test_gather_object_dst_rank(self):
        """Test gather_object on destination rank."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                with patch("thinkrl.utils.distributed_util.get_rank", return_value=0):
                    with patch("torch.distributed.gather_object") as mock_gather:
                        obj = {"key": "value"}
                        result = gather_object(obj, dst=0)

                        mock_gather.assert_called_once()
                        assert result is not None

    def test_gather_object_non_dst_rank(self):
        """Test gather_object on non-destination rank."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("thinkrl.utils.distributed_util.get_world_size", return_value=2):
                with patch("thinkrl.utils.distributed_util.get_rank", return_value=1):
                    with patch("torch.distributed.gather_object") as mock_gather:
                        obj = {"key": "value"}
                        result = gather_object(obj, dst=0)

                        mock_gather.assert_called_once()
                        assert result is None

    def test_broadcast_object_not_distributed(self):
        """Test broadcast_object returns object when not distributed."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            obj = {"key": "value"}
            result = broadcast_object(obj, src=0)
            assert result == obj

    def test_broadcast_object_distributed(self):
        """Test broadcast_object calls dist.broadcast_object_list."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("thinkrl.utils.distributed_util.get_rank", return_value=0):
                with patch("torch.distributed.broadcast_object_list") as mock_broadcast:
                    obj = {"key": "value"}
                    broadcast_object(obj, src=0)

                    mock_broadcast.assert_called_once()


class TestCleanup:
    """Tests for cleanup functions."""

    def test_cleanup_distributed_not_initialized(self):
        """Test cleanup when not distributed is no-op."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=False):
            # Should not raise
            cleanup_distributed()

    def test_cleanup_distributed_initialized(self):
        """Test cleanup destroys process group."""
        with patch("thinkrl.utils.distributed_util.is_distributed", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                cleanup_distributed()
                mock_destroy.assert_called_once()


class TestStatelessProcessGroup:
    """Tests for stateless_init_process_group."""

    def test_stateless_without_vllm(self):
        """Test stateless init returns None without vLLM."""
        with patch("thinkrl.utils.distributed_util._VLLM_AVAILABLE", False):
            result = stateless_init_process_group(
                master_address="localhost",
                master_port=29500,
                rank=0,
                world_size=2,
                device="cuda:0",
            )
            assert result is None

    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_stateless_with_vllm(self):
        """Test stateless init with vLLM (requires vLLM)."""
        # This test would require vLLM to be installed
        pass
