"""
Test Suite for ThinkRL Distributed Sampler Utilities
====================================================

Tests for:
- DistributedSampler
- DistributedBatchSampler
- SequentialDistributedSampler
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Dataset

from thinkrl.utils.distributed_sampler import (
    DistributedBatchSampler,
    DistributedSampler,
    SequentialDistributedSampler,
)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 100):
        self.size = size
        self.data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TestDistributedSampler:
    """Tests for DistributedSampler class."""

    @pytest.fixture
    def dataset(self):
        return SimpleDataset(100)

    @pytest.fixture
    def small_dataset(self):
        return SimpleDataset(10)

    def test_init_basic(self, dataset):
        """Test basic initialization."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=True,
            seed=42,
        )

        assert sampler.dataset is dataset
        assert sampler.num_replicas == 4
        assert sampler.rank == 0
        assert sampler.shuffle is True
        assert sampler.seed == 42
        assert sampler.consumed_samples == 0

    def test_init_with_consumed_samples(self, dataset):
        """Test initialization with consumed samples."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            consumed_samples=20,
        )

        assert sampler.consumed_samples == 20

    def test_init_invalid_rank(self, dataset):
        """Test initialization with invalid rank raises error."""
        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedSampler(
                dataset=dataset,
                num_replicas=4,
                rank=4,  # Invalid: should be 0-3
            )

        with pytest.raises(ValueError, match="Invalid rank"):
            DistributedSampler(
                dataset=dataset,
                num_replicas=4,
                rank=-1,
            )

    def test_init_auto_rank_not_initialized(self, dataset):
        """Test auto rank detection when dist is not initialized."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                sampler = DistributedSampler(dataset=dataset)
                assert sampler.num_replicas == 1
                assert sampler.rank == 0

    def test_iter_shuffled(self, dataset):
        """Test iterator with shuffling."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=True,
            seed=42,
        )

        indices = list(sampler)

        # Should have approximately 25 samples (100 / 4)
        assert len(indices) == 25

        # All indices should be valid
        assert all(0 <= idx < 100 for idx in indices)

    def test_iter_not_shuffled(self, dataset):
        """Test iterator without shuffling."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=False,
        )

        indices = list(sampler)
        assert len(indices) == 25

        # Indices should be deterministic (strided)
        expected = list(range(0, 100, 4))
        assert indices == expected

    def test_iter_different_ranks(self, dataset):
        """Test that different ranks get different indices."""
        samplers = [
            DistributedSampler(
                dataset=dataset,
                num_replicas=4,
                rank=i,
                shuffle=False,
            )
            for i in range(4)
        ]

        all_indices = [list(s) for s in samplers]

        # Each rank should have same number of samples
        assert all(len(idx) == 25 for idx in all_indices)

        # No overlap between ranks
        for i in range(4):
            for j in range(i + 1, 4):
                assert set(all_indices[i]).isdisjoint(set(all_indices[j]))

    def test_iter_with_consumed_samples(self, dataset):
        """Test that consumed samples are skipped."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            consumed_samples=20,  # 10 per replica
        )

        indices = list(sampler)

        # Should skip 10 samples per replica
        assert len(indices) == 40  # 50 - 10

    def test_iter_drop_last(self, small_dataset):
        """Test drop_last functionality."""
        sampler = DistributedSampler(
            dataset=small_dataset,
            num_replicas=3,
            rank=0,
            shuffle=False,
            drop_last=True,
        )

        indices = list(sampler)
        # With drop_last, should truncate to evenly divisible
        assert len(indices) == 3

    def test_len_basic(self, dataset):
        """Test length calculation."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
        )

        assert len(sampler) == 25

    def test_len_with_consumed(self, dataset):
        """Test length with consumed samples."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            consumed_samples=40,  # 10 per replica
        )

        assert len(sampler) == 15  # 25 - 10

    def test_set_epoch(self, dataset):
        """Test set_epoch method."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=True,
        )

        # Get indices for epoch 0
        indices_epoch0 = list(sampler)

        # Set epoch 1 and get new indices
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Different epochs should produce different orderings
        assert indices_epoch0 != indices_epoch1

    def test_set_epoch_with_consumed(self, dataset):
        """Test set_epoch with consumed samples."""
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
        )

        sampler.set_epoch(5, consumed_samples=30)

        assert sampler.epoch == 5
        assert sampler.consumed_samples == 30

    def test_reproducibility(self, dataset):
        """Test that same seed produces same results."""
        sampler1 = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=True,
            seed=123,
        )

        sampler2 = DistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
            shuffle=True,
            seed=123,
        )

        assert list(sampler1) == list(sampler2)


class TestDistributedBatchSampler:
    """Tests for DistributedBatchSampler class."""

    @pytest.fixture
    def base_sampler(self):
        dataset = SimpleDataset(100)
        return DistributedSampler(
            dataset=dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )

    def test_init_basic(self, base_sampler):
        """Test basic initialization."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
            drop_last=False,
        )

        assert batch_sampler.sampler is base_sampler
        assert batch_sampler.batch_size == 10
        assert batch_sampler.drop_last is False
        assert batch_sampler.consumed_batches == 0

    def test_init_invalid_batch_size(self, base_sampler):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            DistributedBatchSampler(
                sampler=base_sampler,
                batch_size=0,
            )

        with pytest.raises(ValueError, match="positive integer"):
            DistributedBatchSampler(
                sampler=base_sampler,
                batch_size=-1,
            )

    def test_iter_batches(self, base_sampler):
        """Test iteration over batches."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
            drop_last=False,
        )

        batches = list(batch_sampler)

        # 50 samples / 10 per batch = 5 batches
        assert len(batches) == 5
        assert all(len(batch) == 10 for batch in batches)

    def test_iter_drop_last(self, base_sampler):
        """Test drop_last functionality."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=15,
            drop_last=True,
        )

        batches = list(batch_sampler)

        # 50 samples / 15 = 3 full batches (drop remaining 5)
        assert len(batches) == 3
        assert all(len(batch) == 15 for batch in batches)

    def test_iter_with_consumed_batches(self, base_sampler):
        """Test skipping consumed batches."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
            drop_last=False,
            consumed_batches=2,
        )

        batches = list(batch_sampler)

        # 5 batches total - 2 consumed = 3 remaining
        assert len(batches) == 3

    def test_len_basic(self, base_sampler):
        """Test length calculation."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
            drop_last=False,
        )

        assert len(batch_sampler) == 5

    def test_len_drop_last(self, base_sampler):
        """Test length with drop_last."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=15,
            drop_last=True,
        )

        assert len(batch_sampler) == 3

    def test_len_with_consumed(self, base_sampler):
        """Test length with consumed batches."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
            consumed_batches=2,
        )

        assert len(batch_sampler) == 3

    def test_set_consumed_batches(self, base_sampler):
        """Test set_consumed_batches method."""
        batch_sampler = DistributedBatchSampler(
            sampler=base_sampler,
            batch_size=10,
        )

        batch_sampler.set_consumed_batches(3)
        assert batch_sampler.consumed_batches == 3
        assert len(batch_sampler) == 2


class TestSequentialDistributedSampler:
    """Tests for SequentialDistributedSampler class."""

    @pytest.fixture
    def dataset(self):
        return SimpleDataset(100)

    @pytest.fixture
    def small_dataset(self):
        return SimpleDataset(10)

    def test_init_basic(self, dataset):
        """Test basic initialization."""
        sampler = SequentialDistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
        )

        assert sampler.dataset is dataset
        assert sampler.num_replicas == 4
        assert sampler.rank == 0

    def test_init_auto_detection(self, dataset):
        """Test auto detection of rank and world size."""
        with patch("torch.distributed.is_available", return_value=True):
            with patch("torch.distributed.is_initialized", return_value=False):
                sampler = SequentialDistributedSampler(dataset=dataset)
                assert sampler.num_replicas == 1
                assert sampler.rank == 0

    def test_iter_sequential_chunks(self, dataset):
        """Test that indices are sequential chunks."""
        samplers = [
            SequentialDistributedSampler(
                dataset=dataset,
                num_replicas=4,
                rank=i,
            )
            for i in range(4)
        ]

        all_indices = [list(s) for s in samplers]

        # Rank 0 should get first 25, rank 1 next 25, etc.
        assert all_indices[0] == list(range(0, 25))
        assert all_indices[1] == list(range(25, 50))
        assert all_indices[2] == list(range(50, 75))
        assert all_indices[3] == list(range(75, 100))

    def test_iter_with_padding(self, small_dataset):
        """Test padding when dataset not evenly divisible."""
        sampler = SequentialDistributedSampler(
            dataset=small_dataset,
            num_replicas=4,
            rank=3,  # Last rank might get padded samples
        )

        indices = list(sampler)

        # All indices should be valid (no padding indices returned)
        assert all(0 <= idx < 10 for idx in indices)

    def test_len_basic(self, dataset):
        """Test length calculation."""
        sampler = SequentialDistributedSampler(
            dataset=dataset,
            num_replicas=4,
            rank=0,
        )

        assert len(sampler) == 25

    def test_len_uneven(self, small_dataset):
        """Test length with uneven division."""
        # 10 samples / 4 replicas = 2-3 samples per replica
        sampler0 = SequentialDistributedSampler(
            dataset=small_dataset,
            num_replicas=4,
            rank=0,
        )
        sampler3 = SequentialDistributedSampler(
            dataset=small_dataset,
            num_replicas=4,
            rank=3,
        )

        # First replica gets full chunk
        assert len(sampler0) == 3
        # Last replica gets remainder (might be less)
        assert len(sampler3) == 1

    def test_coverage(self, dataset):
        """Test that all samples are covered exactly once."""
        samplers = [
            SequentialDistributedSampler(
                dataset=dataset,
                num_replicas=4,
                rank=i,
            )
            for i in range(4)
        ]

        all_indices = []
        for s in samplers:
            all_indices.extend(list(s))

        # All indices should be covered
        assert sorted(all_indices) == list(range(100))
