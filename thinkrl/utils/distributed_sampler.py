"""
ThinkRL Distributed Sampler
============================

Distributed data sampling utilities aligned with OpenRLHF patterns.
Supports resumable training with consumed samples tracking.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import math
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


logger = logging.getLogger(__name__)

_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    """
    Sampler that restricts data loading to a subset of the dataset.

    Enhanced version of PyTorch's DistributedSampler with support for:
    - Consumed samples tracking for resumable training
    - Configurable shuffling and seeding
    - Drop-last functionality for even distribution

    This is aligned with OpenRLHF's DistributedSampler for RLHF training.

    Example:
        ```python
        dataset = MyDataset()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in dataloader:
                # Training step
                pass

        # Resume training with consumed samples
        sampler.set_epoch(resume_epoch, consumed_samples=1000)
        ```
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        consumed_samples: int = 0,
    ) -> None:
        """
        Initialize the distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes in distributed training
            rank: Rank of current process
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling (should be same across all processes)
            drop_last: Whether to drop incomplete batches
            consumed_samples: Number of samples already consumed (for resuming)
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_samples = consumed_samples

        # Calculate number of samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Truncate to make evenly divisible
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            # Pad to make evenly divisible
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        logger.debug(
            f"DistributedSampler initialized: rank={rank}, num_replicas={num_replicas}, "
            f"num_samples={self.num_samples}, total_size={self.total_size}"
        )

    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for this replica.

        Yields:
            Dataset indices for this process
        """
        if self.shuffle:
            # Deterministic shuffling based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Pad indices to make evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Truncate to make evenly divisible
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this replica
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Skip consumed samples (for resuming training)
        consumed_per_replica = self.consumed_samples // self.num_replicas
        if consumed_per_replica > 0:
            indices = indices[consumed_per_replica:]

        return iter(indices)

    def __len__(self) -> int:
        """
        Return number of remaining samples for this replica.

        Returns:
            Number of samples (accounting for consumed samples)
        """
        consumed_per_replica = self.consumed_samples // self.num_replicas
        return max(0, self.num_samples - consumed_per_replica)

    def set_epoch(self, epoch: int, consumed_samples: int = 0) -> None:
        """
        Set the epoch and consumed samples for deterministic shuffling.

        Args:
            epoch: Current epoch number
            consumed_samples: Number of samples already consumed in this epoch
        """
        self.epoch = epoch
        self.consumed_samples = consumed_samples
        logger.debug(f"DistributedSampler epoch set: epoch={epoch}, consumed_samples={consumed_samples}")


class DistributedBatchSampler(Sampler[list[int]]):
    """
    Wraps a sampler to yield batches of indices.

    Enhanced for distributed training with support for:
    - Even distribution across replicas
    - Consumed batch tracking for resuming
    - Variable batch sizes

    Example:
        ```python
        dataset = MyDataset()
        sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        batch_sampler = DistributedBatchSampler(sampler, batch_size=32)

        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        ```
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool = False,
        consumed_batches: int = 0,
    ) -> None:
        """
        Initialize the batch sampler.

        Args:
            sampler: Base sampler to draw indices from
            batch_size: Number of samples per batch
            drop_last: Whether to drop incomplete final batch
            consumed_batches: Number of batches already consumed (for resuming)
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.consumed_batches = consumed_batches

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield batches of indices.

        Yields:
            List of indices for each batch
        """
        batch = []
        batch_idx = 0

        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if batch_idx >= self.consumed_batches:
                    yield batch
                batch_idx += 1
                batch = []

        if batch and not self.drop_last:
            if batch_idx >= self.consumed_batches:
                yield batch

    def __len__(self) -> int:
        """
        Return number of remaining batches.

        Returns:
            Number of batches (accounting for consumed batches)
        """
        if self.drop_last:
            total_batches = len(self.sampler) // self.batch_size
        else:
            total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.consumed_batches)

    def set_consumed_batches(self, consumed_batches: int) -> None:
        """
        Set the number of consumed batches for resuming.

        Args:
            consumed_batches: Number of batches already consumed
        """
        self.consumed_batches = consumed_batches


class SequentialDistributedSampler(Sampler[int]):
    """
    Sequential sampler for distributed evaluation.

    Unlike random sampling, this ensures consistent ordering across
    evaluations while still distributing work across processes.

    Example:
        ```python
        # For evaluation
        eval_sampler = SequentialDistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
        )
        eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)
        ```
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        """
        Initialize the sequential distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes in distributed training
            rank: Rank of current process
        """
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """
        Generate sequential indices for this replica.

        Yields:
            Dataset indices for this process
        """
        indices = list(range(len(self.dataset)))

        # Pad to make evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        assert len(indices) == self.total_size

        # Subsample for this replica (sequential chunks)
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        indices = indices[start_idx:end_idx]

        # Filter out padded indices that exceed dataset length
        indices = [i for i in indices if i < len(self.dataset)]

        return iter(indices)

    def __len__(self) -> int:
        """
        Return number of samples for this replica.

        Returns:
            Number of samples
        """
        # Calculate actual samples (excluding padding)
        start_idx = self.rank * self.num_samples
        end_idx = min(start_idx + self.num_samples, len(self.dataset))
        return max(0, end_idx - start_idx)


# Public API
__all__ = [
    "DistributedSampler",
    "DistributedBatchSampler",
    "SequentialDistributedSampler",
]
