"""
ThinkRL Sequence Length Balancing
==================================

Algorithms for balancing sequence lengths across distributed workers.
Aligned with OpenRLHF patterns for efficient batch processing.

These algorithms help minimize computational waste by grouping sequences
of similar lengths together, reducing padding overhead.

Ported from the Bytedance verl repository (Apache 2.0 License).

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import copy
import heapq
import logging
from typing import Any


logger = logging.getLogger(__name__)


def ceildiv(a: int, b: int) -> int:
    """
    Ceiling division.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Ceiling of a/b
    """
    return (a + b - 1) // b


def get_reverse_idx(idx_map: list[int]) -> list[int]:
    """
    Create reverse index mapping.

    Args:
        idx_map: Forward index mapping

    Returns:
        Reverse index mapping

    Example:
        ```python
        idx_map = [2, 0, 1]
        reverse = get_reverse_idx(idx_map)
        # reverse = [1, 2, 0]  # Maps back to original positions
        ```
    """
    reverse_idx = [0] * len(idx_map)
    for i, idx in enumerate(idx_map):
        reverse_idx[idx] = i
    return reverse_idx


def karmarkar_karp(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = True,
) -> list[list[int]]:
    """
    Karmarkar-Karp algorithm (largest differencing method) for load balancing.

    This algorithm iteratively merges the two partitions with the largest
    difference, producing near-optimal load balancing.

    Args:
        seqlen_list: List of sequence lengths to partition
        k_partitions: Number of partitions to create
        equal_size: Whether partitions must have equal number of elements

    Returns:
        List of k partitions, each containing indices into seqlen_list

    Example:
        ```python
        seqlen_list = [100, 50, 75, 25, 80, 45]
        partitions = karmarkar_karp(seqlen_list, k_partitions=2)
        # partitions might be [[0, 3, 5], [1, 2, 4]]
        # Balances total tokens: 170 vs 205
        ```
    """
    if len(seqlen_list) < k_partitions:
        # Not enough elements for requested partitions
        partitions: list[list[int]] = [[] for _ in range(k_partitions)]
        for i, _ in enumerate(seqlen_list):
            partitions[i % k_partitions].append(i)
        return partitions

    class Set:
        """Represents a set of items with a running sum."""

        def __init__(self, items: list[int] | None = None, total: int = 0):
            self.items = items if items is not None else []
            self.total = total

        def add(self, item: int, value: int) -> None:
            self.items.append(item)
            self.total += value

        def merge(self, other: "Set") -> "Set":
            return Set(
                items=self.items + other.items,
                total=self.total + other.total,
            )

        def __lt__(self, other: "Set") -> bool:
            return self.total < other.total

    class State:
        """Represents the current state of k partitions."""

        def __init__(self, k: int):
            self.sets = [Set() for _ in range(k)]

        def spread(self) -> int:
            """Difference between largest and smallest partition sums."""
            totals = [s.total for s in self.sets]
            return max(totals) - min(totals)

        def get_partitions(self) -> list[list[int]]:
            """Return partition indices."""
            return [s.items for s in self.sets]

        def merge(self, i: int, j: int) -> "State":
            """Merge two partitions into a new state."""
            new_state = State(len(self.sets) - 1)
            merged = self.sets[i].merge(self.sets[j])
            idx = 0
            for k, s in enumerate(self.sets):
                if k != i and k != j:
                    new_state.sets[idx] = copy.deepcopy(s)
                    idx += 1
            new_state.sets[idx] = merged
            return new_state

        def __lt__(self, other: "State") -> bool:
            return self.spread() < other.spread()

        def __repr__(self) -> str:
            return f"State(spread={self.spread()}, partitions={self.get_partitions()})"

    # Sort items by length (descending)
    indexed_items = [(length, i) for i, length in enumerate(seqlen_list)]
    indexed_items.sort(reverse=True)

    if equal_size:
        # Greedy round-robin assignment with heap for balance
        partition_sums = [(0, i) for i in range(k_partitions)]
        heapq.heapify(partition_sums)
        partitions_out: list[list[int]] = [[] for _ in range(k_partitions)]

        for length, orig_idx in indexed_items:
            # Get partition with smallest sum
            min_sum, min_partition = heapq.heappop(partition_sums)
            partitions_out[min_partition].append(orig_idx)
            heapq.heappush(partition_sums, (min_sum + length, min_partition))

        return partitions_out

    # Use differencing method for unequal sizes
    # Initialize with each item in its own set
    initial_state = State(len(seqlen_list))
    for length, orig_idx in indexed_items:
        initial_state.sets[orig_idx].add(orig_idx, length)

    # Priority queue of states (by spread)
    heap: list[tuple[int, int, State]] = []
    counter = 0
    heapq.heappush(heap, (initial_state.spread(), counter, initial_state))

    best_state = initial_state

    while heap:
        _, _, state = heapq.heappop(heap)

        if len(state.sets) == k_partitions:
            if state.spread() < best_state.spread():
                best_state = state
            continue

        # Try all pairwise merges
        for i in range(len(state.sets)):
            for j in range(i + 1, len(state.sets)):
                new_state = state.merge(i, j)
                counter += 1
                heapq.heappush(heap, (new_state.spread(), counter, new_state))

                # Limit search space
                if counter > 10000:
                    break
            if counter > 10000:
                break
        if counter > 10000:
            break

    return best_state.get_partitions()


def greedy_partition(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = True,
) -> list[list[int]]:
    """
    Greedy first-fit algorithm for sequence length balancing.

    Assigns each item to the partition with the smallest current sum.
    Faster than Karmarkar-Karp but may produce slightly less optimal results.

    Args:
        seqlen_list: List of sequence lengths to partition
        k_partitions: Number of partitions to create
        equal_size: Whether partitions must have equal number of elements

    Returns:
        List of k partitions, each containing indices into seqlen_list

    Example:
        ```python
        seqlen_list = [100, 50, 75, 25, 80, 45]
        partitions = greedy_partition(seqlen_list, k_partitions=2)
        ```
    """
    if len(seqlen_list) < k_partitions:
        partitions: list[list[int]] = [[] for _ in range(k_partitions)]
        for i, _ in enumerate(seqlen_list):
            partitions[i % k_partitions].append(i)
        return partitions

    # Sort by length descending
    indexed_items = [(length, i) for i, length in enumerate(seqlen_list)]
    indexed_items.sort(reverse=True)

    # Min-heap of (sum, partition_index)
    partition_sums = [(0, i) for i in range(k_partitions)]
    heapq.heapify(partition_sums)

    partitions_out: list[list[int]] = [[] for _ in range(k_partitions)]

    if equal_size:
        items_per_partition = ceildiv(len(seqlen_list), k_partitions)
        partition_counts = [0] * k_partitions

        for length, orig_idx in indexed_items:
            # Find partition with smallest sum that isn't full
            candidates = [
                (s, p) for s, p in partition_sums
                if partition_counts[p] < items_per_partition
            ]
            if not candidates:
                # All full, use any
                min_sum, min_partition = min(partition_sums)
            else:
                min_sum, min_partition = min(candidates)

            partitions_out[min_partition].append(orig_idx)
            partition_counts[min_partition] += 1

            # Update heap
            partition_sums = [
                (s + length if p == min_partition else s, p)
                for s, p in partition_sums
            ]
            heapq.heapify(partition_sums)
    else:
        for length, orig_idx in indexed_items:
            min_sum, min_partition = heapq.heappop(partition_sums)
            partitions_out[min_partition].append(orig_idx)
            heapq.heappush(partition_sums, (min_sum + length, min_partition))

    return partitions_out


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = True,
    algorithm: str = "greedy",
) -> list[list[int]]:
    """
    Get balanced partitions for sequence lengths.

    Main public interface for sequence length balancing.

    Args:
        seqlen_list: List of sequence lengths
        k_partitions: Number of partitions to create
        equal_size: Whether partitions must have equal number of elements
        algorithm: Algorithm to use ("greedy" or "karmarkar_karp")

    Returns:
        List of k partitions, each containing indices into seqlen_list

    Example:
        ```python
        # For distributed training across 4 GPUs
        seqlen_list = [len(seq) for seq in batch_sequences]
        partitions = get_seqlen_balanced_partitions(
            seqlen_list,
            k_partitions=4,
            algorithm="greedy"
        )

        # Assign sequences to GPUs
        for gpu_id, indices in enumerate(partitions):
            gpu_sequences = [batch_sequences[i] for i in indices]
        ```
    """
    if algorithm == "greedy":
        return greedy_partition(seqlen_list, k_partitions, equal_size)
    elif algorithm == "karmarkar_karp":
        return karmarkar_karp(seqlen_list, k_partitions, equal_size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'greedy' or 'karmarkar_karp'")


def log_seqlen_unbalance(
    seqlen_list: list[int],
    partitions: list[list[int]],
    prefix: str = "",
) -> dict[str, Any]:
    """
    Log and compute metrics about partition imbalance.

    Args:
        seqlen_list: Original sequence lengths
        partitions: Partition assignments (indices)
        prefix: Prefix for log messages

    Returns:
        Dictionary with imbalance metrics

    Example:
        ```python
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=4)
        metrics = log_seqlen_unbalance(seqlen_list, partitions, prefix="batch_0")
        print(f"Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
        ```
    """
    partition_sums = []
    for partition in partitions:
        total = sum(seqlen_list[i] for i in partition)
        partition_sums.append(total)

    min_sum = min(partition_sums) if partition_sums else 0
    max_sum = max(partition_sums) if partition_sums else 0
    mean_sum = sum(partition_sums) / len(partition_sums) if partition_sums else 0

    imbalance_ratio = (max_sum - min_sum) / mean_sum if mean_sum > 0 else 0
    max_min_ratio = max_sum / min_sum if min_sum > 0 else float("inf")

    metrics = {
        "min_sum": min_sum,
        "max_sum": max_sum,
        "mean_sum": mean_sum,
        "imbalance_ratio": imbalance_ratio,
        "max_min_ratio": max_min_ratio,
        "partition_sums": partition_sums,
        "partition_sizes": [len(p) for p in partitions],
    }

    logger.info(
        f"{prefix}Partition balance: min={min_sum}, max={max_sum}, "
        f"imbalance={imbalance_ratio:.3f}, max/min={max_min_ratio:.3f}"
    )

    return metrics


def get_minimum_num_micro_batch_size(
    total_lengths: list[int],
    max_tokens_per_gpu: int,
    ring_attn_size: int = 1,
    ds_tensor_parallel_size: int = 1,
) -> int:
    """
    Determine minimum number of microbatches using first-fit packing.

    Args:
        total_lengths: List of sequence lengths
        max_tokens_per_gpu: Maximum tokens per GPU per microbatch
        ring_attn_size: Ring attention parallelism size
        ds_tensor_parallel_size: DeepSpeed tensor parallel size

    Returns:
        Minimum number of microbatches needed

    Example:
        ```python
        seqlen_list = [512, 256, 1024, 128, 768]
        min_batches = get_minimum_num_micro_batch_size(
            seqlen_list,
            max_tokens_per_gpu=2048,
        )
        print(f"Need at least {min_batches} microbatches")
        ```
    """
    effective_max = max_tokens_per_gpu * ring_attn_size * ds_tensor_parallel_size

    # Sort descending for first-fit decreasing
    sorted_lengths = sorted(total_lengths, reverse=True)

    bins: list[int] = []  # Current fill level of each bin

    for length in sorted_lengths:
        if length > effective_max:
            logger.warning(
                f"Sequence length {length} exceeds max tokens per GPU {effective_max}"
            )

        # First-fit: find first bin with enough space
        placed = False
        for i, bin_fill in enumerate(bins):
            if bin_fill + length <= effective_max:
                bins[i] += length
                placed = True
                break

        if not placed:
            # Create new bin
            bins.append(length)

    return len(bins)


def reorder_by_seqlen(
    sequences: list[Any],
    seqlen_list: list[int] | None = None,
    descending: bool = True,
) -> tuple[list[Any], list[int]]:
    """
    Reorder sequences by length for more efficient batching.

    Args:
        sequences: List of sequences to reorder
        seqlen_list: Precomputed lengths (computed from sequences if None)
        descending: Whether to sort descending (longest first)

    Returns:
        Tuple of (reordered sequences, indices for restoring original order)

    Example:
        ```python
        sequences = ["short", "medium length", "a"]
        reordered, restore_indices = reorder_by_seqlen(sequences)
        # reordered = ["medium length", "short", "a"]
        # restore_indices = [1, 0, 2]

        # Restore original order
        original = [reordered[i] for i in restore_indices]
        ```
    """
    if seqlen_list is None:
        seqlen_list = [len(s) for s in sequences]

    indexed = list(enumerate(sequences))
    indexed.sort(key=lambda x: seqlen_list[x[0]], reverse=descending)

    reordered = [seq for _, seq in indexed]
    original_indices = [i for i, _ in indexed]

    # Compute restore indices
    restore_indices = get_reverse_idx(original_indices)

    return reordered, restore_indices


# Public API
__all__ = [
    # Core algorithms
    "karmarkar_karp",
    "greedy_partition",
    "get_seqlen_balanced_partitions",
    # Utilities
    "log_seqlen_unbalance",
    "get_minimum_num_micro_batch_size",
    "reorder_by_seqlen",
    "ceildiv",
    "get_reverse_idx",
]
