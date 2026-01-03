"""
Sequence Packing Utilities
===========================

Pack multiple short sequences into longer sequences to improve
GPU utilization and training throughput.

Inspired by TRL and OpenRLHF sequence packing implementations.

Author: EllanorAI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class PackingConfig:
    """Configuration for sequence packing."""

    # Maximum packed sequence length
    max_seq_length: int = 2048

    # Padding token ID
    pad_token_id: int = 0

    # EOS token ID (for separating packed sequences)
    eos_token_id: int = 2

    # Whether to add EOS between packed sequences
    add_eos_between: bool = True

    # Minimum sequence length to include
    min_seq_length: int = 1

    # Maximum sequences to pack together
    max_sequences_per_pack: int | None = None

    # Whether to shuffle before packing (recommended)
    shuffle_before_packing: bool = True


def pack_sequences(
    sequences: list[dict[str, torch.Tensor]],
    max_seq_length: int = 2048,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    add_eos_between: bool = True,
) -> list[dict[str, torch.Tensor]]:
    """
    Pack multiple short sequences into longer sequences.

    This reduces padding waste and improves GPU utilization by
    concatenating multiple short sequences into single training examples.

    Args:
        sequences: List of tokenized sequences with "input_ids" and "attention_mask"
        max_seq_length: Maximum length of packed sequences
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for EOS (used to separate sequences)
        add_eos_between: Whether to add EOS tokens between packed sequences

    Returns:
        List of packed sequences

    Example:
        >>> sequences = [
        ...     {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])},
        ...     {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
        ... ]
        >>> packed = pack_sequences(sequences, max_seq_length=10)
        >>> # Now sequences are packed together
    """
    if not sequences:
        return []

    packed = []
    current_input_ids: list[int] = []
    current_attention_mask: list[int] = []
    current_position_ids: list[int] = []
    current_labels: list[int] = []
    has_labels = "labels" in sequences[0]

    for seq in sequences:
        input_ids = seq["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        attention_mask = seq.get("attention_mask")
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
        else:
            attention_mask = [1] * len(input_ids)

        labels = None
        if has_labels:
            labels = seq.get("labels")
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.tolist()

        # Add EOS separator if needed
        if add_eos_between and current_input_ids:
            input_ids = [eos_token_id] + input_ids
            attention_mask = [1] + attention_mask
            if labels is not None:
                labels = [-100] + labels  # Ignore EOS in loss

        seq_len = len(input_ids)

        # Check if we can add this sequence to current pack
        if current_input_ids and len(current_input_ids) + seq_len > max_seq_length:
            # Finalize current pack
            packed.append(
                _finalize_pack(
                    current_input_ids,
                    current_attention_mask,
                    current_position_ids,
                    current_labels if has_labels else None,
                    max_seq_length,
                    pad_token_id,
                )
            )
            current_input_ids = []
            current_attention_mask = []
            current_position_ids = []
            current_labels = []

        # Add sequence to current pack
        start_pos = len(current_input_ids)
        current_input_ids.extend(input_ids)
        current_attention_mask.extend(attention_mask)
        current_position_ids.extend(range(start_pos, start_pos + seq_len))
        if has_labels and labels is not None:
            current_labels.extend(labels)

    # Finalize last pack
    if current_input_ids:
        packed.append(
            _finalize_pack(
                current_input_ids,
                current_attention_mask,
                current_position_ids,
                current_labels if has_labels else None,
                max_seq_length,
                pad_token_id,
            )
        )

    return packed


def _finalize_pack(
    input_ids: list[int],
    attention_mask: list[int],
    position_ids: list[int],
    labels: list[int] | None,
    max_seq_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Finalize a packed sequence with padding."""
    current_len = len(input_ids)
    padding_len = max_seq_length - current_len

    if padding_len > 0:
        input_ids = input_ids + [pad_token_id] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        position_ids = position_ids + list(range(current_len, max_seq_length))
        if labels is not None:
            labels = labels + [-100] * padding_len

    result = {
        "input_ids": torch.tensor(input_ids[:max_seq_length], dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask[:max_seq_length], dtype=torch.long),
        "position_ids": torch.tensor(position_ids[:max_seq_length], dtype=torch.long),
    }

    if labels is not None:
        result["labels"] = torch.tensor(labels[:max_seq_length], dtype=torch.long)

    return result


class PackedDataset(Dataset):
    """
    Dataset that packs sequences on-the-fly or pre-packs them.

    For better efficiency with large datasets, pre-pack sequences
    and save them to disk.
    """

    def __init__(
        self,
        sequences: list[dict[str, torch.Tensor]],
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        pre_pack: bool = True,
    ):
        """
        Initialize packed dataset.

        Args:
            sequences: Original sequences to pack
            max_seq_length: Maximum packed sequence length
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
            pre_pack: Whether to pack all sequences upfront
        """
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if pre_pack:
            self.packed_sequences = pack_sequences(
                sequences,
                max_seq_length=max_seq_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            logger.info(
                f"Packed {len(sequences)} sequences into {len(self.packed_sequences)} "
                f"(compression ratio: {len(sequences) / len(self.packed_sequences):.2f}x)"
            )
        else:
            self.packed_sequences = sequences

    def __len__(self) -> int:
        return len(self.packed_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.packed_sequences[idx]


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset that packs sequences on-the-fly.

    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        data_iterator: Iterable[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 2048,
        buffer_size: int = 1000,
    ):
        """
        Initialize streaming packed dataset.

        Args:
            data_iterator: Iterator over raw data samples
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            buffer_size: Number of sequences to buffer before packing
        """
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 2) or 2

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[dict[str, torch.Tensor]] = []

        for sample in self.data_iterator:
            # Tokenize sample
            if isinstance(sample, str):
                tokenized = self.tokenizer(
                    sample,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
            else:
                tokenized = sample

            buffer.append(tokenized)

            # Pack and yield when buffer is full
            if len(buffer) >= self.buffer_size:
                packed = pack_sequences(
                    buffer,
                    max_seq_length=self.max_seq_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
                for seq in packed:
                    yield seq
                buffer = []

        # Yield remaining
        if buffer:
            packed = pack_sequences(
                buffer,
                max_seq_length=self.max_seq_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            for seq in packed:
                yield seq


def compute_packing_efficiency(
    sequences: list[dict[str, torch.Tensor]],
    packed_sequences: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """
    Compute packing efficiency metrics.

    Args:
        sequences: Original sequences
        packed_sequences: Packed sequences

    Returns:
        Dictionary with efficiency metrics
    """
    original_tokens = sum(
        len(s["input_ids"]) if isinstance(s["input_ids"], list) else s["input_ids"].numel()
        for s in sequences
    )

    packed_tokens = sum(
        s["attention_mask"].sum().item() if isinstance(s["attention_mask"], torch.Tensor)
        else sum(s["attention_mask"])
        for s in packed_sequences
    )

    total_packed_slots = sum(
        len(s["input_ids"]) if isinstance(s["input_ids"], list) else s["input_ids"].numel()
        for s in packed_sequences
    )

    return {
        "compression_ratio": len(sequences) / len(packed_sequences),
        "token_efficiency": packed_tokens / total_packed_slots,
        "original_sequences": len(sequences),
        "packed_sequences": len(packed_sequences),
        "original_tokens": original_tokens,
        "packed_tokens": packed_tokens,
    }


def unpack_sequences(
    packed_sequence: dict[str, torch.Tensor],
    eos_token_id: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """
    Unpack a packed sequence back into individual sequences.

    Useful for evaluation or debugging.

    Args:
        packed_sequence: Packed sequence dictionary
        eos_token_id: EOS token ID used as separator

    Returns:
        List of individual sequences
    """
    input_ids = packed_sequence["input_ids"]
    attention_mask = packed_sequence["attention_mask"]

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.tolist()

    sequences = []
    current_ids = []
    current_mask = []

    for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
        if mask == 0:
            # Padding - end of content
            break

        if token_id == eos_token_id and current_ids:
            # End of sequence
            sequences.append({
                "input_ids": torch.tensor(current_ids, dtype=torch.long),
                "attention_mask": torch.tensor(current_mask, dtype=torch.long),
            })
            current_ids = []
            current_mask = []
        else:
            current_ids.append(token_id)
            current_mask.append(mask)

    # Add final sequence
    if current_ids:
        sequences.append({
            "input_ids": torch.tensor(current_ids, dtype=torch.long),
            "attention_mask": torch.tensor(current_mask, dtype=torch.long),
        })

    return sequences


__all__ = [
    "PackingConfig",
    "pack_sequences",
    "PackedDataset",
    "StreamingPackedDataset",
    "compute_packing_efficiency",
    "unpack_sequences",
]
