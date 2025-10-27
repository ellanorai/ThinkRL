"""
ThinkRL Data Utilities
======================

Data utilities for ThinkRL including:
- Data loading and preprocessing
- Batch collation and padding
- Token masking and attention masks
- Data augmentation
- Efficient data handling for RLHF

Author: Archit Sood @ EllanorAI
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import torch
import numpy as np

# Optional dependencies
try:
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    _TORCH_DATA_AVAILABLE = True
except ImportError:
    _TORCH_DATA_AVAILABLE = False
    warnings.warn("torch.utils.data not available")

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = np  # Fallback to numpy

logger = logging.getLogger(__name__)


@dataclass
class BatchEncoding:
    """
    Container for batch of encoded sequences.
    
    Attributes:
        input_ids: Token IDs
        attention_mask: Attention mask (1 for real tokens, 0 for padding)
        labels: Target labels (optional)
        token_type_ids: Token type IDs (optional)
        position_ids: Position IDs (optional)
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    
    def to(self, device: Union[str, torch.device]) -> "BatchEncoding":
        """Move batch to device."""
        return BatchEncoding(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            token_type_ids=self.token_type_ids.to(device) if self.token_type_ids is not None else None,
            position_ids=self.position_ids.to(device) if self.position_ids is not None else None,
        )
    
    def __getitem__(self, key: str) -> torch.Tensor:
        """Access attributes like a dictionary."""
        return getattr(self, key)
    
    def keys(self) -> List[str]:
        """Get all non-None keys."""
        return [k for k in ["input_ids", "attention_mask", "labels", "token_type_ids", "position_ids"] 
                if getattr(self, k) is not None]


def pad_sequences(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: Optional[int] = None,
    return_tensors: bool = True
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences to pad
        padding_value: Value to use for padding
        padding_side: "right" or "left" padding
        max_length: Maximum length (if None, uses longest sequence)
        return_tensors: Whether to return as tensor or list
        
    Returns:
        Padded sequences
        
    Example:
        ```python
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3, 4, 5])
        ]
        padded = pad_sequences(sequences, padding_value=0)
        # Shape: (3, 5) - padded to longest sequence
        ```
    """
    if not sequences:
        return torch.tensor([]) if return_tensors else []
    
    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        seq_len = len(seq)
        
        if seq_len > max_length:
            # Truncate
            padded_seq = seq[:max_length]
        elif seq_len < max_length:
            # Pad
            padding_length = max_length - seq_len
            padding = torch.full((padding_length,), padding_value, dtype=seq.dtype, device=seq.device)
            
            if padding_side == "right":
                padded_seq = torch.cat([seq, padding])
            else:  # left
                padded_seq = torch.cat([padding, seq])
        else:
            padded_seq = seq
        
        padded_sequences.append(padded_seq)
    
    if return_tensors:
        return torch.stack(padded_sequences)
    return padded_sequences


def create_attention_mask(
    input_ids: torch.Tensor,
    padding_value: int = 0
) -> torch.Tensor:
    """
    Create attention mask from input IDs.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        padding_value: Padding token ID
        
    Returns:
        Attention mask of shape (batch_size, seq_len)
        
    Example:
        ```python
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_attention_mask(input_ids, padding_value=0)
        # tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        ```
    """
    return (input_ids != padding_value).long()


def create_position_ids(
    attention_mask: torch.Tensor,
    past_length: int = 0
) -> torch.Tensor:
    """
    Create position IDs from attention mask.
    
    Args:
        attention_mask: Attention mask of shape (batch_size, seq_len)
        past_length: Length of past context (for autoregressive generation)
        
    Returns:
        Position IDs of shape (batch_size, seq_len)
        
    Example:
        ```python
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        position_ids = create_position_ids(attention_mask)
        # tensor([[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]])
        ```
    """
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids + past_length


def create_causal_mask(
    seq_length: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.
    
    Args:
        seq_length: Sequence length
        device: Device to create mask on
        dtype: Data type of mask
        
    Returns:
        Causal mask of shape (seq_length, seq_length)
        
    Example:
        ```python
        mask = create_causal_mask(4)
        # tensor([[True, False, False, False],
        #         [True,  True, False, False],
        #         [True,  True,  True, False],
        #         [True,  True,  True,  True]])
        ```
    """
    mask = torch.triu(
        torch.ones(seq_length, seq_length, device=device, dtype=dtype),
        diagonal=1
    )
    return mask == 0


def collate_batch(
    batch: List[Dict[str, Any]],
    padding_value: int = 0,
    max_length: Optional[int] = None,
    return_tensors: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of samples into tensors.
    
    Args:
        batch: List of sample dictionaries
        padding_value: Value for padding
        max_length: Maximum sequence length
        return_tensors: Whether to return tensors
        
    Returns:
        Collated batch dictionary
        
    Example:
        ```python
        batch = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [1, 2], "labels": [2, 3]},
        ]
        collated = collate_batch(batch)
        ```
    """
    if not batch:
        return {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        # Get all values for this key
        values = [sample[key] for sample in batch]
        
        # Skip None values
        values = [v for v in values if v is not None]
        if not values:
            continue
        
        # Convert to tensors if needed
        if not isinstance(values[0], torch.Tensor):
            values = [torch.tensor(v) for v in values]
        
        # Pad sequences
        if values[0].dim() >= 1:  # Sequence data
            padded = pad_sequences(
                values,
                padding_value=padding_value,
                max_length=max_length,
                return_tensors=return_tensors
            )
            collated[key] = padded
        else:  # Scalar data
            collated[key] = torch.stack(values) if return_tensors else values
    
    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults for RLHF training.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if distributed)
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop incomplete last batch
        distributed: Whether to use distributed sampler
        rank: Process rank (required if distributed=True)
        world_size: Number of processes (required if distributed=True)
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        Configured DataLoader
        
    Example:
        ```python
        dataloader = create_dataloader(
            dataset=my_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        for batch in dataloader:
            # Train model
            pass
        ```
    """
    if not _TORCH_DATA_AVAILABLE:
        raise ImportError("torch.utils.data is required for create_dataloader")
    
    # Use distributed sampler if needed
    sampler = None
    if distributed:
        if rank is None or world_size is None:
            raise ValueError("rank and world_size are required for distributed training")
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling
    
    # Default collate function
    if collate_fn is None:
        collate_fn = collate_batch
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        **kwargs
    )
    
    return dataloader


def preprocess_text(
    text: str,
    lowercase: bool = False,
    strip: bool = True,
    remove_extra_spaces: bool = True
) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        strip: Whether to strip leading/trailing whitespace
        remove_extra_spaces: Whether to remove extra spaces
        
    Returns:
        Preprocessed text
        
    Example:
        ```python
        text = "  Hello   World!  "
        cleaned = preprocess_text(text, lowercase=True)
        # "hello world!"
        ```
    """
    if strip:
        text = text.strip()
    
    if remove_extra_spaces:
        text = " ".join(text.split())
    
    if lowercase:
        text = text.lower()
    
    return text


def truncate_sequence(
    sequence: Union[List, torch.Tensor],
    max_length: int,
    truncation_side: str = "right"
) -> Union[List, torch.Tensor]:
    """
    Truncate sequence to maximum length.
    
    Args:
        sequence: Sequence to truncate
        max_length: Maximum length
        truncation_side: "right" or "left" truncation
        
    Returns:
        Truncated sequence
        
    Example:
        ```python
        seq = [1, 2, 3, 4, 5]
        truncated = truncate_sequence(seq, max_length=3, truncation_side="right")
        # [1, 2, 3]
        ```
    """
    if len(sequence) <= max_length:
        return sequence
    
    if truncation_side == "right":
        return sequence[:max_length]
    else:  # left
        return sequence[-max_length:]


def create_labels_for_clm(
    input_ids: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Create labels for causal language modeling (next token prediction).
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Labels of shape (batch_size, seq_len)
        
    Example:
        ```python
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
        labels = create_labels_for_clm(input_ids)
        # tensor([[2, 3, 4, 0, -100, -100]])
        ```
    """
    # Shift input_ids to create labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    
    # Mask padding tokens
    labels[input_ids == 0] = ignore_index
    
    return labels


def mask_padding_in_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    padding_value: int = 0,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Mask padding tokens in labels for loss computation.
    
    Args:
        logits: Model logits
        labels: Target labels
        padding_value: Padding token ID
        ignore_index: Index to use for padding in labels
        
    Returns:
        Masked labels
        
    Example:
        ```python
        labels = torch.tensor([[1, 2, 3, 0, 0]])
        masked_labels = mask_padding_in_loss(logits, labels)
        # tensor([[1, 2, 3, -100, -100]])
        ```
    """
    labels = labels.clone()
    labels[labels == padding_value] = ignore_index
    return labels


def split_batch(
    batch: Dict[str, torch.Tensor],
    num_splits: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Split a batch into smaller chunks (useful for gradient accumulation).
    
    Args:
        batch: Batch dictionary
        num_splits: Number of splits
        
    Returns:
        List of split batches
        
    Example:
        ```python
        batch = {"input_ids": torch.randn(16, 128)}
        splits = split_batch(batch, num_splits=4)
        # 4 batches of size 4 each
        ```
    """
    batch_size = next(iter(batch.values())).size(0)
    split_size = batch_size // num_splits
    
    if batch_size % num_splits != 0:
        logger.warning(f"Batch size {batch_size} not evenly divisible by {num_splits}")
    
    splits = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < num_splits - 1 else batch_size
        
        split = {
            key: value[start_idx:end_idx]
            for key, value in batch.items()
        }
        splits.append(split)
    
    return splits


def compute_sequence_lengths(
    input_ids: torch.Tensor,
    padding_value: int = 0
) -> torch.Tensor:
    """
    Compute actual sequence lengths (excluding padding).
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        padding_value: Padding token ID
        
    Returns:
        Sequence lengths of shape (batch_size,)
        
    Example:
        ```python
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        lengths = compute_sequence_lengths(input_ids)
        # tensor([3, 2])
        ```
    """
    return (input_ids != padding_value).sum(dim=-1)


def shuffle_batch(
    batch: Dict[str, torch.Tensor],
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Shuffle examples in a batch (while maintaining correspondence).
    
    Args:
        batch: Batch dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Shuffled batch
        
    Example:
        ```python
        batch = {"input_ids": torch.randn(16, 128), "labels": torch.randn(16)}
        shuffled = shuffle_batch(batch, seed=42)
        ```
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    batch_size = next(iter(batch.values())).size(0)
    indices = torch.randperm(batch_size, generator=generator)
    
    return {
        key: value[indices]
        for key, value in batch.items()
    }


def to_device(
    batch: Union[Dict[str, torch.Tensor], torch.Tensor, List],
    device: Union[str, torch.device]
) -> Union[Dict[str, torch.Tensor], torch.Tensor, List]:
    """
    Move batch to device (handles nested structures).
    
    Args:
        batch: Batch to move (dict, tensor, or list)
        device: Target device
        
    Returns:
        Batch on device
        
    Example:
        ```python
        batch = {"input_ids": torch.randn(4, 128), "labels": torch.randn(4)}
        batch_gpu = to_device(batch, "cuda")
        ```
    """
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [to_device(item, device) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(to_device(item, device) for item in batch)
    else:
        return batch


def prepare_batch_for_training(
    batch: Dict[str, Any],
    device: Union[str, torch.device],
    create_labels: bool = True,
    ignore_index: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Prepare batch for training (move to device, create labels, etc.).
    
    Args:
        batch: Input batch
        device: Target device
        create_labels: Whether to create labels from input_ids
        ignore_index: Index to ignore in loss
        
    Returns:
        Prepared batch
        
    Example:
        ```python
        batch = {"input_ids": [[1, 2, 3, 4]]}
        prepared = prepare_batch_for_training(batch, device="cuda")
        # Has input_ids, attention_mask, labels
        ```
    """
    # Convert to tensors if needed
    if "input_ids" in batch and not isinstance(batch["input_ids"], torch.Tensor):
        batch["input_ids"] = torch.tensor(batch["input_ids"])
    
    # Create attention mask if not present
    if "attention_mask" not in batch and "input_ids" in batch:
        batch["attention_mask"] = create_attention_mask(batch["input_ids"])
    
    # Create labels if requested
    if create_labels and "labels" not in batch and "input_ids" in batch:
        batch["labels"] = create_labels_for_clm(batch["input_ids"], ignore_index)
    
    # Move to device
    batch = to_device(batch, device)
    
    return batch


# Public API
__all__ = [
    "BatchEncoding",
    "pad_sequences",
    "create_attention_mask",
    "create_position_ids",
    "create_causal_mask",
    "collate_batch",
    "create_dataloader",
    "preprocess_text",
    "truncate_sequence",
    "create_labels_for_clm",
    "mask_padding_in_loss",
    "split_batch",
    "compute_sequence_lengths",
    "shuffle_batch",
    "to_device",
    "prepare_batch_for_training",
]