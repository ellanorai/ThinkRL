"""
ThinkRL Data Loaders
====================

Custom DataLoaders and Collate functions for RLHF.
"""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def create_rlhf_collate_fn(tokenizer, padding_side="right"):
    """
    Create a collate function that handles variable length sequences and padding.
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        keys = batch[0].keys()
        collated = {}

        padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for key in keys:
            if key in ["prompt_text", "prompt"]:  # Pass through strings
                collated[key] = [item[key] for item in batch]
                continue

            # Handle tensors (input_ids, attention_mask, etc.)
            if isinstance(batch[0][key], torch.Tensor):
                tensors = [item[key] for item in batch]

                # Determine appropriate padding value based on key
                if key == "attention_mask":
                    pad_value = 0  # Attention mask should be 0 for padding
                elif key == "labels":
                    pad_value = -100  # Labels use -100 to ignore in loss
                else:
                    pad_value = padding_value  # Use tokenizer pad_token_id for input_ids etc.

                # Pad sequences
                if padding_side == "left":
                    # Reverse, pad, reverse back for left padding
                    tensors_rev = [t.flip(0) for t in tensors]
                    padded_rev = pad_sequence(tensors_rev, batch_first=True, padding_value=pad_value)
                    collated[key] = padded_rev.flip(1)
                else:
                    collated[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            else:
                # Fallback for other types
                collated[key] = [item[key] for item in batch]

        return collated

    return collate_fn


class RLHFDataLoader(DataLoader):
    """
    A wrapper around standard DataLoader with RLHF-specific defaults.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        padding_side="right",
        **kwargs,
    ):
        collate_fn = create_rlhf_collate_fn(tokenizer, padding_side)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            **kwargs,
        )


def create_rlhf_dataloader(
    dataset,
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    padding_side: str = "right",
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs,
) -> RLHFDataLoader:
    """
    Factory function to create an RLHF DataLoader with sensible defaults.

    Args:
        dataset: The dataset to load
        tokenizer: Tokenizer for padding configuration
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        padding_side: Side to pad sequences ("left" or "right")
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        RLHFDataLoader instance
    """
    return RLHFDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        padding_side=padding_side,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )
