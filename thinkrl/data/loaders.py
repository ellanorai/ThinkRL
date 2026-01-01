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

                # Pad sequences
                if padding_side == "left":
                    # Reverse, pad, reverse back for left padding
                    tensors_rev = [t.flip(0) for t in tensors]
                    padded_rev = pad_sequence(tensors_rev, batch_first=True, padding_value=padding_value)
                    collated[key] = padded_rev.flip(1)
                else:
                    collated[key] = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
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
