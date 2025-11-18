"""
ThinkRL Dataset Utilities
=========================

Helper functions for data processing, padding, masking, and batch preparation.
"""

import torch
from typing import List, Dict, Any, Optional, Union

class BatchEncoding(dict):
    """Holds the output of the tokenizer."""
    def __init__(self, data: Dict[str, Any], encoding: Any = None, tensor_type: str = "pt"):
        super().__init__(data)
        self.encoding = encoding
        self.tensor_type = tensor_type

    def to(self, device: Union[str, torch.device]) -> "BatchEncoding":
        """Move all tensors to device."""
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self

def pad_sequences(
    sequences: List[torch.Tensor], 
    padding_value: int = 0, 
    padding_side: str = "right"
) -> torch.Tensor:
    """Pad a list of sequences to the same length."""
    from torch.nn.utils.rnn import pad_sequence
    
    if not sequences:
        return torch.tensor([])

    if padding_side == "left":
        # Reverse, pad, reverse back
        sequences = [s.flip(0) for s in sequences]
        padded = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        return padded.flip(1)
    
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

def create_attention_mask(input_ids: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """Create attention mask from input_ids."""
    return (input_ids != padding_value).long()

def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create position IDs from attention mask."""
    # Cumulative sum to get positions, masked by attention_mask
    return torch.cumsum(attention_mask, dim=1) * attention_mask

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create a causal (lower triangular) mask for auto-regressive attention."""
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    return mask.view(1, 1, seq_len, seq_len)

def collate_batch(
    batch: List[Dict[str, Any]], 
    tokenizer: Any = None, 
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """Collate a list of samples into a batch."""
    if not batch:
        return {}
    
    keys = batch[0].keys()
    collated = {}
    
    # Determine padding value
    pad_token_id = 0
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id

    for key in keys:
        items = [b[key] for b in batch]
        
        if isinstance(items[0], torch.Tensor):
            # Check if all tensors have same shape (no padding needed)
            shapes = [x.shape for x in items]
            if all(s == shapes[0] for s in shapes):
                collated[key] = torch.stack(items)
            else:
                # Pad sequences
                if items[0].dim() == 1:
                     collated[key] = pad_sequences(items, padding_value=pad_token_id)
                else:
                     # Fallback: stack if possible or list
                     try:
                         collated[key] = torch.stack(items)
                     except Exception:
                        collated[key] = items
        elif isinstance(items[0], (int, float)):
             collated[key] = torch.tensor(items)
        else:
             collated[key] = items
             
    if device:
        collated = to_device(collated, device)
        
    return collated

def create_dataloader(
    dataset, 
    batch_size: int, 
    shuffle: bool = True, 
    collate_fn=None, 
    **kwargs
):
    """Create a PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    if collate_fn is None:
        collate_fn = collate_batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)

def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    if not isinstance(text, str):
        return str(text)
    return text.strip()

def truncate_sequence(
    sequence: Union[List, torch.Tensor], 
    max_length: int, 
    side: str = "right"
) -> Union[List, torch.Tensor]:
    """Truncate a sequence to max_length."""
    if len(sequence) <= max_length:
        return sequence
    
    if side == "right":
        return sequence[:max_length]
    else: # left
        return sequence[-max_length:]

def create_labels_for_clm(input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Create labels for Causal Language Modeling (usually same as input_ids)."""
    return input_ids.clone()

def mask_padding_in_loss(
    labels: torch.Tensor, 
    attention_mask: torch.Tensor, 
    ignore_index: int = -100
) -> torch.Tensor:
    """Mask padding tokens in labels so they don't contribute to loss."""
    labels = labels.clone()
    labels[attention_mask == 0] = ignore_index
    return labels

def split_batch(batch: Dict[str, Any], micro_batch_size: int) -> List[Dict[str, Any]]:
    """Split a large batch into smaller micro-batches."""
    # Find batch size from first tensor item
    batch_size = 0
    for v in batch.values():
        if isinstance(v, (torch.Tensor, list)):
            batch_size = len(v)
            break
            
    if batch_size == 0:
        return [batch]

    micro_batches = []
    for i in range(0, batch_size, micro_batch_size):
        micro_batch = {}
        for k, v in batch.items():
            if isinstance(v, (torch.Tensor, list)):
                micro_batch[k] = v[i : i + micro_batch_size]
            else:
                micro_batch[k] = v
        micro_batches.append(micro_batch)
    
    return micro_batches

def compute_sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute sequence lengths from attention mask."""
    return attention_mask.sum(dim=1)

def shuffle_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Shuffle items within a batch."""
    # Find batch size
    batch_size = 0
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.size(0)
            break
    
    if batch_size == 0:
        return batch

    indices = torch.randperm(batch_size)
    shuffled = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.size(0) == batch_size:
            shuffled[k] = v[indices]
        elif isinstance(v, list) and len(v) == batch_size:
            shuffled[k] = [v[i] for i in indices]
        else:
             shuffled[k] = v
    return shuffled

def to_device(batch: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """Move all tensors in batch to device."""
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        elif isinstance(v, dict):
             new_batch[k] = to_device(v, device)
        else:
            new_batch[k] = v
    return new_batch

def prepare_batch_for_training(batch: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """Prepare batch for training (move to device, etc.)."""
    return to_device(batch, device)