"""
ThinkRL Data Module
====================

Dataset handling, data loading, and sequence packing utilities.

Author: EllanorAI
"""

from .datasets import (
    BaseRLHFDataset,
    PreferenceDataset,
    RLHFDataset,
)
from .loaders import (
    create_rlhf_collate_fn,
    create_rlhf_dataloader,
)
from .packing import (
    PackedDataset,
    PackingConfig,
    StreamingPackedDataset,
    compute_packing_efficiency,
    pack_sequences,
    unpack_sequences,
)
from .processors import (
    DataProcessor,
    get_data_processor,
)


__all__ = [
    # Datasets
    "BaseRLHFDataset",
    "PreferenceDataset",
    "RLHFDataset",
    # Loaders
    "create_rlhf_dataloader",
    "create_rlhf_collate_fn",
    # Processors
    "DataProcessor",
    "get_data_processor",
    # Packing
    "PackingConfig",
    "pack_sequences",
    "PackedDataset",
    "StreamingPackedDataset",
    "compute_packing_efficiency",
    "unpack_sequences",
]
