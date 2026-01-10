"""
ThinkRL Model Factory
======================

Factory functions for creating RLHF models.
Aligned with OpenRLHF patterns.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging

import torch.nn as nn

from .loader import (
    create_reference_model,
    get_actor_model,
    get_llm_for_sequence_regression,
)


logger = logging.getLogger(__name__)


# Re-export these for backward compatibility
__all__ = [
    "get_llm_for_sequence_regression",
    "get_actor_model",
    "create_reference_model",
    "compute_model_size",
]


def compute_model_size(model: nn.Module) -> dict[str, int | float]:
    """
    Compute model size statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
    }
