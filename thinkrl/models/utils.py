"""
Model Utilities
================

Utilities for model management including weight tying, EMA models,
and reference model handling.

Author: EllanorAI
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# =============================================================================
# Weight Tying and Reference Models
# =============================================================================


def create_reference_model(
    model: nn.Module,
    share_weights: bool = False,
) -> nn.Module:
    """
    Create a reference model for KL divergence computation.

    Args:
        model: Policy model to copy
        share_weights: If True, share weights instead of copying (saves memory)

    Returns:
        Frozen reference model

    Note:
        If share_weights=True, the reference model will reflect updates to
        the policy model. Call update_reference_model periodically if you
        want a static snapshot.
    """
    if share_weights:
        # Share weights - reference model is the same object
        ref_model = model
        logger.info("Created reference model with shared weights")
    else:
        # Deep copy - creates separate memory
        ref_model = copy.deepcopy(model)
        logger.info("Created reference model with copied weights")

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()
    return ref_model


def update_reference_model(
    ref_model: nn.Module,
    policy_model: nn.Module,
    tau: float = 1.0,
) -> None:
    """
    Update reference model from policy model.

    Args:
        ref_model: Reference model to update
        policy_model: Policy model to copy from
        tau: Interpolation factor (1.0 = full copy, 0.0 = no update)

    Note:
        tau < 1.0 creates a Polyak average, which can improve stability.
    """
    with torch.no_grad():
        for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
            if tau == 1.0:
                ref_param.data.copy_(policy_param.data)
            else:
                ref_param.data.mul_(1 - tau).add_(policy_param.data, alpha=tau)


def share_model_weights(
    source: nn.Module,
    target: nn.Module,
) -> None:
    """
    Share weights between models (makes target point to source's weights).

    This saves memory by avoiding weight duplication.

    Args:
        source: Source model
        target: Target model (will point to source's weights)
    """
    target.load_state_dict(source.state_dict())
    logger.info("Shared weights from source to target model")


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all model parameters.

    Args:
        model: Model to freeze

    Returns:
        Frozen model
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    Unfreeze all model parameters.

    Args:
        model: Model to unfreeze

    Returns:
        Unfrozen model
    """
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model


def freeze_layers(
    model: nn.Module,
    layer_names: list[str] | None = None,
    num_layers: int | None = None,
    freeze_embeddings: bool = False,
) -> nn.Module:
    """
    Freeze specific layers of a model.

    Args:
        model: Model to partially freeze
        layer_names: List of layer name patterns to freeze
        num_layers: Number of initial layers to freeze
        freeze_embeddings: Whether to freeze embedding layers

    Returns:
        Model with specified layers frozen
    """
    if layer_names is not None:
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    break

    if num_layers is not None:
        # Common pattern for transformer models
        layer_count = 0
        for name, param in model.named_parameters():
            if "layer" in name.lower() or "block" in name.lower():
                # Extract layer number
                import re

                match = re.search(r"(?:layers?|blocks?)[._]?(\d+)", name.lower())
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx < num_layers:
                        param.requires_grad = False

    if freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = False

    # Count frozen vs trainable
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    logger.info(f"Froze {frozen}/{total} parameters")

    return model


# =============================================================================
# EMA (Exponential Moving Average) Model
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains a running average of model weights for more stable
    evaluation and inference.

    Example:
        >>> ema = EMAModel(model, decay=0.999)
        >>> for batch in dataloader:
        ...     loss = train_step(model, batch)
        ...     ema.update()  # Update EMA weights
        >>> # Use EMA model for evaluation
        >>> with ema.average_parameters():
        ...     eval_loss = evaluate(model, eval_data)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        """
        Initialize EMA model.

        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower updates)
            update_after_step: Start EMA after this many steps
            update_every: Update EMA every N steps
        """
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every

        # Store shadow parameters
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.backup_params: dict[str, torch.Tensor] = {}

        self.step = 0
        self._init_shadow_params()

    def _init_shadow_params(self) -> None:
        """Initialize shadow parameters as copy of model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self) -> None:
        """Update EMA parameters."""
        self.step += 1

        if self.step < self.update_after_step:
            return

        if self.step % self.update_every != 0:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow_params:
                    shadow = self.shadow_params[name]
                    shadow.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def copy_to_model(self) -> None:
        """Copy EMA parameters to model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

    def store_model_params(self) -> None:
        """Store current model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.shadow_params:
                self.backup_params[name] = param.data.clone()

    def restore_model_params(self) -> None:
        """Restore stored model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup_params:
                    param.data.copy_(self.backup_params[name])

    def average_parameters(self):
        """Context manager to temporarily use EMA parameters."""
        return _EMAContext(self)

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "shadow_params": self.shadow_params,
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.shadow_params = state["shadow_params"]
        self.step = state.get("step", 0)


class _EMAContext:
    """Context manager for temporarily using EMA parameters."""

    def __init__(self, ema: EMAModel):
        self.ema = ema

    def __enter__(self):
        self.ema.store_model_params()
        self.ema.copy_to_model()
        return self

    def __exit__(self, *args):
        self.ema.restore_model_params()


# =============================================================================
# Model Analysis Utilities
# =============================================================================


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of a model's parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Get the dtype of a model's parameters."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def model_memory_footprint(model: nn.Module) -> dict[str, float]:
    """
    Estimate model memory footprint in MB.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with memory estimates
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "parameters_mb": param_bytes / (1024 * 1024),
        "buffers_mb": buffer_bytes / (1024 * 1024),
        "total_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
    }


def enable_gradient_checkpointing(model: nn.Module) -> bool:
    """
    Enable gradient checkpointing for memory efficiency.

    Args:
        model: Model to modify

    Returns:
        True if gradient checkpointing was enabled
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Enabled gradient checkpointing")
        return True

    logger.warning("Model does not support gradient checkpointing")
    return False


def disable_gradient_checkpointing(model: nn.Module) -> bool:
    """
    Disable gradient checkpointing.

    Args:
        model: Model to modify

    Returns:
        True if gradient checkpointing was disabled
    """
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing")
        return True

    return False


__all__ = [
    # Reference models
    "create_reference_model",
    "update_reference_model",
    "share_model_weights",
    # Freezing
    "freeze_model",
    "unfreeze_model",
    "freeze_layers",
    # EMA
    "EMAModel",
    # Analysis
    "count_parameters",
    "get_model_device",
    "get_model_dtype",
    "model_memory_footprint",
    # Gradient checkpointing
    "enable_gradient_checkpointing",
    "disable_gradient_checkpointing",
]
