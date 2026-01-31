"""
Mixed Precision Training Utilities
===================================

Helpers for BF16/FP16 mixed precision training with proper loss scaling.

Supports:
- BF16 (recommended for modern GPUs)
- FP16 with dynamic loss scaling
- Automatic dtype context management

Author: EllanorAI
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class PrecisionType(str, Enum):
    """Precision types for training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    # Precision type
    precision: PrecisionType = PrecisionType.BF16

    # FP16 loss scaling parameters
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # Gradient clipping (applied after unscaling for FP16)
    max_grad_norm: float | None = 1.0

    # Auto-detect best precision for hardware
    auto_detect: bool = True


class MixedPrecisionTrainer:
    """
    Mixed precision training helper.

    Handles BF16/FP16 training with proper loss scaling and gradient management.

    Example:
        >>> mp_trainer = MixedPrecisionTrainer(precision="bf16")
        >>> with mp_trainer.autocast():
        ...     outputs = model(inputs)
        ...     loss = criterion(outputs, targets)
        >>> mp_trainer.backward(loss)
        >>> mp_trainer.step(optimizer)
    """

    def __init__(
        self,
        precision: PrecisionType | str = PrecisionType.BF16,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_grad_norm: float | None = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize mixed precision trainer.

        Args:
            precision: Precision type ("fp32", "fp16", "bf16", "auto")
            init_scale: Initial loss scale for FP16
            growth_factor: Scale growth factor for FP16
            backoff_factor: Scale backoff factor for FP16
            growth_interval: Steps between scale growth attempts
            max_grad_norm: Maximum gradient norm for clipping
            device: Device type ("cuda" or "cpu")
        """
        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        # Auto-detect best precision
        if precision == PrecisionType.AUTO:
            precision = self._auto_detect_precision(device)

        self.precision = precision
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize loss scaler for FP16
        self.scaler = None
        self._grads_unscaled = False  # Track if gradients have been unscaled
        if precision == PrecisionType.FP16 and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )

        # Determine autocast dtype
        self._autocast_dtype = self._get_autocast_dtype()

        logger.info(
            f"Initialized MixedPrecisionTrainer with precision={precision.value}, "
            f"autocast_dtype={self._autocast_dtype}"
        )

    @classmethod
    def from_config(cls, config: MixedPrecisionConfig, device: str = "cuda") -> MixedPrecisionTrainer:
        """Create trainer from config."""
        return cls(
            precision=config.precision,
            init_scale=config.init_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval,
            max_grad_norm=config.max_grad_norm,
            device=device,
        )

    def _auto_detect_precision(self, device: str) -> PrecisionType:
        """Auto-detect best precision for the hardware."""
        if device == "cpu":
            return PrecisionType.FP32

        if not torch.cuda.is_available():
            return PrecisionType.FP32

        # Check for BF16 support (Ampere and later)
        if torch.cuda.is_bf16_supported():
            logger.info("BF16 supported, using BF16 precision")
            return PrecisionType.BF16

        # Check compute capability for FP16
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:  # Volta and later
            logger.info("FP16 supported (compute >= 7.0), using FP16 precision")
            return PrecisionType.FP16

        logger.info("Using FP32 precision (no mixed precision support)")
        return PrecisionType.FP32

    def _get_autocast_dtype(self) -> torch.dtype:
        """Get the appropriate autocast dtype."""
        if self.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.precision == PrecisionType.FP16:
            return torch.float16
        else:
            return torch.float32

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """
        Context manager for automatic mixed precision.

        Example:
            >>> with mp_trainer.autocast():
            ...     outputs = model(inputs)
            ...     loss = criterion(outputs, targets)
        """
        if self.precision == PrecisionType.FP32:
            yield
            return

        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.autocast(device_type=device_type, dtype=self._autocast_dtype):
            yield

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module | None = None,
        create_graph: bool = False,
    ) -> None:
        """
        Perform backward pass with loss scaling for FP16.

        Args:
            loss: Loss tensor
            model: Model (for gradient checkpointing compatibility)
            create_graph: Whether to create graph for higher-order gradients
        """
        if self.scaler is not None:
            # FP16 with loss scaling
            self.scaler.scale(loss).backward(create_graph=create_graph)
        else:
            # BF16 or FP32 - direct backward
            loss.backward(create_graph=create_graph)

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients for FP16 training.

        Call before gradient clipping or inspection.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float | None:
        """
        Clip gradients with proper handling for mixed precision.

        Args:
            model: Model with gradients
            optimizer: Optimizer (needed for FP16 unscaling)

        Returns:
            Total gradient norm before clipping, or None if no clipping
        """
        if self.max_grad_norm is None:
            return None

        # For FP16, unscale first if not already done
        if self.scaler is not None and optimizer is not None and not self._grads_unscaled:
            self.scaler.unscale_(optimizer)
            self._grads_unscaled = True

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_grad_norm,
        )

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module | None = None,
    ) -> None:
        """
        Perform optimizer step with proper scaling.

        Args:
            optimizer: Optimizer to step
            model: Model (for gradient clipping)
        """
        # Clip gradients if needed
        if model is not None and self.max_grad_norm is not None:
            self.clip_gradients(model, optimizer)

        if self.scaler is not None:
            # FP16 with scaled step
            self.scaler.step(optimizer)
            self.scaler.update()
            self._grads_unscaled = False  # Reset for next iteration
        else:
            # BF16 or FP32 - direct step
            optimizer.step()

    def zero_grad(self, optimizer: torch.optim.Optimizer, set_to_none: bool = True) -> None:
        """Zero gradients."""
        optimizer.zero_grad(set_to_none=set_to_none)

    def forward_backward_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """
        Combined forward-backward-step for convenience.

        Args:
            model: Model
            loss: Computed loss
            optimizer: Optimizer

        Returns:
            Dictionary with loss and grad_norm
        """
        # Backward
        self.backward(loss, model)

        # Get gradient norm
        grad_norm = self.clip_gradients(model, optimizer)

        # Step
        self.step(optimizer, model)

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
        }

    def get_loss_scale(self) -> float:
        """Get current loss scale (for FP16 only)."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        state = {
            "precision": self.precision.value,
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


def get_autocast_dtype(precision: str) -> torch.dtype:
    """Get autocast dtype for a precision string."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(precision.lower(), torch.float32)


def cast_model_to_dtype(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Cast model parameters to specified dtype."""
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model


def enable_tf32() -> None:
    """Enable TF32 for faster matmul on Ampere+ GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for matmul operations")


def disable_tf32() -> None:
    """Disable TF32."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


__all__ = [
    "PrecisionType",
    "MixedPrecisionConfig",
    "MixedPrecisionTrainer",
    "get_autocast_dtype",
    "cast_model_to_dtype",
    "enable_tf32",
    "disable_tf32",
]
