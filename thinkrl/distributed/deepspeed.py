"""
DeepSpeed Engine Wrapper
========================

Unified interface for DeepSpeed initialization and management.

Provides:
- Simple engine initialization
- Checkpoint saving/loading with ZeRO compatibility
- Gradient accumulation handling
- Mixed precision training

Author: EllanorAI
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn


# Check DeepSpeed availability
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine as _DSEngine

    DEEPSPEED_AVAILABLE = True
except ImportError:
    deepspeed = None  # type: ignore
    _DSEngine = None
    DEEPSPEED_AVAILABLE = False


if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

    from thinkrl.distributed.strategies import DeepSpeedStrategy


logger = logging.getLogger(__name__)


class DeepSpeedEngine:
    """
    Wrapper around DeepSpeed engine for RLHF training.

    Handles:
    - Engine initialization with strategy
    - Forward/backward passes
    - Gradient accumulation
    - Checkpointing (ZeRO-compatible)
    - Weight synchronization

    Example:
        >>> from thinkrl.distributed import ZeRO2Strategy, DeepSpeedEngine
        >>> strategy = ZeRO2Strategy(gradient_accumulation_steps=4)
        >>> engine = DeepSpeedEngine(model, strategy)
        >>> for batch in dataloader:
        ...     loss = engine.forward(batch)
        ...     engine.backward(loss)
        ...     engine.step()
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: DeepSpeedStrategy,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        model_parameters: list | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        """
        Initialize DeepSpeed engine.

        Args:
            model: PyTorch model to wrap
            strategy: DeepSpeed strategy (ZeRO1, ZeRO2, ZeRO3)
            optimizer: Optional optimizer (DeepSpeed can create one)
            scheduler: Optional LR scheduler
            model_parameters: Parameters to optimize (default: all model params)
            config_overrides: Override strategy config values
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError(
                "DeepSpeed is not installed. Install with: pip install deepspeed"
            )

        self.model = model
        self.strategy = strategy
        self._step_count = 0

        # Build config
        config = strategy.get_config()
        if config_overrides:
            config.update(config_overrides)

        # Initialize DeepSpeed
        self._engine, self._optimizer, _, self._scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=model_parameters or list(model.parameters()),
            config=config,
        )

        logger.info(
            f"Initialized DeepSpeed engine "
            f"(ZeRO stage {config['zero_optimization']['stage']}, "
            f"world_size={self._engine.world_size})"
        )

    @property
    def module(self) -> nn.Module:
        """Get the underlying model (unwrapped from DDP/DeepSpeed)."""
        return self._engine.module

    @property
    def optimizer(self):
        """Get the optimizer."""
        return self._optimizer

    @property
    def scheduler(self):
        """Get the LR scheduler."""
        return self._scheduler

    @property
    def global_step(self) -> int:
        """Get global training step."""
        return self._engine.global_steps

    @property
    def local_rank(self) -> int:
        """Get local rank."""
        return self._engine.local_rank

    @property
    def world_size(self) -> int:
        """Get world size."""
        return self._engine.world_size

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model."""
        return self._engine(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient accumulation handling."""
        self._engine.backward(loss)

    def step(self) -> None:
        """Optimizer step (handles gradient accumulation internally)."""
        self._engine.step()
        self._step_count += 1

    def zero_grad(self) -> None:
        """Zero gradients."""
        self._engine.zero_grad()

    def train(self) -> None:
        """Set model to training mode."""
        self._engine.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self._engine.eval()

    def save_checkpoint(
        self,
        save_dir: str | Path,
        tag: str | None = None,
        client_state: dict[str, Any] | None = None,
        save_latest: bool = True,
    ) -> str:
        """
        Save checkpoint (ZeRO-compatible).

        Args:
            save_dir: Directory to save checkpoint
            tag: Checkpoint tag/name (default: step number)
            client_state: Additional state to save
            save_latest: Whether to save as latest checkpoint

        Returns:
            Path to saved checkpoint
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if tag is None:
            tag = f"step_{self.global_step}"

        self._engine.save_checkpoint(
            save_dir=str(save_dir),
            tag=tag,
            client_state=client_state or {},
            save_latest=save_latest,
        )

        checkpoint_path = save_dir / tag
        logger.info(f"Saved DeepSpeed checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        load_dir: str | Path,
        tag: str | None = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> dict[str, Any]:
        """
        Load checkpoint (ZeRO-compatible).

        Args:
            load_dir: Directory containing checkpoint
            tag: Checkpoint tag to load (default: latest)
            load_optimizer_states: Whether to load optimizer state
            load_lr_scheduler_states: Whether to load scheduler state

        Returns:
            Client state dictionary
        """
        load_dir = Path(load_dir)

        _, client_state = self._engine.load_checkpoint(
            load_dir=str(load_dir),
            tag=tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )

        logger.info(f"Loaded DeepSpeed checkpoint from: {load_dir}")
        return client_state or {}

    def gather_16bit_weights(self) -> dict[str, torch.Tensor]:
        """
        Gather 16-bit weights from ZeRO-3 partitions.

        Useful for saving HuggingFace-compatible checkpoints.
        """
        if hasattr(self._engine, "module"):
            return self._engine.module.state_dict()
        return self._engine.state_dict()

    def save_hf_checkpoint(
        self,
        save_path: str | Path,
        tokenizer=None,
    ) -> None:
        """
        Save HuggingFace-compatible checkpoint.

        Args:
            save_path: Directory to save
            tokenizer: Optional tokenizer to save alongside
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Gather weights (handles ZeRO-3)
        state_dict = self.gather_16bit_weights()

        # Save model
        if hasattr(self.module, "save_pretrained"):
            self.module.save_pretrained(
                save_path,
                state_dict=state_dict,
                safe_serialization=True,
            )
        else:
            torch.save(state_dict, save_path / "pytorch_model.bin")

        # Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)

        logger.info(f"Saved HuggingFace checkpoint: {save_path}")

    def __call__(self, *args, **kwargs) -> Any:
        """Alias for forward."""
        return self.forward(*args, **kwargs)


def create_deepspeed_config(
    strategy: str = "zero2",
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    bf16: bool = True,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """
    Create DeepSpeed config dictionary.

    Convenience function for creating configs without strategy classes.

    Args:
        strategy: ZeRO strategy ("zero1", "zero2", "zero3")
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        gradient_clipping: Max gradient norm
        bf16: Use BF16 mixed precision
        offload_optimizer: Offload optimizer to CPU
        offload_param: Offload parameters to CPU (ZeRO-3 only)
        **kwargs: Additional config options

    Returns:
        DeepSpeed configuration dictionary
    """
    from thinkrl.distributed.strategies import get_strategy

    strat = get_strategy(
        strategy,
        train_micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=gradient_clipping,
        bf16_enabled=bf16,
        **kwargs,
    )

    # Handle offloading
    if hasattr(strat, "offload_optimizer"):
        strat.offload_optimizer = offload_optimizer
    if hasattr(strat, "offload_param"):
        strat.offload_param = offload_param

    return strat.get_config()


def init_deepspeed(
    model: nn.Module,
    strategy: str = "zero2",
    optimizer: Optimizer | None = None,
    **config_kwargs,
) -> DeepSpeedEngine:
    """
    Initialize DeepSpeed with sensible defaults.

    Convenience function for quick DeepSpeed setup.

    Args:
        model: Model to wrap
        strategy: Strategy name ("zero1", "zero2", "zero3")
        optimizer: Optional optimizer
        **config_kwargs: Additional config options

    Returns:
        DeepSpeedEngine instance

    Example:
        >>> engine = init_deepspeed(model, "zero2", bf16=True)
        >>> for batch in dataloader:
        ...     loss = engine(batch)
        ...     engine.backward(loss)
        ...     engine.step()
    """
    from thinkrl.distributed.strategies import get_strategy

    strat = get_strategy(strategy, **config_kwargs)
    return DeepSpeedEngine(model, strat, optimizer=optimizer)


def save_deepspeed_config(config: dict[str, Any], path: str | Path) -> None:
    """Save DeepSpeed config to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_deepspeed_config(path: str | Path) -> dict[str, Any]:
    """Load DeepSpeed config from JSON file."""
    with open(path) as f:
        return json.load(f)


__all__ = [
    "DEEPSPEED_AVAILABLE",
    "DeepSpeedEngine",
    "create_deepspeed_config",
    "init_deepspeed",
    "save_deepspeed_config",
    "load_deepspeed_config",
]
