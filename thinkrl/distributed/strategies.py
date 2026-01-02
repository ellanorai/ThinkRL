"""
DeepSpeed Strategies
====================

Configuration strategies for DeepSpeed ZeRO optimization.

ZeRO Stages:
- Stage 1: Optimizer state partitioning
- Stage 2: Optimizer state + gradient partitioning
- Stage 3: Full parameter, gradient, and optimizer state partitioning

Author: EllanorAI
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeepSpeedStrategy(ABC):
    """Base class for DeepSpeed configuration strategies."""

    # Common settings
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    train_batch_size: int | None = None  # Auto-computed if None
    train_micro_batch_size_per_gpu: int = 4

    # Mixed precision
    fp16_enabled: bool = False
    bf16_enabled: bool = True

    # Activation checkpointing
    activation_checkpointing: bool = False
    partition_activations: bool = False

    @abstractmethod
    def get_zero_config(self) -> dict[str, Any]:
        """Get ZeRO-specific configuration."""
        pass

    def get_config(self) -> dict[str, Any]:
        """Get complete DeepSpeed configuration."""
        config: dict[str, Any] = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": self.get_zero_config(),
        }

        # Mixed precision
        if self.bf16_enabled:
            config["bf16"] = {"enabled": True}
        elif self.fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }

        # Activation checkpointing
        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": False,
            }

        return config


@dataclass
class ZeRO1Strategy(DeepSpeedStrategy):
    """
    ZeRO Stage 1: Optimizer state partitioning.

    Memory savings: ~4x for optimizer states.
    Communication overhead: Minimal.
    Use when: Model fits in GPU memory, want optimizer state savings.
    """

    reduce_scatter: bool = True
    overlap_comm: bool = True

    def get_zero_config(self) -> dict[str, Any]:
        return {
            "stage": 1,
            "reduce_scatter": self.reduce_scatter,
            "overlap_comm": self.overlap_comm,
        }


@dataclass
class ZeRO2Strategy(DeepSpeedStrategy):
    """
    ZeRO Stage 2: Optimizer state + gradient partitioning.

    Memory savings: ~8x for optimizer states and gradients.
    Communication overhead: Low (all-reduce replaced with reduce-scatter).
    Use when: Model fits in GPU, want more memory for activations/batch size.
    """

    # Communication optimization
    overlap_comm: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000

    # CPU offloading
    offload_optimizer: bool = False
    offload_optimizer_device: str = "cpu"
    offload_optimizer_pin_memory: bool = True

    # Contiguous gradients
    contiguous_gradients: bool = True

    def get_zero_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "stage": 2,
            "overlap_comm": self.overlap_comm,
            "reduce_scatter": self.reduce_scatter,
            "reduce_bucket_size": self.reduce_bucket_size,
            "allgather_bucket_size": self.allgather_bucket_size,
            "contiguous_gradients": self.contiguous_gradients,
        }

        if self.offload_optimizer:
            config["offload_optimizer"] = {
                "device": self.offload_optimizer_device,
                "pin_memory": self.offload_optimizer_pin_memory,
            }

        return config


@dataclass
class ZeRO3Strategy(DeepSpeedStrategy):
    """
    ZeRO Stage 3: Full parameter sharding.

    Memory savings: Linear with GPU count (model partitioned across GPUs).
    Communication overhead: Higher (parameter gathering required for forward/backward).
    Use when: Model doesn't fit on single GPU, need massive scale.
    """

    # Communication optimization
    overlap_comm: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000

    # Prefetching
    stage3_prefetch_bucket_size: int = 500_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000

    # CPU/NVMe offloading
    offload_optimizer: bool = True
    offload_optimizer_device: str = "cpu"
    offload_optimizer_pin_memory: bool = True

    offload_param: bool = False
    offload_param_device: str = "cpu"
    offload_param_pin_memory: bool = True

    # Sub-group configuration
    sub_group_size: int = 1_000_000_000

    # Gather 16-bit weights on model save
    stage3_gather_16bit_weights_on_model_save: bool = True

    def get_zero_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "stage": 3,
            "overlap_comm": self.overlap_comm,
            "reduce_scatter": self.reduce_scatter,
            "reduce_bucket_size": self.reduce_bucket_size,
            "allgather_bucket_size": self.allgather_bucket_size,
            "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
            "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
            "stage3_max_live_parameters": self.stage3_max_live_parameters,
            "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
            "sub_group_size": self.sub_group_size,
            "stage3_gather_16bit_weights_on_model_save": self.stage3_gather_16bit_weights_on_model_save,
        }

        if self.offload_optimizer:
            config["offload_optimizer"] = {
                "device": self.offload_optimizer_device,
                "pin_memory": self.offload_optimizer_pin_memory,
            }

        if self.offload_param:
            config["offload_param"] = {
                "device": self.offload_param_device,
                "pin_memory": self.offload_param_pin_memory,
            }

        return config


# Strategy presets
STRATEGIES = {
    "zero1": ZeRO1Strategy,
    "zero2": ZeRO2Strategy,
    "zero3": ZeRO3Strategy,
    "zero2_offload": lambda: ZeRO2Strategy(offload_optimizer=True),
    "zero3_offload": lambda: ZeRO3Strategy(offload_optimizer=True, offload_param=True),
}


def get_strategy(name: str, **kwargs) -> DeepSpeedStrategy:
    """
    Get a DeepSpeed strategy by name.

    Args:
        name: Strategy name ("zero1", "zero2", "zero3", "zero2_offload", "zero3_offload")
        **kwargs: Override default strategy parameters

    Returns:
        Configured DeepSpeedStrategy instance

    Example:
        >>> strategy = get_strategy("zero2", offload_optimizer=True)
        >>> config = strategy.get_config()
    """
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")

    strategy_class = STRATEGIES[name]

    # Handle presets that return callables
    if callable(strategy_class) and not isinstance(strategy_class, type):
        strategy = strategy_class()
        # Apply kwargs to the returned instance
        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        return strategy

    return strategy_class(**kwargs)


__all__ = [
    "DeepSpeedStrategy",
    "ZeRO1Strategy",
    "ZeRO2Strategy",
    "ZeRO3Strategy",
    "STRATEGIES",
    "get_strategy",
]
