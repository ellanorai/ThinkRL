"""
LoRA Configuration and Utilities
=================================

First-class LoRA support for efficient fine-tuning of language models.

Features:
- Pre-configured targets for common architectures (Llama, Qwen, Mistral)
- Simple injection and merging APIs
- DeepSpeed ZeRO-3 compatibility
- Trainable parameter tracking

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import torch.nn as nn


# Check PEFT availability
try:
    from peft import (
        LoraConfig as PeftLoraConfig,
    )
    from peft import (
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    PeftLoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# Architecture-specific target modules
ARCHITECTURE_TARGETS = {
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mistral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen": [
        "c_attn",
        "c_proj",
        "w1",
        "w2",
    ],
    "qwen2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "phi": [
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2",
    ],
    "gemma": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bloom": [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    # Default fallback
    "default": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
}


@dataclass
class LoRAConfig:
    """
    LoRA configuration with sensible defaults.

    Aligned with the peft library but with architecture-aware presets.
    """

    # LoRA hyperparameters
    r: int = 8  # Rank of the update matrices
    lora_alpha: int = 16  # Alpha parameter for scaling
    lora_dropout: float = 0.05  # Dropout probability
    target_modules: list[str] | None = None  # Modules to apply LoRA to
    modules_to_save: list[str] | None = None  # Additional modules to train

    # Task type
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, TOKEN_CLS

    # Bias handling
    bias: str = "none"  # "none", "all", "lora_only"

    # Advanced options
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora: bool = False  # Weight-decomposed LoRA

    # Quantization compatibility
    prepare_for_kbit: bool = False

    @classmethod
    def for_architecture(cls, architecture: str, r: int = 8, **kwargs) -> LoRAConfig:
        """
        Create LoRA config for a specific architecture.

        Args:
            architecture: Model architecture ("llama", "qwen", "mistral", etc.)
            r: LoRA rank
            **kwargs: Additional config overrides

        Returns:
            LoRAConfig with appropriate target modules
        """
        arch_lower = architecture.lower()

        # Find matching architecture - prioritize exact matches first
        target_modules = None

        # First try exact match
        if arch_lower in ARCHITECTURE_TARGETS:
            target_modules = ARCHITECTURE_TARGETS[arch_lower]
        else:
            # Then try substring matching, prioritizing longer matches
            matches = []
            for arch_name, modules in ARCHITECTURE_TARGETS.items():
                if arch_name in arch_lower or arch_lower in arch_name:
                    matches.append((arch_name, modules))

            # Sort by length descending to prioritize more specific matches
            if matches:
                matches.sort(key=lambda x: len(x[0]), reverse=True)
                target_modules = matches[0][1]

        if target_modules is None:
            target_modules = ARCHITECTURE_TARGETS["default"]
            logger.warning(f"Unknown architecture '{architecture}', using default targets: {target_modules}")

        return cls(
            r=r,
            lora_alpha=r * 2,  # Common practice: alpha = 2 * r
            target_modules=target_modules,
            **kwargs,
        )

    @classmethod
    def for_llama(cls, r: int = 8, **kwargs) -> LoRAConfig:
        """LoRA config for Llama models."""
        return cls.for_architecture("llama", r=r, **kwargs)

    @classmethod
    def for_qwen(cls, r: int = 8, **kwargs) -> LoRAConfig:
        """LoRA config for Qwen models."""
        return cls.for_architecture("qwen", r=r, **kwargs)

    @classmethod
    def for_qwen2(cls, r: int = 8, **kwargs) -> LoRAConfig:
        """LoRA config for Qwen2 models."""
        return cls.for_architecture("qwen2", r=r, **kwargs)

    @classmethod
    def for_mistral(cls, r: int = 8, **kwargs) -> LoRAConfig:
        """LoRA config for Mistral models."""
        return cls.for_architecture("mistral", r=r, **kwargs)

    def to_peft_config(self) -> PeftLoraConfig:
        """Convert to peft library LoraConfig."""
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required. Install with: pip install peft")

        # Map task type string to enum
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
        }

        task_type = task_type_map.get(self.task_type.upper(), TaskType.CAUSAL_LM)

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            task_type=task_type,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )


def is_peft_available() -> bool:
    """Check if peft library is available."""
    return PEFT_AVAILABLE


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Inject LoRA adapters into model.

    Args:
        model: Model to modify
        config: LoRA configuration

    Returns:
        PeftModel with LoRA adapters

    Example:
        >>> config = LoRAConfig.for_llama(r=16)
        >>> model = inject_lora(model, config)
        >>> print(get_trainable_parameters(model))
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required. Install with: pip install peft")

    # Prepare model for quantized training if needed
    if config.prepare_for_kbit and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    # Convert config and apply
    peft_config = config.to_peft_config()
    peft_model = get_peft_model(model, peft_config)

    # Log parameter count
    trainable, frozen = _count_parameters(peft_model)
    total = trainable + frozen
    logger.info(
        f"Injected LoRA adapters: "
        f"{trainable:,} trainable / {total:,} total parameters "
        f"({100 * trainable / total if total > 0 else 0:.2f}%)"
    )

    return peft_model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model.

    After merging, the model can be saved without the adapter overhead.

    Args:
        model: PeftModel with LoRA adapters

    Returns:
        Model with merged weights
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required")

    if not hasattr(model, "merge_and_unload"):
        logger.warning("Model does not have merge_and_unload method, returning as-is")
        return model

    merged_model = model.merge_and_unload()
    logger.info("Merged LoRA weights into base model")
    return merged_model


def unload_lora(model: nn.Module) -> nn.Module:
    """
    Remove LoRA adapters without merging.

    Args:
        model: PeftModel with LoRA adapters

    Returns:
        Base model without adapters
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required")

    if hasattr(model, "unload"):
        return model.unload()

    # Fallback: get base model
    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "model"):
            return model.base_model.model
        return model.base_model

    return model


def get_trainable_parameters(model: nn.Module) -> dict[str, int]:
    """
    Get trainable parameter counts.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with trainable, frozen, and total counts
    """
    trainable, frozen = _count_parameters(model)
    total = trainable + frozen

    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and frozen parameters."""
    trainable = 0
    frozen = 0

    for param in model.parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable += num_params
        else:
            frozen += num_params

    return trainable, frozen


def get_lora_config_for_model(model: nn.Module, r: int = 8, **kwargs) -> LoRAConfig:
    """
    Auto-detect model architecture and return appropriate LoRA config.

    Args:
        model: Model to analyze
        r: LoRA rank
        **kwargs: Additional config options

    Returns:
        LoRAConfig for the detected architecture
    """
    # Try to detect architecture from model config
    if hasattr(model, "config"):
        config = model.config

        # Check model type
        if hasattr(config, "model_type") and config.model_type is not None:
            model_type = config.model_type.lower()

            # Map common model types - order matters for specificity
            arch_map = [
                ("qwen2", "qwen2"),  # Check qwen2 before qwen
                ("qwen", "qwen"),
                ("llama", "llama"),
                ("mistral", "mistral"),
                ("mixtral", "mistral"),
                ("phi3", "phi"),
                ("phi", "phi"),
                ("gemma2", "gemma"),
                ("gemma", "gemma"),
                ("bloom", "bloom"),
            ]

            for key, arch in arch_map:
                if key in model_type:
                    return LoRAConfig.for_architecture(arch, r=r, **kwargs)

        # Check architectures list
        if hasattr(config, "architectures") and config.architectures:
            arch_name = config.architectures[0].lower()
            # Sort keys by length descending for more specific matches first
            sorted_keys = sorted(ARCHITECTURE_TARGETS.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in arch_name:
                    return LoRAConfig.for_architecture(key, r=r, **kwargs)

    # Default fallback
    logger.warning("Could not detect model architecture, using default LoRA config")
    return LoRAConfig(r=r, **kwargs)


def prepare_model_for_deepspeed_lora(
    model: nn.Module,
    lora_config: LoRAConfig,
    zero_stage: int = 2,
) -> nn.Module:
    """
    Prepare model for training with DeepSpeed + LoRA.

    Handles the complexity of combining ZeRO optimization with LoRA adapters.

    Args:
        model: Base model
        lora_config: LoRA configuration
        zero_stage: DeepSpeed ZeRO stage (1, 2, or 3)

    Returns:
        Model ready for DeepSpeed training
    """
    # Inject LoRA first
    model = inject_lora(model, lora_config)

    # For ZeRO-3, we need special handling
    if zero_stage == 3:
        # Ensure LoRA parameters are not sharded incorrectly
        # This is handled automatically by peft + deepspeed in most cases
        logger.info("Prepared model for ZeRO-3 + LoRA training")

    return model


__all__ = [
    "PEFT_AVAILABLE",
    "ARCHITECTURE_TARGETS",
    "LoRAConfig",
    "is_peft_available",
    "inject_lora",
    "merge_lora_weights",
    "unload_lora",
    "get_trainable_parameters",
    "get_lora_config_for_model",
    "prepare_model_for_deepspeed_lora",
]
