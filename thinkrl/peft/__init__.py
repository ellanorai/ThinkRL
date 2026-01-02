"""
ThinkRL PEFT Integration
========================

First-class PEFT/LoRA support for efficient fine-tuning.

Provides:
- LoRA configuration for common architectures
- Adapter injection and merging
- DeepSpeed compatibility

Author: EllanorAI
"""

from thinkrl.peft.lora import (
    LoRAConfig,
    get_lora_config_for_model,
    get_trainable_parameters,
    inject_lora,
    is_peft_available,
    merge_lora_weights,
    unload_lora,
)


__all__ = [
    "LoRAConfig",
    "inject_lora",
    "merge_lora_weights",
    "unload_lora",
    "get_trainable_parameters",
    "get_lora_config_for_model",
    "is_peft_available",
]
