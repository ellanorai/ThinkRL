"""
ThinkRL Model Factory
======================

Factory functions for creating RLHF models.
Aligned with OpenRLHF patterns.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: Literal["reward", "critic"] = "reward",
    use_flash_attention: bool = True,
    bf16: bool = True,
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    device_map: str | dict | None = None,
    trust_remote_code: bool = False,
    normalize_reward: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function for creating reward or critic models.

    This is the main entry point for creating sequence regression models
    (reward models and critic models) from pretrained transformers.

    Args:
        model_name_or_path: Pretrained model name or path
        model_type: "reward" or "critic"
        use_flash_attention: Use Flash Attention 2
        bf16: Use bfloat16 precision
        load_in_4bit: Load in 4-bit quantization
        lora_rank: LoRA rank (0 to disable)
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        target_modules: LoRA target modules
        device_map: Device placement strategy
        trust_remote_code: Trust remote code
        normalize_reward: Normalize rewards (for reward model)
        **kwargs: Additional model arguments

    Returns:
        Reward or Critic model instance

    Example:
        ```python
        # Create reward model
        reward_model = get_llm_for_sequence_regression(
            "meta-llama/Llama-2-7b-hf",
            model_type="reward",
            lora_rank=8,
        )

        # Create critic model
        critic_model = get_llm_for_sequence_regression(
            "meta-llama/Llama-2-7b-hf",
            model_type="critic",
            lora_rank=8,
        )
        ```
    """
    common_kwargs = {
        "pretrained_model": model_name_or_path,
        "use_flash_attention": use_flash_attention,
        "bf16": bf16,
        "load_in_4bit": load_in_4bit,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }

    if model_type == "reward":
        from .reward_model import RewardModel
        return RewardModel(
            normalize_reward=normalize_reward,
            **common_kwargs,
        )
    elif model_type == "critic":
        from .critic import Critic
        return Critic(**common_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'reward' or 'critic'.")


def get_actor_model(
    model_name_or_path: str,
    use_flash_attention: bool = True,
    bf16: bool = True,
    load_in_4bit: bool = False,
    lora_rank: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    device_map: str | dict | None = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function for creating actor/policy models.

    Args:
        model_name_or_path: Pretrained model name or path
        use_flash_attention: Use Flash Attention 2
        bf16: Use bfloat16 precision
        load_in_4bit: Load in 4-bit quantization
        lora_rank: LoRA rank (0 to disable)
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        target_modules: LoRA target modules
        device_map: Device placement strategy
        trust_remote_code: Trust remote code
        **kwargs: Additional model arguments

    Returns:
        Actor model instance

    Example:
        ```python
        actor = get_actor_model(
            "meta-llama/Llama-2-7b-hf",
            lora_rank=8,
        )
        ```
    """
    from .actor import Actor
    return Actor(
        pretrained_model=model_name_or_path,
        use_flash_attention=use_flash_attention,
        bf16=bf16,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def create_reference_model(
    model_name_or_path: str,
    bf16: bool = True,
    device_map: str | dict | None = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a frozen reference model for KL computation.

    The reference model is used to compute KL divergence penalty
    during RLHF training.

    Args:
        model_name_or_path: Pretrained model name or path
        bf16: Use bfloat16 precision
        device_map: Device placement strategy
        trust_remote_code: Trust remote code
        **kwargs: Additional model arguments

    Returns:
        Frozen Actor model instance

    Example:
        ```python
        ref_model = create_reference_model("meta-llama/Llama-2-7b-hf")
        # ref_model is frozen and won't be trained
        ```
    """
    from .actor import Actor

    ref_model = Actor(
        pretrained_model=model_name_or_path,
        bf16=bf16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        lora_rank=0,  # No LoRA for reference model
        **kwargs,
    )

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()
    logger.info(f"Created frozen reference model from: {model_name_or_path}")

    return ref_model


def compute_model_size(model: nn.Module) -> dict[str, int]:
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
