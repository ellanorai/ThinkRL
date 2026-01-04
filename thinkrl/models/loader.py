"""
ThinkRL Model Loader
====================

Unified model loading interface supporting multiple sources:
- Hugging Face Hub (hf:// or plain model ID)
- Local Checkpoints (file:// or path)
- Cloud Storage (s3://, gs:// - placeholder)

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import torch.nn as nn


logger = logging.getLogger(__name__)


def get_model(
    model_name_or_path: str,
    model_type: Literal["actor", "reward", "critic", "ref"] = "actor",
    **kwargs,
) -> nn.Module:
    """
    Unified entry point for loading models.

    Args:
        model_name_or_path: URI or path to model
        model_type: Type of model to load
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        Loaded PyTorch model
    """
    if model_name_or_path.startswith("s3://") or model_name_or_path.startswith("gs://"):
        return _load_from_cloud(model_name_or_path, model_type, **kwargs)
    elif os.path.isdir(model_name_or_path) or model_name_or_path.startswith("file://"):
        # Local checkpoint or path
        path = model_name_or_path.replace("file://", "")
        return _load_from_local(path, model_type, **kwargs)
    else:
        # Default to Hugging Face
        return _load_from_hf(model_name_or_path, model_type, **kwargs)


def _load_from_hf(
    model_name_or_path: str,
    model_type: str,
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
    """Load from Hugging Face Hub."""
    logger.info(f"Loading {model_type} model from HF: {model_name_or_path}")

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

    if model_type == "actor":
        from .actor import Actor

        return Actor(**common_kwargs)
    elif model_type == "reward":
        from .reward_model import RewardModel

        return RewardModel(normalize_reward=normalize_reward, **common_kwargs)
    elif model_type == "critic":
        from .critic import Critic

        return Critic(**common_kwargs)
    elif model_type == "ref":
        # Reference model is just a frozen actor
        from .actor import Actor

        model = Actor(**common_kwargs)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _load_from_local(path: str, model_type: str, **kwargs) -> nn.Module:
    """Load from local path (wrapper around HF load for now)."""
    logger.info(f"Loading {model_type} model from local path: {path}")
    if os.path.exists(os.path.join(path, "config.json")):
        # If it looks like a HF model directory, use standard loader
        return _load_from_hf(path, model_type, **kwargs)

    raise NotImplementedError("Arbitrary checkpoint loading not yet implemented. Use HF format.")


def _load_from_cloud(uri: str, model_type: str, **kwargs) -> nn.Module:
    """Placeholder for cloud loading."""
    raise NotImplementedError(f"Cloud loading from {uri} not yet supported.")


# Backward compatibility wrappers
def get_actor_model(model_name_or_path: str, **kwargs) -> nn.Module:
    return get_model(model_name_or_path, "actor", **kwargs)


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: Literal["reward", "critic"] = "reward",
    **kwargs,
) -> nn.Module:
    return get_model(model_name_or_path, model_type, **kwargs)


def create_reference_model(model_name_or_path: str, **kwargs) -> nn.Module:
    return get_model(model_name_or_path, "ref", lora_rank=0, **kwargs)
