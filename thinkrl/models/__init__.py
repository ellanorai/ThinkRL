"""
ThinkRL Models Module
======================

Model wrappers and utilities for RLHF training.
Aligned with OpenRLHF patterns.

Available classes:
- Actor: Policy model for generation
- Critic: Value model for advantage estimation
- RewardModel: Reward model for preference learning
- get_llm_for_sequence_regression: Factory for reward/critic models

Author: Archit Sood @ EllanorAI
"""

from .actor import Actor
from .critic import Critic
from .loader import get_actor_model, get_model
from .loss import (
    DPOLoss,
    GPTLMLoss,
    KTOLoss,
    LogExpLoss,
    PairWiseLoss,
    PolicyLoss,
    SFTLoss,
    ValueLoss,
)
from .model import get_llm_for_sequence_regression
from .prm import (
    PRMConfig,
    PRMTrainer,
    ProcessRewardModel,
    create_prm,
)
from .reward_model import RewardModel
from .utils import (
    EMAModel,
    count_parameters,
    create_reference_model,
    disable_gradient_checkpointing,
    enable_gradient_checkpointing,
    freeze_layers,
    freeze_model,
    get_model_device,
    get_model_dtype,
    model_memory_footprint,
    share_model_weights,
    unfreeze_model,
    update_reference_model,
)


__all__ = [
    # Models
    "Actor",
    "Critic",
    "RewardModel",
    "get_model",
    "get_llm_for_sequence_regression",
    "get_actor_model",
    # Process Reward Model
    "PRMConfig",
    "ProcessRewardModel",
    "PRMTrainer",
    "create_prm",
    # Losses
    "GPTLMLoss",
    "PolicyLoss",
    "ValueLoss",
    "PairWiseLoss",
    "DPOLoss",
    "KTOLoss",
    "SFTLoss",
    "LogExpLoss",
    # Utils
    "create_reference_model",
    "update_reference_model",
    "share_model_weights",
    "freeze_model",
    "unfreeze_model",
    "freeze_layers",
    "EMAModel",
    "count_parameters",
    "get_model_device",
    "get_model_dtype",
    "model_memory_footprint",
    "enable_gradient_checkpointing",
    "disable_gradient_checkpointing",
]
