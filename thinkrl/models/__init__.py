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
from .reward_model import RewardModel
from .model import get_llm_for_sequence_regression
from .prm import (
    PRMConfig,
    ProcessRewardModel,
    PRMTrainer,
    create_prm,
)
from .loss import (
    GPTLMLoss,
    PolicyLoss,
    ValueLoss,
    PairWiseLoss,
    DPOLoss,
    KTOLoss,
    SFTLoss,
    LogExpLoss,
)

from .utils import (
    create_reference_model,
    update_reference_model,
    share_model_weights,
    freeze_model,
    unfreeze_model,
    freeze_layers,
    EMAModel,
    count_parameters,
    get_model_device,
    get_model_dtype,
    model_memory_footprint,
    enable_gradient_checkpointing,
    disable_gradient_checkpointing,
)


__all__ = [
    # Models
    "Actor",
    "Critic",
    "RewardModel",
    "get_llm_for_sequence_regression",
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
