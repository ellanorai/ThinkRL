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


__all__ = [
    # Models
    "Actor",
    "Critic",
    "RewardModel",
    "get_llm_for_sequence_regression",
    # Losses
    "GPTLMLoss",
    "PolicyLoss",
    "ValueLoss",
    "PairWiseLoss",
    "DPOLoss",
    "KTOLoss",
    "SFTLoss",
    "LogExpLoss",
]
