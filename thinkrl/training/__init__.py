"""
ThinkRL Training Module
========================

Training utilities including KL control, mixed precision, RL helpers,
and trainers for SFT, reward modeling, etc.

Author: EllanorAI
"""

from .kl_controller import (
    AdaptiveKLController,
    FixedKLController,
    KLController,
    KLControllerConfig,
    KLControllerType,
    create_kl_controller,
)
from .mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    PrecisionType,
    cast_model_to_dtype,
    disable_tf32,
    enable_tf32,
    get_autocast_dtype,
)
from .rl_utils import (
    RewardProcessor,
    # Reward processing
    RewardProcessorConfig,
    compute_advantages_from_returns,
    compute_clipped_surrogate_loss,
    # Entropy
    compute_entropy,
    compute_entropy_bonus,
    # Advantage estimation
    compute_gae,
    compute_group_advantages,
    # KL divergence
    compute_kl_divergence,
    compute_kl_penalty,
    compute_length_penalty,
    # Clip ratios
    compute_policy_ratio,
    compute_returns,
    # Token-level
    compute_token_advantages,
    filter_zero_variance_groups,
    # Group sampling
    sample_groups,
)
from .sft_trainer import (
    SFTConfig,
    SFTTrainer,
    create_sft_trainer,
)


__all__ = [
    # SFT Trainer
    "SFTConfig",
    "SFTTrainer",
    "create_sft_trainer",
    # KL Controller
    "KLControllerType",
    "KLControllerConfig",
    "KLController",
    "AdaptiveKLController",
    "FixedKLController",
    "create_kl_controller",
    # Mixed Precision
    "PrecisionType",
    "MixedPrecisionConfig",
    "MixedPrecisionTrainer",
    "get_autocast_dtype",
    "cast_model_to_dtype",
    "enable_tf32",
    "disable_tf32",
    # RL Utils - Advantage estimation
    "compute_gae",
    "compute_returns",
    "compute_advantages_from_returns",
    # RL Utils - Reward processing
    "RewardProcessorConfig",
    "RewardProcessor",
    # RL Utils - KL divergence
    "compute_kl_divergence",
    "compute_kl_penalty",
    # RL Utils - Clip ratios
    "compute_policy_ratio",
    "compute_clipped_surrogate_loss",
    # RL Utils - Group sampling
    "sample_groups",
    "compute_group_advantages",
    "filter_zero_variance_groups",
    # RL Utils - Token-level
    "compute_token_advantages",
    "compute_length_penalty",
    # RL Utils - Entropy
    "compute_entropy",
    "compute_entropy_bonus",
]
