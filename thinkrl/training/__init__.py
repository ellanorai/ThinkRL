"""
ThinkRL Training Module
========================

Training utilities including KL control, mixed precision, and RL helpers.

Author: EllanorAI
"""

from .kl_controller import (
    KLControllerType,
    KLControllerConfig,
    KLController,
    AdaptiveKLController,
    FixedKLController,
    create_kl_controller,
)

from .mixed_precision import (
    PrecisionType,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    get_autocast_dtype,
    cast_model_to_dtype,
    enable_tf32,
    disable_tf32,
)

from .rl_utils import (
    # Advantage estimation
    compute_gae,
    compute_returns,
    compute_advantages_from_returns,
    # Reward processing
    RewardProcessorConfig,
    RewardProcessor,
    # KL divergence
    compute_kl_divergence,
    compute_kl_penalty,
    # Clip ratios
    compute_policy_ratio,
    compute_clipped_surrogate_loss,
    # Group sampling
    sample_groups,
    compute_group_advantages,
    filter_zero_variance_groups,
    # Token-level
    compute_token_advantages,
    compute_length_penalty,
    # Entropy
    compute_entropy,
    compute_entropy_bonus,
)


__all__ = [
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
