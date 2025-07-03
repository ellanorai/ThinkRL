"""
Proximal Policy Optimization (PPO) algorithm implementation.

PPO is a policy gradient method that uses a clipped surrogate objective to ensure
stable training by preventing large policy updates. This implementation includes
value function learning, advantage estimation via GAE, and entropy regularization.

Key Features:
- Clipped surrogate objective for stable training
- Generalized Advantage Estimation (GAE)
- Value function learning with critic network
- Entropy regularization for exploration
- Multiple epochs of mini-batch updates

References:
    - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    - OpenAI implementation and best practices
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig(AlgorithmConfig):
    """
    Configuration for PPO algorithm.

    This config includes all hyperparameters specific to PPO,
    including clipping parameters, GAE settings, and training dynamics.

    Args:
        # Core PPO hyperparameters
        clip_ratio: Clipping parameter for the policy loss
        value_loss_coeff: Coefficient for value function loss
        entropy_coeff: Coefficient for entropy regularization

        # Advantage estimation
        gamma: Discount factor for rewards
        gae_lambda: GAE lambda parameter
        use_gae: Whether to use Generalized Advantage Estimation
        advantage_normalization: Whether to normalize advantages

        # Training dynamics
        ppo_epochs: Number of PPO epochs per update
        num_mini_batches: Number of mini-batches per epoch
        target_kl: KL divergence threshold for early stopping

        # Value function
        use_value_clipping: Whether to clip value function updates
        value_clip_range: Clipping range for value function
        value_loss_weight: Weight for combining clipped vs unclipped value loss

        # Learning rates (can be different for actor and critic)
        critic_learning_rate: Learning rate for critic (None = same as actor)

        # Numerical stability
        eps: Small constant for numerical stability
        max_grad_norm: Maximum gradient norm for clipping

        # Reproducibility
        shuffle_mini_batches: Whether to shuffle mini-batches
    """

    #Core PPO hyperparameters
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01

    #Advantage estimation
    