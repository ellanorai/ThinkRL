"""
ThinkRL Dr.GRPO Algorithm
=========================

Dr.GRPO (Doctor GRPO) is a modified GRPO that removes local group
normalization for improved stability.

Key features:
- Removes local group normalization from GRPO
- Global advantage normalization
- More stable training dynamics
- Critic-free like GRPO

Reference:
- OpenRLHF implementation
- DAPO paper discussions

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from thinkrl.algorithms.base import BaseRLHFAlgorithm


@dataclass
class DrGRPOConfig:
    """Configuration for Dr.GRPO algorithm."""

    # Learning rate
    learning_rate: float = 1e-6

    # Clipping
    epsilon: float = 0.2

    # KL penalty
    kl_coeff: float = 0.1
    target_kl: float | None = None

    # Entropy
    entropy_coeff: float = 0.01

    # Group settings (but no local normalization)
    group_size: int = 4
    use_global_normalization: bool = True  # Key difference from GRPO

    # Training
    n_epochs: int = 1
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Gradient clipping
    clip_grad_norm: float = 1.0


class DrGRPOAlgorithm(BaseRLHFAlgorithm):
    """
    Dr.GRPO (Doctor GRPO) Algorithm.

    Modified GRPO that uses global advantage normalization instead of
    per-group normalization, leading to more stable training.

    TODO: Implement algorithm
    """

    def __init__(
        self,
        policy_model,
        config: DrGRPOConfig | None = None,
        **kwargs,
    ):
        raise NotImplementedError("Dr.GRPO algorithm not yet implemented")

    def compute_global_advantages(self, batch: dict[str, Any]) -> Any:
        """Compute globally normalized advantages."""
        raise NotImplementedError

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute Dr.GRPO loss."""
        raise NotImplementedError

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError

    def train_on_rollout(self, rollout: dict[str, Any]) -> dict[str, Any]:
        """Train on collected rollout data."""
        raise NotImplementedError


def create_dr_grpo(
    policy_model,
    config: DrGRPOConfig | None = None,
    **kwargs,
) -> DrGRPOAlgorithm:
    """Factory function to create Dr.GRPO algorithm."""
    return DrGRPOAlgorithm(policy_model, config, **kwargs)


__all__ = ["DrGRPOConfig", "DrGRPOAlgorithm", "create_dr_grpo"]
