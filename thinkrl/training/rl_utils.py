"""
Reinforcement Learning Utilities
=================================

Core utilities for RLHF algorithms including advantage estimation,
reward processing, and sampling utilities.

Used by: PPO, GRPO, DAPO, VAPO, REINFORCE

Author: EllanorAI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Advantage Estimation
# =============================================================================


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Reward tensor [batch_size, seq_len]
        values: Value estimates [batch_size, seq_len + 1]
        dones: Done flags [batch_size, seq_len] (optional)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize: Whether to normalize advantages

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, seq_len = rewards.shape

    # Handle missing dones
    if dones is None:
        dones = torch.zeros_like(rewards)

    # Ensure values has correct shape
    if values.shape[1] == seq_len:
        # Add bootstrap value of 0
        values = F.pad(values, (0, 1), value=0)

    # Compute TD residuals
    not_dones = 1.0 - dones
    deltas = rewards + gamma * values[:, 1:] * not_dones - values[:, :-1]

    # Compute GAE
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        gae = deltas[:, t] + gamma * gae_lambda * not_dones[:, t] * gae
        advantages[:, t] = gae

    # Compute returns
    returns = advantages + values[:, :-1]

    # Normalize advantages
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_returns(
    rewards: torch.Tensor,
    gamma: float = 0.99,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute discounted returns (Monte Carlo).

    Args:
        rewards: Reward tensor [batch_size, seq_len]
        gamma: Discount factor
        normalize: Whether to normalize returns

    Returns:
        Returns tensor [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_advantages_from_returns(
    returns: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute advantages as returns minus baseline values.

    Args:
        returns: Computed returns [batch_size, seq_len]
        values: Baseline values [batch_size, seq_len]
        normalize: Whether to normalize advantages

    Returns:
        Advantages tensor
    """
    advantages = returns - values

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


# =============================================================================
# Reward Processing
# =============================================================================


@dataclass
class RewardProcessorConfig:
    """Configuration for reward processing."""

    # Normalization
    normalize: bool = True
    normalize_method: str = "running"  # "running", "batch", "none"
    eps: float = 1e-8

    # Clipping
    clip_reward: bool = False
    clip_min: float = -10.0
    clip_max: float = 10.0

    # Running stats momentum
    momentum: float = 0.99


class RewardProcessor:
    """
    Process and normalize rewards for stable training.

    Supports multiple normalization strategies:
    - Running: Use running mean/std (recommended)
    - Batch: Normalize within each batch
    - None: No normalization
    """

    def __init__(
        self,
        normalize: bool = True,
        normalize_method: str = "running",
        clip_reward: bool = False,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        momentum: float = 0.99,
        eps: float = 1e-8,
    ):
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.clip_reward = clip_reward
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.momentum = momentum
        self.eps = eps

        # Running statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    @classmethod
    def from_config(cls, config: RewardProcessorConfig) -> "RewardProcessor":
        return cls(
            normalize=config.normalize,
            normalize_method=config.normalize_method,
            clip_reward=config.clip_reward,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            momentum=config.momentum,
            eps=config.eps,
        )

    def process(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Process rewards with normalization and clipping.

        Args:
            rewards: Raw reward tensor

        Returns:
            Processed rewards
        """
        # Clip first if enabled
        if self.clip_reward:
            rewards = torch.clamp(rewards, self.clip_min, self.clip_max)

        # Normalize
        if self.normalize:
            if self.normalize_method == "running":
                rewards = self._normalize_running(rewards)
            elif self.normalize_method == "batch":
                rewards = self._normalize_batch(rewards)

        return rewards

    def _normalize_running(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize using running statistics."""
        # Update running stats
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()

        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

        self.count += 1

        # Normalize
        return (rewards - self.running_mean) / (self.running_var ** 0.5 + self.eps)

    def _normalize_batch(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize within batch."""
        mean = rewards.mean()
        std = rewards.std() + self.eps
        return (rewards - mean) / std

    def reset(self) -> None:
        """Reset running statistics."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.running_mean = state.get("running_mean", 0.0)
        self.running_var = state.get("running_var", 1.0)
        self.count = state.get("count", 0)


# =============================================================================
# KL Divergence Utilities
# =============================================================================


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference policy.

    Uses the approximation: KL = E[exp(log_ratio) - 1 - log_ratio]
    which is more numerically stable than the exact KL.

    Args:
        log_probs: Log probabilities from policy [batch, seq_len]
        ref_log_probs: Log probabilities from reference [batch, seq_len]
        action_mask: Mask for valid actions [batch, seq_len]

    Returns:
        KL divergence (scalar)
    """
    log_ratio = log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)

    # Approximate KL
    approx_kl = ratio - 1 - log_ratio

    if action_mask is not None:
        approx_kl = approx_kl * action_mask
        return approx_kl.sum() / (action_mask.sum() + 1e-8)

    return approx_kl.mean()


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_coef: float = 0.1,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute per-token KL penalty for RLHF loss.

    Args:
        log_probs: Policy log probs [batch, seq_len]
        ref_log_probs: Reference log probs [batch, seq_len]
        kl_coef: KL penalty coefficient
        action_mask: Mask for valid actions

    Returns:
        KL penalty tensor [batch, seq_len]
    """
    log_ratio = log_probs - ref_log_probs
    kl_penalty = kl_coef * (torch.exp(log_ratio) - 1 - log_ratio)

    if action_mask is not None:
        kl_penalty = kl_penalty * action_mask

    return kl_penalty


# =============================================================================
# Clip Ratio Utilities (for PPO/VAPO/DAPO)
# =============================================================================


def compute_policy_ratio(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute policy ratio: pi(a|s) / pi_old(a|s).

    Args:
        log_probs: Current policy log probs
        old_log_probs: Old policy log probs

    Returns:
        Ratio tensor
    """
    log_ratio = log_probs - old_log_probs
    return torch.exp(log_ratio)


def compute_clipped_surrogate_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute PPO clipped surrogate loss.

    Args:
        log_probs: Current policy log probs
        old_log_probs: Old policy log probs
        advantages: Advantage estimates
        clip_range: Clipping range epsilon
        action_mask: Mask for valid actions

    Returns:
        Tuple of (loss, metrics dict)
    """
    # Compute ratio
    ratio = compute_policy_ratio(log_probs, old_log_probs)

    # Clipped surrogate
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages

    # PPO loss (negative because we maximize)
    loss = -torch.min(surrogate1, surrogate2)

    if action_mask is not None:
        loss = loss * action_mask
        loss = loss.sum() / (action_mask.sum() + 1e-8)
    else:
        loss = loss.mean()

    # Compute metrics
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()
        approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

    metrics = {
        "clip_fraction": clip_frac,
        "approx_kl": approx_kl,
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
    }

    return loss, metrics


# =============================================================================
# Group Sampling Utilities (for GRPO/DAPO)
# =============================================================================


def sample_groups(
    prompts: list[str],
    num_samples_per_prompt: int = 4,
) -> tuple[list[str], list[int]]:
    """
    Expand prompts for group sampling.

    Args:
        prompts: Original prompts
        num_samples_per_prompt: Number of samples per prompt

    Returns:
        Tuple of (expanded prompts, group indices)
    """
    expanded_prompts = []
    group_indices = []

    for i, prompt in enumerate(prompts):
        for _ in range(num_samples_per_prompt):
            expanded_prompts.append(prompt)
            group_indices.append(i)

    return expanded_prompts, group_indices


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    baseline_type: str = "group_mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute advantages within groups for GRPO/DAPO.

    Args:
        rewards: Reward tensor [batch_size]
        group_size: Number of samples per group
        baseline_type: "group_mean", "group_max", "none"
        eps: Epsilon for normalization

    Returns:
        Advantages tensor [batch_size]
    """
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size

    # Reshape to groups
    grouped_rewards = rewards.view(num_groups, group_size)

    if baseline_type == "group_mean":
        baseline = grouped_rewards.mean(dim=1, keepdim=True)
    elif baseline_type == "group_max":
        baseline = grouped_rewards.max(dim=1, keepdim=True).values
    elif baseline_type == "none":
        baseline = torch.zeros(num_groups, 1, device=rewards.device)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    # Compute advantages
    advantages = grouped_rewards - baseline

    # Normalize within groups
    std = advantages.std(dim=1, keepdim=True) + eps
    advantages = advantages / std

    # Flatten back
    return advantages.view(-1)


def filter_zero_variance_groups(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out groups with zero reward variance (for DAPO).

    Args:
        rewards: Reward tensor [batch_size]
        group_size: Number of samples per group
        eps: Threshold for zero variance

    Returns:
        Tuple of (valid_mask, group_variances)
    """
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size

    # Reshape to groups
    grouped_rewards = rewards.view(num_groups, group_size)

    # Compute variance per group
    group_variances = grouped_rewards.var(dim=1, unbiased=False)

    # Create valid mask (expand back to batch size)
    valid_groups = group_variances > eps
    valid_mask = valid_groups.unsqueeze(1).expand(-1, group_size).reshape(-1)

    return valid_mask.float(), group_variances


# =============================================================================
# Token-Level Utilities (for VAPO)
# =============================================================================


def compute_token_advantages(
    rewards: torch.Tensor,
    token_rewards: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute token-level advantages for VAPO.

    Args:
        rewards: Sequence-level rewards [batch_size]
        token_rewards: Optional per-token rewards [batch_size, seq_len]
        action_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Token-level advantages [batch_size, seq_len]
    """
    if token_rewards is not None:
        # Use provided token rewards
        advantages = token_rewards
    else:
        # Broadcast sequence reward to all tokens
        # Assumes last token gets full reward, others get 0
        if action_mask is None:
            raise ValueError("action_mask required when token_rewards not provided")

        batch_size = rewards.shape[0]
        seq_len = action_mask.shape[1]

        # Find last valid token in each sequence
        last_token_idx = action_mask.sum(dim=1).long() - 1

        advantages = torch.zeros_like(action_mask, dtype=rewards.dtype)
        advantages[torch.arange(batch_size), last_token_idx] = rewards

    if action_mask is not None:
        advantages = advantages * action_mask

    return advantages


def compute_length_penalty(
    sequence_lengths: torch.Tensor,
    target_length: int | None = None,
    penalty_coef: float = 0.01,
    penalty_type: str = "linear",  # "linear", "quadratic", "none"
) -> torch.Tensor:
    """
    Compute length penalty for controlling generation length.

    Args:
        sequence_lengths: Length of each sequence [batch_size]
        target_length: Target length (if None, uses mean)
        penalty_coef: Penalty coefficient
        penalty_type: Type of penalty

    Returns:
        Length penalty [batch_size]
    """
    if penalty_type == "none":
        return torch.zeros_like(sequence_lengths, dtype=torch.float)

    if target_length is None:
        target_length = sequence_lengths.float().mean().item()

    deviation = (sequence_lengths.float() - target_length).abs()

    if penalty_type == "linear":
        return -penalty_coef * deviation
    elif penalty_type == "quadratic":
        return -penalty_coef * deviation ** 2
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")


# =============================================================================
# Entropy Utilities
# =============================================================================


def compute_entropy(
    logits: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute entropy of the policy distribution.

    Args:
        logits: Policy logits [batch, seq_len, vocab_size]
        action_mask: Mask for valid actions [batch, seq_len]

    Returns:
        Mean entropy
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    if action_mask is not None:
        entropy = entropy * action_mask
        return entropy.sum() / (action_mask.sum() + 1e-8)

    return entropy.mean()


def compute_entropy_bonus(
    logits: torch.Tensor,
    entropy_coef: float = 0.01,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Compute entropy bonus for exploration.

    Args:
        logits: Policy logits
        entropy_coef: Entropy coefficient
        action_mask: Mask for valid actions

    Returns:
        Tuple of (entropy loss term, entropy value)
    """
    entropy = compute_entropy(logits, action_mask)
    entropy_loss = -entropy_coef * entropy

    return entropy_loss, entropy.item()


__all__ = [
    # Advantage estimation
    "compute_gae",
    "compute_returns",
    "compute_advantages_from_returns",
    # Reward processing
    "RewardProcessorConfig",
    "RewardProcessor",
    # KL divergence
    "compute_kl_divergence",
    "compute_kl_penalty",
    # Clip ratios
    "compute_policy_ratio",
    "compute_clipped_surrogate_loss",
    # Group sampling
    "sample_groups",
    "compute_group_advantages",
    "filter_zero_variance_groups",
    # Token-level
    "compute_token_advantages",
    "compute_length_penalty",
    # Entropy
    "compute_entropy",
    "compute_entropy_bonus",
]
