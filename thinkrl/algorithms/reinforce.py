"""
REINFORCE Algorithm Implementation
==================================

Classic REINFORCE (Monte Carlo Policy Gradient) algorithm for LLMs.

REINFORCE is the simplest policy gradient method:
- No value function (critic-free)
- Uses full episode returns as baseline
- High variance but unbiased gradients

Use cases:
- Baseline for comparing more sophisticated algorithms
- Simple reward-based fine-tuning
- When value function estimation is unreliable

Author: EllanorAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import EntropyLoss, ReinforceLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE algorithm."""

    learning_rate: float = 1e-5

    # Baseline options
    baseline_type: str = "moving_average"  # "none", "moving_average", "batch_mean"
    baseline_momentum: float = 0.99  # For moving average baseline

    # Regularization
    entropy_coeff: float = 0.01  # Entropy bonus coefficient
    kl_coeff: float = 0.0  # KL penalty coefficient (0 = disabled)

    # Training stability
    clip_grad_norm: float = 1.0
    normalize_returns: bool = True

    # Discount factor (usually 1.0 for episodic LLM tasks)
    gamma: float = 1.0

    def __post_init__(self):
        assert self.baseline_type in ["none", "moving_average", "batch_mean"]
        assert 0 <= self.baseline_momentum < 1


class REINFORCEAlgorithm(BaseRLHFAlgorithm):
    """
    REINFORCE (Monte Carlo Policy Gradient) for Language Models.

    The simplest policy gradient algorithm:
        grad J(theta) = E[grad log pi(a|s) * (R - b)]

    Where:
        - R is the total return (sum of rewards)
        - b is a baseline (reduces variance)

    This implementation supports:
        - Moving average baseline (running mean of returns)
        - Batch mean baseline (mean return in current batch)
        - No baseline (high variance, unbiased)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: REINFORCEConfig | None = None,
        **kwargs,
    ):
        config = config or REINFORCEConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.kl_coeff,
            clip_grad_norm=config.clip_grad_norm,
            gamma=config.gamma,
            **kwargs,
        )

        self.config: REINFORCEConfig = config

        # Moving average baseline state
        self._baseline_mean: float = 0.0
        self._baseline_initialized: bool = False

        # Initialize Loss Functions
        self.loss_fn = ReinforceLoss()
        self.entropy_loss_fn = EntropyLoss(coef=config.entropy_coeff)

        logger.info(
            f"Initialized REINFORCE (baseline={config.baseline_type}, " f"entropy_coeff={config.entropy_coeff})"
        )

    def compute_returns(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Monte Carlo returns (discounted cumulative rewards).

        For LLMs with sequence-level rewards, this typically means
        the reward is assigned to the final token and gamma=1.0,
        so returns = rewards for all response tokens.

        Args:
            rewards: Tensor of shape [B] (sequence-level) or [B, T] (token-level)
            attention_mask: Optional mask for valid positions

        Returns:
            Returns tensor with same shape as rewards
        """
        if rewards.dim() == 1:
            # Sequence-level rewards - return as-is
            return rewards

        # Token-level rewards - compute discounted returns
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)

        # Backward pass to compute cumulative discounted returns
        running_return = torch.zeros(batch_size, device=rewards.device)
        for t in reversed(range(seq_len)):
            running_return = rewards[:, t] + self.config.gamma * running_return
            returns[:, t] = running_return

        return returns

    def compute_baseline(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline for variance reduction.

        Args:
            returns: Returns tensor [B] or [B, T]

        Returns:
            Baseline value (scalar or broadcastable tensor)
        """
        if self.config.baseline_type == "none":
            return torch.tensor(0.0, device=returns.device)

        # Flatten returns for computing statistics
        flat_returns = returns.view(-1)

        if self.config.baseline_type == "batch_mean":
            return flat_returns.mean()

        elif self.config.baseline_type == "moving_average":
            batch_mean = flat_returns.mean().item()

            if not self._baseline_initialized:
                self._baseline_mean = batch_mean
                self._baseline_initialized = True
            else:
                # Exponential moving average
                self._baseline_mean = (
                    self.config.baseline_momentum * self._baseline_mean
                    + (1 - self.config.baseline_momentum) * batch_mean
                )

            return torch.tensor(self._baseline_mean, device=returns.device)

        raise ValueError(f"Unknown baseline type: {self.config.baseline_type}")

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute REINFORCE loss.

        Loss = -E[log pi(a|s) * (R - b)] - entropy_coeff * H[pi] + kl_coeff * KL

        Args:
            batch: Dictionary containing:
                - input_ids: [B, T]
                - attention_mask: [B, T]
                - labels: [B, T] (with -100 for prompt tokens)
                - rewards: [B] (sequence-level) or [B, T] (token-level)

        Returns:
            Dictionary with loss and metrics
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"]

        device = input_ids.device

        # Forward pass
        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        # Compute returns
        returns = self.compute_returns(rewards, attention_mask)

        # Compute baseline
        baseline = self.compute_baseline(returns)

        # Normalize returns if configured
        if self.config.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages (returns - baseline)
        advantages = returns - baseline

        # Token mask (only compute loss on response tokens)
        token_mask = labels != -100

        # Expand advantages to token level if needed
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1).expand_as(log_probs)

        # Policy gradient loss: -log(pi) * A
        policy_loss = self.loss_fn(
            log_probs=log_probs,
            advantages=advantages,
            action_mask=token_mask,
        )

        # Entropy bonus (encourages exploration)
        self.entropy_loss_fn.coef = self.config.entropy_coeff
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        entropy_loss = self.entropy_loss_fn(logits, action_mask=token_mask)
        # Note: EntropyLoss returns negative entropy (scalar loss), but we might want raw entropy for metric
        # Recalculating metric entropy or extracting it if EntropyLoss supports it
        # EntropyLoss returns -coeff * entropy.
        # entropy = -entropy_loss / coeff
        entropy = -entropy_loss / (self.config.entropy_coeff + 1e-8)

        # KL penalty (optional, for staying close to reference)
        kl_loss = torch.tensor(0.0, device=device)
        kl_div = torch.tensor(0.0, device=device)
        if self.config.kl_coeff > 0 and self.ref_model is not None:
            kl_div = self.compute_kl_penalty(batch)
            kl_loss = self.config.kl_coeff * kl_div

        total_loss = policy_loss + entropy_loss + kl_loss

        # Metrics
        with torch.no_grad():
            if rewards.dim() == 1:
                reward_mean = rewards.mean()
                reward_std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0)
            else:
                reward_mean = rewards[token_mask].mean()
                reward_std = rewards[token_mask].std() if token_mask.sum() > 1 else torch.tensor(0.0)

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "entropy": entropy.detach(),
            "entropy_loss": entropy_loss.detach(),
            "kl_div": kl_div.detach() if isinstance(kl_div, torch.Tensor) else kl_div,
            "kl_loss": kl_loss.detach(),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "baseline": baseline.detach() if isinstance(baseline, torch.Tensor) else baseline,
            "advantage_mean": advantages[token_mask].mean() if token_mask.any() else torch.tensor(0.0),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute single REINFORCE training step.

        Args:
            batch: Training batch with input_ids, attention_mask, labels, rewards

        Returns:
            Dictionary of metrics
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.clip_grad_norm,
        )

        self.optimizer.step()

        # Sync weights if using vLLM
        if self.use_vllm and self.vllm_client:
            self.sync_vllm_weights()

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        return metrics


def create_reinforce(
    policy_model: nn.Module,
    ref_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-5,
    baseline_type: str = "moving_average",
    entropy_coeff: float = 0.01,
    **kwargs,
) -> REINFORCEAlgorithm:
    """
    Factory function to create REINFORCE algorithm.

    Args:
        policy_model: The policy model to optimize
        ref_model: Optional reference model for KL penalty
        optimizer: Optional optimizer
        learning_rate: Learning rate
        baseline_type: Type of baseline ("none", "moving_average", "batch_mean")
        entropy_coeff: Entropy bonus coefficient
        **kwargs: Additional config parameters

    Returns:
        Configured REINFORCEAlgorithm
    """
    config = REINFORCEConfig(
        learning_rate=learning_rate,
        baseline_type=baseline_type,
        entropy_coeff=entropy_coeff,
        **kwargs,
    )

    if optimizer is None:
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    return REINFORCEAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
    )


__all__ = ["REINFORCEAlgorithm", "REINFORCEConfig", "create_reinforce"]
