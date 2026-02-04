"""
ThinkRL Dr.GRPO Algorithm
=========================

Dr.GRPO (Doctor GRPO) is a modified GRPO that removes local group
normalization for improved stability.

Key features:
- Removes local group normalization from GRPO
- No global advantage normalization (paper-faithful)
- More stable training dynamics
- Critic-free like GRPO

Reference:
- OpenRLHF implementation
- DAPO paper discussions

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import EntropyLoss, GRPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


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

    # Group settings
    group_size: int = 4

    # Training
    n_epochs: int = 1
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Gradient clipping
    clip_grad_norm: float = 1.0

    # Execution
    use_vllm: bool = False


class DrGRPOAlgorithm(BaseRLHFAlgorithm):
    """
    Dr.GRPO (Doctor GRPO) Algorithm.

    Modified GRPO that uses group mean centering (without variance normalization)
    as the advantage estimator, avoiding both local and global scale biases.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: DrGRPOConfig | None = None,
        **kwargs,
    ):
        config = config or DrGRPOConfig()

        # Warn if KL is requested but no reference model is provided
        if ref_model is None and config.kl_coeff > 0:
            logger.warning("Dr.GRPO initialized without ref_model but kl_coeff > 0. KL penalty will be 0.")

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.kl_coeff,
            clip_grad_norm=config.clip_grad_norm,
            use_vllm=config.use_vllm,
            **kwargs,
        )

        self.config: DrGRPOConfig = config

        # Initialize Loss Functions
        self.loss_fn = GRPOLoss(clip_eps=config.epsilon, beta=config.kl_coeff)
        self.entropy_loss_fn = EntropyLoss(coef=config.entropy_coeff)

    def compute_advantages(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Compute Dr.GRPO advantages.

        Strategy:
        A_i = R_i - mean(R_group)

        No standard deviation normalization is applied (neither group-local nor global).
        This preserves the unbiased nature of the policy gradient estimator.
        """
        rewards = batch["rewards"]
        cfg = self.config
        batch_size = rewards.size(0)

        if batch_size % cfg.group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} is not divisible by group_size {cfg.group_size}. "
                "Ensure data is sampled and batched in complete groups."
            )

        # 1. Group-relative baseline (Centering)
        # Reshape to [Num_Groups, Group_Size]
        grouped_rewards = rewards.view(-1, cfg.group_size)

        # Mean per group (the baseline)
        group_means = grouped_rewards.mean(dim=1, keepdim=True)

        # A_i = R_i - mean(group)
        # We do NOT divide by std (sigma), avoiding difficulty bias.
        advantages = grouped_rewards - group_means

        # Flatten back to [BatchSize]
        return advantages.view(-1)

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Compute Dr.GRPO loss.

        Objective:
        L = E[ min(ratio*A, clip(ratio)*A) - kl_coeff * KL + entropy_coeff * H ]
        """
        cfg = self.config

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        old_log_probs = batch["old_log_probs"]

        # 1. Compute Advantages
        advantages = self.compute_advantages(batch)
        # advantages is [B], need to broadcast to [B, S]
        advantages_expanded = advantages.unsqueeze(1).expand_as(old_log_probs)

        # 2. Forward pass policy model
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        # 3. Compute reference log probs
        if self.ref_model is not None and cfg.kl_coeff > 0:
            with torch.inference_mode():
                self.ref_model.eval()
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_log_probs = self.get_log_probs(ref_outputs, labels)
        else:
            ref_log_probs = log_probs.detach()

        # 4. Token Mask (completion tokens only)
        token_mask = (labels != -100).float()

        # 5. Compute KL Divergence (Eq 4) needed for GRPOLoss
        log_ratio_ref = ref_log_probs - log_probs
        ratio_ref = torch.exp(log_ratio_ref)
        kl_div = ratio_ref - log_ratio_ref - 1.0

        # 6-8. Compute GRPO Loss using GRPOLoss
        total_loss, metrics_loss = self.loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages_expanded,
            kl_div=kl_div,
            action_mask=token_mask,
        )

        # 7. Entropy (Optional)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        entropy_loss = self.entropy_loss_fn(logits, action_mask=token_mask)

        # Add entropy loss to total (EntropyLoss returns -coeff * entropy)
        total_loss = total_loss + entropy_loss

        # Metrics
        with torch.no_grad():
            num_tokens = token_mask.sum().clamp(min=1.0)

            metrics = {
                "loss": total_loss.item(),
                "kl_mean": (kl_div * token_mask).sum().item() / num_tokens.item(),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item(),
                "clip_fraction": metrics_loss["clip_frac"].item(),
                "entropy_loss": entropy_loss.item(),
            }

        return {
            "loss": total_loss,
            **metrics,
        }

    def training_step(
        self,
        batch: dict[str, Any],
        old_log_probs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Perform a single training step."""
        if old_log_probs is None:
            old_log_probs = self.compute_rollout_log_probs(batch)

        batch["old_log_probs"] = old_log_probs

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

        if self.use_vllm and self.vllm_client:
            self.sync_vllm_weights()

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items() if k != "loss"}
        metrics["loss"] = loss.item()
        metrics["grad_norm"] = grad_norm.item()

        return metrics

    @torch.no_grad()
    def compute_rollout_log_probs(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute log probs of the batch using the current policy (for 'old' policy)."""
        self.policy_model.eval()
        outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return self.get_log_probs(outputs, batch["labels"])

    def train_on_rollout(self, rollout: dict[str, Any]) -> list[dict[str, Any]]:
        """Train on collected rollout data."""
        old_log_probs = self.compute_rollout_log_probs(rollout)

        epoch_metrics = []
        for epoch in range(self.config.n_epochs):
            metrics = self.training_step(rollout, old_log_probs=old_log_probs)
            metrics["epoch"] = epoch
            epoch_metrics.append(metrics)

            current_kl = metrics.get("kl_mean", 0.0)
            if self.config.target_kl is not None and current_kl > self.config.target_kl:
                logger.info(
                    f"Early stopping at epoch {epoch} due to high KL: {current_kl:.4f} > {self.config.target_kl}"
                )
                break

        return epoch_metrics


def create_dr_grpo(
    policy_model,
    config: DrGRPOConfig | None = None,
    **kwargs,
) -> DrGRPOAlgorithm:
    """Factory function to create Dr.GRPO algorithm."""
    return DrGRPOAlgorithm(policy_model, config, **kwargs)


__all__ = ["DrGRPOConfig", "DrGRPOAlgorithm", "create_dr_grpo"]
