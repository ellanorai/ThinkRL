"""
REINFORCE++ Algorithm Implementation
====================================

REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization.

Key Features:
1. Critic-Free: Removes the value function (critic) to save memory/compute.
2. Global Normalization: Normalizes advantages across the entire global batch
   (synced across GPUs), proving to be theoretically unbiased compared to
   GRPO's local normalization.
3. Two Variants:
   - k=1 (General): Standard PPO reward formulation (R - beta*KL) + Global Norm.
   - k>1 (Reasoning): Group Mean Baseline (R - Mean(R)) + Global Norm + Separate k2 KL Loss.

Reference:
    REINFORCE++: https://arxiv.org/abs/2501.03262
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import PolicyLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class REINFORCEPPConfig:
    """
    Configuration for REINFORCE++ algorithm.
    """

    # Optimization
    learning_rate: float = 1e-6
    gamma: float = 1.0

    # Algorithm variant
    # k=1: Use 'general' (Standard REINFORCE++)
    # k>1: Use 'baseline' (REINFORCE++ w/ Baseline for reasoning)
    mode: str = "baseline"

    # KL Penalties
    beta: float = 0.04  # Used in reward (k=1)
    kl_loss_coeff: float = 0.1  # Used as separate loss (k>1)

    # Entropy Regularization (Crucial for stability/exploration)
    entropy_coeff: float = 0.01

    # PPO Clipping (REINFORCE++ uses PPO's surrogate objective)
    clip_epsilon: float = 0.2

    # Training
    n_epochs: int = 1
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0

    # Group size for 'baseline' mode (k)
    group_size: int = 4


class REINFORCEPPAlgorithm(BaseRLHFAlgorithm):
    """
    REINFORCE++ Algorithm.

    Implements Global Advantage Normalization for stable critic-free training.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: REINFORCEPPConfig | None = None,
        **kwargs,
    ):
        config = config or REINFORCEPPConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            **kwargs,
        )
        self.config = config

        # Ensure reference model exists
        if self.ref_model is None:
            # For k=1, ref model is needed for reward calculation
            # For k>1, ref model is needed for k2 KL loss
            raise ValueError("REINFORCE++ requires a reference model.")

        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # We reuse the PPO Clipped Loss for the policy update
        self.policy_loss_fn = PolicyLoss(clip_eps=config.clip_epsilon)

    def _global_statistics(
        self, tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and std of a tensor across all distributed processes.
        This implements the "Global Advantage Normalization".

        Handles masking to correctly ignore padding/invalid tokens, ensuring
        unbiased statistics even with uneven batch sizes.
        """
        # Filter tensor by mask if provided
        if mask is not None:
            # Flatten and select only valid elements
            tensor = tensor[mask.bool()]

        # If tensor is empty (e.g. all masked), handle gracefully
        if tensor.numel() == 0:
            return torch.tensor(0.0, device=tensor.device), torch.tensor(1.0, device=tensor.device)

        # If not distributed, standard mean/std
        if not dist.is_initialized():
            return tensor.mean(), tensor.std()

        # Distributed Global Statistics
        # We compute global sum, global sq_sum, and global count
        local_sum = tensor.sum()
        local_sq_sum = (tensor**2).sum()
        local_count = torch.tensor(tensor.numel(), device=tensor.device, dtype=torch.float)

        # All-reduce to get global aggregates
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        # Compute global stats
        global_mean = local_sum / local_count
        global_var = (local_sq_sum / local_count) - (global_mean**2)
        global_std = torch.sqrt(torch.clamp(global_var, min=1e-8))

        return global_mean, global_std

    def compute_k2_kl_loss(
        self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the 'k2' estimator for KL divergence loss (Eq 8 + Appendix B.1).
        J_k2 = 0.5 * (log pi - log ref)^2

        Computed at token-level and masked.
        """
        # log_ratio = log(pi) - log(ref)
        log_ratio = log_probs - ref_log_probs
        k2_loss = 0.5 * (log_ratio**2)

        # Mask and average
        if mask.sum() > 0:
            loss = (k2_loss * mask).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=log_probs.device)

        return loss

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """
        Compute REINFORCE++ Loss.
        Returns a dictionary containing the 'loss' tensor (with grad) and 'metrics' (detached).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"]  # [B]
        old_log_probs = batch["old_log_probs"]  # [B, S]

        # Token mask (completion only)
        if "labels" in batch:
            token_mask = (labels != -100).float()
        else:
            token_mask = attention_mask.float()  # Fallback

        # --- 1. Current Policy Forward ---
        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming actor returns logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        log_probs = self.get_log_probs(logits, labels)

        # --- 2. Reference Policy Forward ---
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs
            ref_log_probs = self.get_log_probs(ref_logits, labels)

        # --- 3. Advantage Estimation & Normalization ---
        batch_size = rewards.shape[0]
        metrics = {}

        if self.config.mode == "general":
            # --- Variant 1: REINFORCE++ (k=1) ---
            # Eq 4: A = R - beta * Sum(KL)

            # Calculate sample-wise KL sum
            # Note: Paper uses old_log_probs for this reward calculation step
            kl_div_tokens = old_log_probs - ref_log_probs
            kl_sum = (kl_div_tokens * token_mask).sum(dim=1)  # [B]

            # Adjusted Reward
            advantages_raw = rewards - self.config.beta * kl_sum

            # Global Normalization (Eq 5)
            # Normalize across batch samples (mask is implicit as advantages_raw is per-sample)
            g_mean, g_std = self._global_statistics(advantages_raw)
            advantages_norm = (advantages_raw - g_mean) / (g_std + 1e-8)

            # Broadcast to token level [B, S]
            advantages_expanded = advantages_norm.unsqueeze(1).expand_as(log_probs)

            # Loss is just PPO surrogate
            ppo_loss, ppo_metrics = self.policy_loss_fn(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages_expanded,
                action_mask=token_mask,
            )

            total_loss = ppo_loss
            kl_loss = torch.tensor(0.0, device=input_ids.device)
            metrics.update(ppo_metrics)
            metrics["kl_reward_penalty"] = (self.config.beta * kl_sum).mean().detach()

        else:
            # --- Variant 2: REINFORCE++ w/ Baseline (k>1) ---
            group_size = self.config.group_size
            assert (
                batch_size % group_size == 0
            ), f"Batch size {batch_size} must be divisible by group size {group_size}"

            # Reshape to [Groups, K]
            r_grouped = rewards.view(-1, group_size)

            # Step 1: Subtract Group Mean (Local Baseline)
            # A' = R - Mean(R_group)
            group_mean = r_grouped.mean(dim=1, keepdim=True)
            advantages_centered = r_grouped - group_mean

            # Flatten back to [B]
            advantages_centered = advantages_centered.view(-1)

            # Step 2: Global Normalization (Eq 7)
            # Normalize the centered advantages across the GLOBAL batch
            g_mean, g_std = self._global_statistics(advantages_centered)
            advantages_norm = (advantages_centered - g_mean) / (g_std + 1e-8)

            # Broadcast to token level
            advantages_expanded = advantages_norm.unsqueeze(1).expand_as(log_probs)

            # Step 3: Compute Loss (Eq 8)
            # L = L_PPO - lambda * J_k2
            ppo_loss, ppo_metrics = self.policy_loss_fn(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages_expanded,
                action_mask=token_mask,
            )
            metrics.update(ppo_metrics)

            # k2 KL Estimator
            kl_val = self.compute_k2_kl_loss(log_probs, ref_log_probs, token_mask)
            kl_loss = self.config.kl_loss_coeff * kl_val

            total_loss = ppo_loss + kl_loss

            metrics["kl_k2_loss"] = kl_loss.detach()
            metrics["kl_k2_val"] = kl_val.detach()

        # --- 4. Entropy Bonus ---
        # Add entropy bonus to prevent collapse (L_total - coeff * Entropy)
        # Estimating entropy via -log_probs (valid for sampled actions)
        if self.config.entropy_coeff > 0:
            if token_mask.sum() > 0:
                entropy = -(log_probs * token_mask).sum() / token_mask.sum()
            else:
                entropy = torch.tensor(0.0, device=input_ids.device)

            # We want to Maximize entropy -> Minimize -Entropy
            entropy_loss = -self.config.entropy_coeff * entropy
            total_loss += entropy_loss
            metrics["entropy"] = entropy.detach()
            metrics["entropy_loss"] = entropy_loss.detach()

        # --- Metrics Collection ---
        with torch.no_grad():
            metrics["loss"] = total_loss.detach()
            metrics["reward_mean"] = rewards.mean().detach()
            metrics["reward_std"] = rewards.std().detach()
            metrics["advantages_mean"] = advantages_expanded[token_mask.bool()].mean().detach()

        # Return TENSOR loss for backprop, and detached metrics for logging
        return {"loss": total_loss, "metrics": metrics}

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute single training step.
        """
        # Compute old log probs if not present (though usually done in rollout)
        if "old_log_probs" not in batch:
            batch["old_log_probs"] = self.compute_rollout_log_probs(batch)

        self.policy_model.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss_output = self.compute_loss(batch)
        loss = loss_output["loss"]  # Raw tensor with grad
        metrics = loss_output["metrics"]  # Detached dict

        # Backprop
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.clip_grad_norm,
        )

        self.optimizer.step()

        # Add grad norm to metrics
        metrics["grad_norm"] = grad_norm.item()

        # Ensure all metrics are floats (for logging)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    def train_on_rollout(self, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        """
        Train on collected rollout data.
        """
        # Ensure we have old_log_probs for PPO constraint
        with torch.no_grad():
            if "old_log_probs" not in batch:
                outputs = self.policy_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                # Assuming actor returns logits
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                batch["old_log_probs"] = self.get_log_probs(logits, batch["labels"])

        epoch_metrics = []
        for _ in range(self.config.n_epochs):
            metrics = self.training_step(batch)
            epoch_metrics.append(metrics)

        return epoch_metrics


def create_reinforce_pp(
    policy_model: nn.Module,
    ref_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-6,
    batch_size: int = 64,
    mode: str = "baseline",
    group_size: int = 4,
    **kwargs,
) -> REINFORCEPPAlgorithm:
    """
    Factory function to create REINFORCE++ algorithm.
    """
    # Extract config args
    config_args = {k: v for k, v in kwargs.items() if hasattr(REINFORCEPPConfig, k)}

    config = REINFORCEPPConfig(
        learning_rate=learning_rate, batch_size=batch_size, mode=mode, group_size=group_size, **config_args
    )

    # Remaining args
    algo_kwargs = {k: v for k, v in kwargs.items() if k not in config_args}

    return REINFORCEPPAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
        learning_rate=learning_rate,  # Base algo LR
        **algo_kwargs,
    )


__all__ = ["REINFORCEPPConfig", "REINFORCEPPAlgorithm", "create_reinforce_pp"]
