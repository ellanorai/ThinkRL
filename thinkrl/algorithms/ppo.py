"""
PPO Algorithm (RLHF / Token-level)
==================================

Infrastructure-grade implementation of Proximal Policy Optimization (PPO) for LLMs.
Focuses strictly on the RLHF use-case (token-level policies).

Key features:
- Explicit separation of Policy and Value models (or unified architecture with clear contract)
- Proper GAE calculation at trajectory level
- Frozen "Old Policy" logic across PPO epochs
- DDP-safe batch processing (via BaseRLHFAlgorithm)

This implementation assumes the user provides a batch of completed rollouts
(input_ids, attention_mask, rewards) and handles the internal PPO update loop.

Author: EllanorAI
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import EntropyLoss, PolicyLoss, ValueLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    learning_rate: float = 3e-4
    value_lr: float | None = None  # Optional separate LR for critic

    # PPO Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_clip: float | None = 0.2  # Clip value function updates

    # Coefficients
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Training Loop
    n_epochs: int = 4
    batch_size: int = 64  # Mini-batch size for PPO updates

    # Stability
    clip_grad_norm: float = 1.0
    normalize_advantages: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.n_epochs >= 1, "n_epochs must be >= 1"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert 0 < self.policy_clip < 1.0, "policy_clip should be in (0, 1)"


class PPOAlgorithm(BaseRLHFAlgorithm):
    """
    PPO Algorithm specialized for Language Models (RLHF).

    This class manages the PPO update logic given a buffer of rollouts.
    It supports both unified models (Policy + Value head) and separate models.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        value_optimizer: Optimizer | None = None,
        config: PPOConfig | None = None,
        **kwargs,
    ):
        """
        Initialize PPO Algorithm.

        Args:
            policy_model: Language model (Actor). If value_model is None, this
                         must also output 'values' in its forward pass.
            value_model: Optional separate Critic model.
            ref_model: Reference model for KL penalty.
            optimizer: Optimizer for policy (and value if unified).
            value_optimizer: Optional separate optimizer for value_model.
            config: PPO configuration.
        """
        config = config or PPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=0.0,  # PPO often handles KL as a reward penalty, handled in data prep
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )

        self.config = config
        self.value_model = value_model

        # Handle Optimizers
        if self.value_model is not None:
            if value_optimizer is None:
                lr = config.value_lr if config.value_lr is not None else config.learning_rate
                self.value_optimizer = torch.optim.AdamW(self.value_model.parameters(), lr=lr)
            else:
                self.value_optimizer = value_optimizer
        else:
            # Unified model case: One optimizer handled by BaseRLHFAlgorithm
            self.value_optimizer = None

        # Initialize Loss Functions
        self.policy_loss_fn = PolicyLoss(clip_eps=config.policy_clip)
        self.value_loss_fn = ValueLoss(clip_eps=config.value_clip)
        self.entropy_loss_fn = EntropyLoss(coef=config.entropy_coeff)

        logger.info(
            f"Initialized PPO (epochs={config.n_epochs}, clip={config.policy_clip}, "
            f"separate_critic={self.value_model is not None})"
        )

    def forward_value(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, outputs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        Get value estimates.

        If separate value model exists, use it.
        Otherwise, expect 'values' in policy_model output.
        """
        if self.value_model is not None:
            # Separate Critic
            v_out = self.value_model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(v_out, dict):
                return v_out["values"] if "values" in v_out else v_out["logits"].squeeze(-1)
            return v_out.squeeze(-1)
        else:
            # Unified Actor-Critic
            if outputs is None:
                outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)

            if isinstance(outputs, dict) and "values" in outputs:
                return outputs["values"]

            raise ValueError(
                "Policy model did not return 'values' and no separate value_model was provided. "
                "Ensure your model includes a value head (e.g., AutoModelForCausalLMWithValueHead)."
            )

    def train_on_rollout(self, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        """
        Perform PPO updates on a batch of collected rollouts.

        This method:
        1. Computes 'old' log probs and values (frozen for the epoch loop).
        2. Computes GAE advantages.
        3. Runs N epochs of PPO mini-batch updates.

        Args:
            batch: Dictionary containing:
                - input_ids: [B, T]
                - attention_mask: [B, T]
                - labels: [B, T] (usually input_ids masked)
                - rewards: [B] (sequence-level) or [B, T] (token-level)

        Returns:
            List of metrics dictionaries (one per epoch).
        """
        # 1. Setup Data
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids.clone())
        # Rewards should typically be sequence level scalars or token-level map
        rewards = batch["rewards"]

        device = input_ids.device

        # 2. Compute "Old" Policy & Values (No Grad)
        # We need these to compute advantages and for the ratio denominator
        with torch.no_grad():
            if self.policy_model.training:
                self.policy_model.eval()

            outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            old_log_probs = self.get_log_probs(outputs, labels)
            old_values = self.forward_value(input_ids, attention_mask, outputs=outputs)

            # Ensure proper shapes for GAE
            # Values: [B, T] or [B, T-1] depending on implementation.
            # We align with log_probs shape [B, T]
            if old_values.shape[-1] != old_log_probs.shape[-1]:
                # If value head returns 1 value per token, straightforward.
                # If it mimics causal LM, check alignment.
                # Here we assume [B, T] alignment.
                old_values = old_values[:, : old_log_probs.shape[-1]]

        # 3. Compute Advantages (GAE)
        # If rewards are sequence-level [B], we assume they apply to the last token or are distributed
        # Ideally, we construct a dense reward tensor [B, T] including KL penalties here.
        # For this infra implementation, we assume `rewards` is already dense or properly shaped [B, T]
        # or user wants sparse rewards.

        # Simplified assumption for infra: rewards is [B] (score) and we assign to last token
        # Or rewards is [B, T] (dense).
        dense_rewards = torch.zeros_like(old_values)
        if rewards.dim() == 1:
            # Assign sparse reward to last token of each sequence
            # (Finding last non-pad token)
            last_indices = attention_mask.sum(dim=1) - 1
            dense_rewards[torch.arange(rewards.size(0)), last_indices] = rewards
        else:
            dense_rewards = rewards

        # GAE
        advantages = self.compute_gae_advantages(dense_rewards, old_values)
        returns = advantages + old_values

        # 4. Prepare Dataset for Mini-batching
        # Flatten batches for shuffling
        # Note: We flatten [B, T] to [B] samples (sequences), not tokens,
        # because causal masking requires full context.
        dataset = TensorDataset(input_ids, attention_mask, labels, old_log_probs, old_values, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 5. Training Loop
        self.policy_model.train()
        if self.value_model:
            self.value_model.train()

        all_metrics = []

        for epoch in range(self.config.n_epochs):
            epoch_metrics = []

            for mb in dataloader:
                (mb_input_ids, mb_mask, mb_labels, mb_old_log_probs, mb_old_values, mb_advantages, mb_returns) = (
                    x.to(device) for x in mb
                )

                # Normalize advantages (Minibatch level)
                if self.config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                training_batch = {
                    "input_ids": mb_input_ids,
                    "attention_mask": mb_mask,
                    "labels": mb_labels,
                    "old_log_probs": mb_old_log_probs,
                    "old_values": mb_old_values,
                    "advantages": mb_advantages,
                    "returns": mb_returns,
                }

                metrics = self.training_step(training_batch)
                epoch_metrics.append(metrics)

            # Average metrics for the epoch
            avg_metrics = {k: sum(d[k] for d in epoch_metrics) / len(epoch_metrics) for k in epoch_metrics[0]}
            avg_metrics["epoch"] = epoch
            all_metrics.append(avg_metrics)

        return all_metrics

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute PPO Loss (Policy + Value + Entropy).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # 1. Forward Pass
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        new_log_probs = self.get_log_probs(outputs, labels)
        new_values = self.forward_value(input_ids, attention_mask, outputs)

        # 2. Policy Loss (Clipped Surrogate)
        # log_probs shape: [B, T]
        # Mask out padding/ignored tokens
        token_mask = labels != -100

        policy_loss, policy_metrics = self.policy_loss_fn(
            log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            action_mask=token_mask,
        )

        # 3. Value Loss
        value_loss = self.value_loss_fn(
            values=new_values,
            old_values=old_values,
            returns=returns,
            action_mask=token_mask,
        )

        # 4. Entropy Bonus
        # Approximate entropy from logits (categorical)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        entropy_loss = self.entropy_loss_fn(logits, action_mask=token_mask)

        # 5. Total Loss
        total_loss = policy_loss + self.config.value_coeff * value_loss + entropy_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "clip_frac": policy_metrics["clip_fraction"],
            "approx_kl": policy_metrics["approx_kl"],
            "mean_advantage": advantages.mean(),
            "mean_return": returns.mean(),
            "mean_value": new_values.mean(),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training update with handling for separate/unified optimizers."""

        # Zero Grads
        if self.optimizer:
            self.optimizer.zero_grad()
        if self.value_optimizer:
            self.value_optimizer.zero_grad()

        # Forward & Loss
        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]
        loss.backward()

        # Clipping & Stepping
        # Policy
        if self.optimizer:
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

        # Value (if separate)
        if self.value_optimizer and self.value_model:
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.clip_grad_norm)
            self.value_optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    def state_dict(self) -> dict[str, Any]:
        """
        Get PPO state dictionary, including separate value model if present.
        """
        state = super().state_dict()

        if self.value_model is not None:
            state["value_model"] = self.value_model.state_dict()

        if self.value_optimizer is not None:
            state["value_optimizer"] = self.value_optimizer.state_dict()

        return state

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        """
        Load PPO state dictionary.
        """
        super().load_state_dict(state, strict=strict)

        if self.value_model is not None and "value_model" in state:
            self.value_model.load_state_dict(state["value_model"], strict=strict)

        if self.value_optimizer is not None and "value_optimizer" in state:
            self.value_optimizer.load_state_dict(state["value_optimizer"])

        logger.info("Loaded PPO state (including value model/optimizer if present)")


def create_ppo(
    policy_model: nn.Module,
    value_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    value_optimizer: Optimizer | None = None,
    learning_rate: float = 3e-4,
    n_epochs: int = 4,
    batch_size: int = 64,
    **kwargs,
) -> PPOAlgorithm:
    """
    Factory function to create PPOAlgorithm instance.

    Args:
        policy_model: The policy model to optimize
        value_model: Optional separate value model
        optimizer: Optional optimizer for policy
        value_optimizer: Optional optimizer for value model
        learning_rate: Learning rate
        n_epochs: Number of PPO epochs per rollout
        batch_size: Mini-batch size
        **kwargs: Additional args for PPOConfig

    Returns:
        Configured PPOAlgorithm
    """
    config = PPOConfig(learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size, **kwargs)

    # If optimizer is None, BaseRLHFAlgorithm handles policy optimizer creation.
    # We pass it through.

    return PPOAlgorithm(
        policy_model=policy_model,
        value_model=value_model,
        optimizer=optimizer,
        value_optimizer=value_optimizer,
        config=config,
    )


__all__ = ["PPOAlgorithm", "PPOConfig", "create_ppo"]
