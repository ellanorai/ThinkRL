"""
VAPO Algorithm
==============

Value-model-based Augmented Proximal Policy Optimization (VAPO) as described in
"VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks"
(arXiv:2504.05118v3).

Key Features:
1. Length-Adaptive GAE: Adjusts lambda based on sequence length to balance bias/variance.
2. Decoupled GAE: Uses lambda=1.0 for value targets and adaptive lambda for policy advantages.
3. Clip-Higher: Asymmetric clipping (epsilon_low, epsilon_high) to encourage exploration.
4. Token-Level Loss: Aggregates loss by total valid tokens instead of sample averaging.
5. Positive Example LM Loss: Auxiliary NLL loss for correct reasoning paths.

Author: EllanorAI
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import EntropyLoss, ValueLoss, VAPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class VAPOConfig:
    """Configuration for VAPO training."""

    learning_rate: float = 1e-6
    value_lr: float | None = 2e-6  # Critic often needs higher LR in VAPO

    # VAPO Specifics
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28

    # Length-Adaptive GAE
    # lambda_policy = 1 - 1 / (alpha * length)
    adaptive_gae_alpha: float = 0.05

    # Decoupled GAE
    gamma: float = 1.0  # Usually 1.0 for reasoning tasks
    lambda_value: float = 1.0  # Critic learns from unbiased returns (Monte Carlo-ish)

    # Positive Example LM Loss
    lm_loss_coeff: float = 0.1
    positive_reward_threshold: float = 0.99  # Threshold to consider a sample "correct"

    # Training Loop
    n_epochs: int = 2
    batch_size: int = 64

    # Stability
    clip_grad_norm: float = 1.0
    value_clip: float | None = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Misc
    normalize_advantages: bool = True

    def __post_init__(self):
        if self.epsilon_low <= 0:
            raise ValueError(f"epsilon_low must be positive, got {self.epsilon_low}")
        if self.epsilon_high < self.epsilon_low:
            raise ValueError(f"epsilon_high ({self.epsilon_high}) must be >= epsilon_low ({self.epsilon_low})")
        if self.adaptive_gae_alpha <= 0:
            raise ValueError(f"adaptive_gae_alpha must be positive, got {self.adaptive_gae_alpha}")


class VAPOAlgorithm(BaseRLHFAlgorithm):
    """
    Value-model-based Augmented PPO (VAPO).

    Outperforms standard PPO and value-model-free methods (like DAPO/GRPO)
    on long-chain-of-thought reasoning tasks by addressing value bias and
    reward sparsity.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        value_optimizer: Optimizer | None = None,
        config: VAPOConfig | None = None,
        **kwargs,
    ):
        config = config or VAPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=0.0,  # VAPO typically doesn't use explicit KL penalty in reward
            clip_grad_norm=config.clip_grad_norm,
            gamma=config.gamma,
            **kwargs,
        )

        self.config = config
        self.value_model = value_model

        # Initialize Value Optimizer
        if self.value_model is not None:
            if value_optimizer is None:
                lr = config.value_lr if config.value_lr is not None else config.learning_rate
                self.value_optimizer = torch.optim.AdamW(self.value_model.parameters(), lr=lr)
            else:
                self.value_optimizer = value_optimizer
        else:
            # Unified model case
            self.value_optimizer = None

        # Initialize Loss Functions
        self.policy_loss_fn = VAPOLoss(
            epsilon_low=config.epsilon_low,
            epsilon_high=config.epsilon_high,
            lm_loss_coeff=config.lm_loss_coeff,
        )
        self.value_loss_fn = ValueLoss(clip_eps=config.value_clip)
        self.entropy_loss_fn = EntropyLoss(coef=config.entropy_coeff)

        logger.info(
            f"Initialized VAPO (eps_low={config.epsilon_low}, eps_high={config.epsilon_high}, "
            f"adaptive_alpha={config.adaptive_gae_alpha})"
        )

    def forward_value(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, outputs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Get value estimates (same logic as PPO)."""
        if self.value_model is not None:
            v_out = self.value_model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(v_out, dict):
                return v_out["values"] if "values" in v_out else v_out["logits"].squeeze(-1)
            return v_out.squeeze(-1)
        else:
            if outputs is None:
                outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict) and "values" in outputs:
                return outputs["values"]
            raise ValueError("No value head found in policy model and no separate value model provided.")

    def compute_adaptive_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        lambdas: torch.Tensor,  # [B] or scalar
    ) -> torch.Tensor:
        """
        Compute GAE with support for per-sample (adaptive) lambda.

        Args:
            rewards: [B, T]
            values: [B, T] (should match rewards length, or be T+1)
            lambdas: [B] tensor of lambda values per sample

        Returns:
            advantages: [B, T]
        """
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0.0

        # Ensure lambdas is [B, 1] for broadcasting
        if lambdas.dim() == 1:
            lambdas = lambdas.unsqueeze(1)  # [B, 1]

        # Check if values includes bootstrap value (T+1) or we need to pad
        if values.shape[1] == rewards.shape[1]:
            # Append 0 for terminal value if not provided
            next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        else:
            next_values = values[:, 1:]
            values = values[:, :-1]

        # Vectorized GAE loop
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        deltas = rewards + self.config.gamma * next_values - values

        # We iterate backwards
        # A_t = delta_t + (gamma * lambda) * A_{t+1}
        # Since lambda varies per sample, we use the tensor `lambdas` in the calculation

        seq_len = rewards.shape[1]
        for t in reversed(range(seq_len)):
            # GAE formula: A_t = delta_t + gamma * lambda * A_{t+1}
            # Note: last_gae_lam implicitly holds A_{t+1}
            last_gae_lam = deltas[:, t] + (self.config.gamma * lambdas[:, 0] * last_gae_lam)
            advantages[:, t] = last_gae_lam

        return advantages

    def train_on_rollout(self, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        """
        VAPO training loop on collected rollouts.

        Steps:
        1. Compute Old Log Probs & Values.
        2. Decoupled GAE:
           - Calculate Returns using lambda=1.0 (for Critic).
           - Calculate Advantages using Adaptive Lambda (for Policy).
        3. Multi-epoch updates.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids.clone())
        rewards = batch["rewards"]  # [B] or [B, T]

        device = input_ids.device
        batch_size = input_ids.size(0)

        # --- 1. Preparation & Old Policy ---
        with torch.no_grad():
            if self.policy_model.training:
                self.policy_model.eval()

            outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            old_log_probs = self.get_log_probs(outputs, labels)
            old_values = self.forward_value(input_ids, attention_mask, outputs=outputs)

            # Align shapes
            if old_values.shape[-1] != old_log_probs.shape[-1]:
                old_values = old_values[:, : old_log_probs.shape[-1]]

        # Construct dense rewards if sparse
        dense_rewards = torch.zeros_like(old_values)
        if rewards.dim() == 1:
            # Assign sparse reward to last token
            last_indices = attention_mask.sum(dim=1) - 1
            # Clamp to ensure indices are valid
            last_indices = last_indices.clamp(max=dense_rewards.size(1) - 1)
            dense_rewards[torch.arange(batch_size), last_indices] = rewards
        else:
            dense_rewards = rewards

        # --- 2. Decoupled GAE ---

        # A) Critic Targets (Returns): Use lambda = 1.0
        # This reduces bias for the value function (Monte Carlo-like targets)
        lambdas_critic = torch.ones(batch_size, device=device) * self.config.lambda_value
        adv_critic = self.compute_adaptive_gae(dense_rewards, old_values, lambdas_critic)
        returns = adv_critic + old_values

        # B) Policy Advantages: Use Adaptive Lambda
        # Calculate response lengths (number of generated tokens)
        # Assuming labels have -100 for prompt/pad, count valid tokens
        valid_tokens_mask = labels != -100
        seq_lengths = valid_tokens_mask.sum(dim=1).float()

        # Formula: lambda = 1 - 1 / (alpha * length)
        # Clamp length to avoid division by zero or extreme values
        seq_lengths = seq_lengths.clamp(min=1.0)
        lambdas_policy = 1.0 - (1.0 / (self.config.adaptive_gae_alpha * seq_lengths))
        lambdas_policy = lambdas_policy.clamp(min=0.0, max=1.0)

        advantages = self.compute_adaptive_gae(dense_rewards, old_values, lambdas_policy)

        # --- 3. Training Loop ---
        dataset = TensorDataset(
            input_ids,
            attention_mask,
            labels,
            old_log_probs,
            old_values,
            advantages,
            returns,
            rewards if rewards.dim() == 1 else rewards.sum(dim=1),  # Keep scalar rewards for LM loss check
        )
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.policy_model.train()
        if self.value_model:
            self.value_model.train()

        all_metrics = []

        for epoch in range(self.config.n_epochs):
            epoch_metrics = []

            for mb in dataloader:
                (
                    mb_input_ids,
                    mb_mask,
                    mb_labels,
                    mb_old_log_probs,
                    mb_old_values,
                    mb_advantages,
                    mb_returns,
                    mb_raw_rewards,
                ) = (x.to(device) for x in mb)

                if self.config.normalize_advantages:
                    # Token-level normalization often better for VAPO/Reasoning
                    mask = mb_labels != -100
                    adv_mean = (mb_advantages * mask).sum() / mask.sum()
                    adv_std = torch.sqrt(((mb_advantages - adv_mean) ** 2 * mask).sum() / mask.sum())
                    mb_advantages = (mb_advantages - adv_mean) / (adv_std + 1e-8)

                training_batch = {
                    "input_ids": mb_input_ids,
                    "attention_mask": mb_mask,
                    "labels": mb_labels,
                    "old_log_probs": mb_old_log_probs,
                    "old_values": mb_old_values,
                    "advantages": mb_advantages,
                    "returns": mb_returns,
                    "raw_rewards": mb_raw_rewards,
                }

                metrics = self.training_step(training_batch)
                epoch_metrics.append(metrics)

            avg_metrics = {k: sum(d[k] for d in epoch_metrics) / len(epoch_metrics) for k in epoch_metrics[0]}
            avg_metrics["epoch"] = epoch
            all_metrics.append(avg_metrics)

        return all_metrics

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute VAPO Loss:
        1. Token-level PPO Loss with Asymmetric Clipping.
        2. Positive Example LM Loss.
        3. Value Loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        raw_rewards = batch["raw_rewards"]  # [B]

        # 1. Forward Pass
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        new_log_probs = self.get_log_probs(outputs, labels)
        new_values = self.forward_value(input_ids, attention_mask, outputs)

        # 2. Token Masking
        token_mask = labels != -100

        # 3. LM Loss Calculation (Auxiliary)
        # Identify correct samples
        positive_mask = raw_rewards >= self.config.positive_reward_threshold
        lm_loss = torch.tensor(0.0, device=input_ids.device)

        if positive_mask.any():
            # Calculate NLL only for positive samples
            # NLL = -log_prob
            # We use new_log_probs computed above

            # Expand mask to [B, T]
            pos_token_mask = token_mask & positive_mask.unsqueeze(1)

            if pos_token_mask.any():
                nll = -new_log_probs
                lm_loss = (nll * pos_token_mask).sum() / pos_token_mask.sum().clamp(min=1.0)

        # 4. Policy Loss via VAPOLoss (includes LM loss integration)
        total_loss, metrics_policy = self.policy_loss_fn(
            log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            action_mask=token_mask,
            lm_loss=lm_loss,
        )

        # 5. Value Loss
        value_loss = self.value_loss_fn(
            values=new_values,
            old_values=old_values,
            returns=returns,
            action_mask=token_mask,
        )

        # 6. Entropy
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        entropy_loss = self.entropy_loss_fn(logits, action_mask=token_mask)

        # Total Loss aggregation
        # VAPOLoss returns (policy_loss + lm_loss_term), but we need to sum everything.
        # Wait, VAPOLoss.forward returns (total_policy_loss, metrics).
        # We need to add value_loss and entropy_loss here.
        # Let's adjust self.policy_loss_fn logic or simpler: use VAPOLoss only for policy + lm.

        final_total_loss = total_loss + self.config.value_coeff * value_loss + entropy_loss

        return {
            "loss": final_total_loss,
            "policy_loss": metrics_policy["policy_loss"],
            "value_loss": value_loss,
            "lm_loss": lm_loss,
            "entropy_loss": entropy_loss,
            "clip_frac": metrics_policy["clip_frac"],
            "approx_kl": 0.5 * ((new_log_probs - old_log_probs) ** 2).mean().detach(),  # Re-compute simpler KL
            "mean_advantage": advantages[token_mask].mean(),
            "mean_return": returns[token_mask].mean(),
            "mean_value": new_values[token_mask].mean(),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""

        if self.optimizer:
            self.optimizer.zero_grad()
        if self.value_optimizer:
            self.value_optimizer.zero_grad()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]
        loss.backward()

        if self.optimizer:
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

        if self.value_optimizer and self.value_model:
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.clip_grad_norm)
            self.value_optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
