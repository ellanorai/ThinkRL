"""
GRPO Algorithm Implementation
=============================

Group Relative Policy Optimization (GRPO) as described in
"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models".

GRPO reinforces the policy by:
1. Sampling a group of outputs {o_1, ..., o_G} for each question q.
2. Computing group-relative advantages (normalizing rewards within the group).
3. Optimizing the policy using a PPO-like surrogate objective with a direct KL penalty term.
4. Foregoing the need for a separate value function (critic) by using the group average as the baseline.

References:
    DeepSeekMath: https://arxiv.org/abs/2402.03300
    Appendix A.1.6: Group Relative Policy Optimization

Author: Archit Sood @ EllanorAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm."""

    learning_rate: float = 1e-6
    group_size: int = 64  # G in the paper
    clip_epsilon: float = 0.2  # Epsilon for PPO clipping
    beta: float = 0.04  # Coefficient for KL penalty (beta in Eq 19)

    # Training Loop
    n_epochs: int = 1  # Number of optimization epochs per rollout batch (mu in Alg 1)

    # Stability
    clip_grad_norm: float = 1.0
    advantage_eps: float = 1e-8

    # Execution
    use_vllm: bool = False

    def __post_init__(self):
        assert self.group_size > 1, "group_size must be > 1 to compute variance"
        assert self.beta >= 0, "beta (KL coeff) must be non-negative"


class GRPOAlgorithm(BaseRLHFAlgorithm):
    """
    Group Relative Policy Optimization (GRPO).

    This algorithm removes the critic model used in PPO and estimates the baseline
    from the average reward of a group of outputs sampled from the same prompt.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: GRPOConfig | None = None,
        **kwargs,
    ):
        config = config or GRPOConfig()

        # GRPO relies heavily on the reference model for the KL term in the loss
        if ref_model is None and config.beta > 0:
            logger.warning("GRPO initialized without ref_model but beta > 0. KL penalty will be 0.")

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.beta,
            clip_grad_norm=config.clip_grad_norm,
            use_vllm=config.use_vllm,
            **kwargs,
        )

        self.config = config

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages (Outcome Supervision).

        Equation 2 / Section 4.1.2:
            A_i = (r_i - mean(r_group)) / std(r_group)

        Args:
            rewards: Tensor of shape [BatchSize] containing rewards for each sample.
                     The batch must be ordered such that every `group_size` elements
                     correspond to the same prompt.

        Returns:
            Normalized advantages of shape [BatchSize].
        """
        cfg = self.config
        batch_size = rewards.size(0)

        if batch_size % cfg.group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} is not divisible by group_size {cfg.group_size}. "
                "Ensure data is sampled and batched in complete groups."
            )

        # Reshape to [Num_Groups, Group_Size]
        # grouped_rewards: {r_1, ..., r_G} for each question q
        grouped_rewards = rewards.view(-1, cfg.group_size)

        # Compute mean and std per group
        # Note: Using unbiased=False (population std) matches standard normalization practices in DL.
        mean = grouped_rewards.mean(dim=1, keepdim=True)
        std = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)

        # Normalize
        # A_i = (r_i - mean) / std
        advantages = (grouped_rewards - mean) / (std + cfg.advantage_eps)

        # Flatten back to [BatchSize]
        return advantages.view(-1)

    def get_log_probs(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract per-token log probabilities.

        Args:
            outputs: Model outputs containing logits.
            labels: Input IDs (used as targets), with -100 for masked (prompt) tokens.

        Returns:
            Log probs of shape [B, S]. Masked positions are 0.
        """
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Logits: [B, S, V]
        # We predict the next token, so logits at position t predict labels at position t+1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Create a gather index, replacing -100 with 0 to avoid index errors
        gather_index = shift_labels.clone()
        gather_index[gather_index == -100] = 0

        # Gather log probs of the ground truth tokens
        token_log_probs = torch.gather(log_probs, -1, gather_index.unsqueeze(-1)).squeeze(-1)

        # Apply mask (zero out log probs where label was -100)
        # Using multiplication is safer for gradients than in-place assignment
        token_log_probs = token_log_probs * (shift_labels != -100).float()

        # Pad one token at the end to maintain length [B, S]
        # We use torch.cat to be explicit and match DAPO implementation style
        padding = torch.zeros(token_log_probs.size(0), 1, device=token_log_probs.device, dtype=token_log_probs.dtype)

        return torch.cat([token_log_probs, padding], dim=1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute GRPO Loss (Equation 19).

        J_GRPO = E [ 1/G Sum_i ( min(ratio * A, clip(ratio) * A) - beta * D_KL ) ]

        Args:
            batch: Dict containing:
                - input_ids: [B, S]
                - attention_mask: [B, S]
                - labels: [B, S] (with -100 for prompt)
                - rewards: [B]
                - old_log_probs: [B, S] (fixed from sampling phase)
        """
        cfg = self.config

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"]
        old_log_probs = batch["old_log_probs"]  # Log probs from pi_theta_old (fixed)

        # 1. Compute Advantages (Group Relative)
        # Reshape and normalize rewards within groups
        advantages = self.compute_advantages(rewards)
        # advantages is [B], need to broadcast to [B, S] for token-level weighting
        advantages_expanded = advantages.unsqueeze(1).expand_as(old_log_probs)

        # 2. Forward pass policy model (pi_theta)
        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        # 3. Compute reference model log probabilities if needed
        if self.ref_model is not None and cfg.beta > 0:
            # We need to compute reference log probs without breaking gradient flow
            # Use inference mode or eval to ensure no gradients for ref model
            with torch.inference_mode():
                self.ref_model.eval()
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_log_probs = self.get_log_probs(ref_outputs, labels)
        else:
            # If no ref model, use policy log probs (detached) to avoid KL penalty
            ref_log_probs = log_probs.detach()

        # 4. Token Mask (completion tokens only)
        # Note: get_log_probs returns padded [B, S], we align mask accordingly
        token_mask = (labels != -100).float()

        # 5. Compute Ratio = pi_theta / pi_old
        # log_ratio = log(pi) - log(old)
        ratio = torch.exp(log_probs - old_log_probs)

        # 6. Surrogate Objective (PPO style)
        # min( r*A, clip(r, 1-eps, 1+eps)*A )
        ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
        surr1 = ratio * advantages_expanded
        surr2 = ratio_clipped * advantages_expanded
        surrogate = torch.min(surr1, surr2)

        # 7. KL Divergence Penalty (Eq 4)
        # D_KL = pi_ref/pi - log(pi_ref/pi) - 1
        # log(pi_ref/pi) = ref_log_probs - log_probs
        # pi_ref/pi = exp(ref_log_probs - log_probs)
        log_ratio_ref = ref_log_probs - log_probs
        ratio_ref = torch.exp(log_ratio_ref)
        kl_div = ratio_ref - log_ratio_ref - 1.0

        # 8. Total Loss
        # Objective J = E [ surrogate - beta * KL ]
        # We want to maximize J, so minimize Loss = -J = -surrogate + beta * KL
        # Averaged over valid tokens

        loss_per_token = -surrogate + cfg.beta * kl_div

        # Apply mask and average
        sum_loss = (loss_per_token * token_mask).sum()
        num_tokens = token_mask.sum().clamp(min=1.0)

        total_loss = sum_loss / num_tokens

        # Metrics - detach for logging
        with torch.no_grad():
            metrics = {
                "kl_mean": (kl_div * token_mask).sum() / num_tokens,
                "advantage_mean": advantages.mean(),
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "clip_fraction": ((ratio < 1.0 - cfg.clip_epsilon) | (ratio > 1.0 + cfg.clip_epsilon)).float().mean(),
            }

        return {
            "loss": total_loss,
            **metrics,
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        old_log_probs: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Perform a single training update.

        Args:
            batch: Batch of data.
            old_log_probs: Pre-computed log probs for the batch (cached from rollout).
        """
        # GRPO requires old_log_probs for the ratio computation
        if old_log_probs is None:
            # If not provided, compute them (first epoch or single pass)
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

        # Return scalars
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["grad_norm"] = grad_norm.item()

        return metrics

    @torch.no_grad()
    def compute_rollout_log_probs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log probs of the batch using the current policy (treated as 'old')."""
        self.policy_model.eval()
        outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return self.get_log_probs(outputs, batch["labels"])

    def train_on_rollout(self, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        """
        Execute the GRPO inner loop (iterations 1..mu).

        Eq 19 optimizes pi_theta against a fixed pi_old (from which data was sampled).
        We calculate old_log_probs once and reuse them.
        """
        # Freeze old policy distribution (Algorithm 1, Line 6/7 context)
        # In this implementation, 'batch' corresponds to D_b sampled from pi_theta_old
        old_log_probs = self.compute_rollout_log_probs(batch)

        epoch_metrics = []
        for epoch in range(self.config.n_epochs):
            metrics = self.training_step(batch, old_log_probs=old_log_probs)
            metrics["epoch"] = epoch
            epoch_metrics.append(metrics)

        return epoch_metrics
