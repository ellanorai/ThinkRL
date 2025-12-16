"""
DAPO Algorithm Implementation (Fixed)
======================================

Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) as described in
"DAPO: An Open-Source LLM Reinforcement Learning System at Scale".

Key innovations over GRPO/PPO:
    1. Clip-Higher: Asymmetric clipping (ε_low, ε_high) to encourage exploration
    2. Dynamic Sampling: Filter prompts with zero reward variance (proper implementation)
    3. Token-Level Loss: Aggregate by total tokens, not samples
    4. Overlong Reward Shaping: Soft penalty for excessive generation length

Fixes from original implementation:
    1. Proper dynamic sampling with over-sampling and filtering before training
    2. Multi-epoch training support with frozen old_log_probs from rollout
    3. Corrected clip diagnostics accounting for advantage sign
    4. Fixed method signature compatibility with BaseRLHFAlgorithm

Author: Archit Sood @ EllanorAI
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DAPOConfig:
    """Configuration for DAPO algorithm."""

    learning_rate: float = 1e-6
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    group_size: int = 16
    beta: float = 0.0  # KL coefficient (0 = disabled, per DAPO paper)

    # Multi-epoch training (Algorithm 1: for iteration = 1, ..., μ)
    n_epochs: int = 1  # Number of optimization epochs per rollout

    # Dynamic sampling
    dynamic_sampling: bool = True
    min_batch_size: int = 256  # Minimum valid samples required
    max_sampling_attempts: int = 10  # Max over-sampling rounds

    # Overlong punishment
    use_overlong_punishment: bool = True
    max_len: int = 16384
    cache_len: int = 4096

    # Training stability
    clip_grad_norm: float = 1.0
    advantage_eps: float = 1e-8

    # Optional entropy bonus
    entropy_coeff: float = 0.0

    def __post_init__(self):
        assert self.epsilon_low > 0, "epsilon_low must be positive"
        assert self.epsilon_high >= self.epsilon_low, "epsilon_high >= epsilon_low"
        assert self.group_size >= 2, "group_size must be >= 2 for variance computation"
        assert self.cache_len > 0, "cache_len must be positive"
        assert self.max_len > self.cache_len, "max_len must exceed cache_len"
        assert self.n_epochs >= 1, "n_epochs must be >= 1"


class DynamicSamplingBuffer:
    """
    Buffer for dynamic sampling strategy (Section 3.2).

    Over-samples and filters prompts with accuracy = 0 or 1 (zero variance),
    keeping batch size constant with only effective gradient samples.
    """

    def __init__(self, config: DAPOConfig):
        self.config = config
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.total_sampled = 0
        self.total_filtered = 0

    def add_samples(
        self,
        samples: dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> int:
        """
        Add samples to buffer, filtering zero-variance groups.

        Args:
            samples: Batch of samples with input_ids, attention_mask, labels
            rewards: Per-sample rewards [B]

        Returns:
            Number of valid samples added
        """
        cfg = self.config
        batch_size = rewards.size(0)

        if batch_size % cfg.group_size != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by group_size {cfg.group_size}")

        num_groups = batch_size // cfg.group_size
        grouped_rewards = rewards.view(num_groups, cfg.group_size)

        # Compute variance per group
        group_std = grouped_rewards.std(dim=1, unbiased=False)
        valid_groups = group_std > cfg.advantage_eps

        self.total_sampled += num_groups
        self.total_filtered += (~valid_groups).sum().item()

        # Extract valid samples
        valid_count = 0
        for g in range(num_groups):
            if valid_groups[g]:
                start_idx = g * cfg.group_size
                end_idx = (g + 1) * cfg.group_size

                group_sample = {k: v[start_idx:end_idx].clone() for k, v in samples.items()}
                group_sample["rewards"] = rewards[start_idx:end_idx].clone()
                self.buffer.append(group_sample)
                valid_count += cfg.group_size

        return valid_count

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) * self.config.group_size >= self.config.min_batch_size

    def get_batch(self, batch_size: int) -> Optional[dict[str, torch.Tensor]]:
        """
        Extract a batch from buffer.

        Returns None if insufficient samples.
        """
        num_groups_needed = batch_size // self.config.group_size

        if len(self.buffer) < num_groups_needed:
            return None

        # Pop groups from buffer
        groups = [self.buffer.pop(0) for _ in range(num_groups_needed)]

        # Concatenate into batch
        batch = {}
        for key in groups[0].keys():
            batch[key] = torch.cat([g[key] for g in groups], dim=0)

        return batch

    def get_stats(self) -> dict[str, float]:
        """Return sampling statistics."""
        if self.total_sampled == 0:
            return {"filter_ratio": 0.0, "buffer_size": 0}

        return {
            "filter_ratio": self.total_filtered / self.total_sampled,
            "buffer_size": len(self.buffer) * self.config.group_size,
            "total_sampled": self.total_sampled,
            "total_filtered": self.total_filtered,
        }

    def clear(self):
        """Reset buffer state."""
        self.buffer.clear()
        self.total_sampled = 0
        self.total_filtered = 0


class DAPOAlgorithm(BaseRLHFAlgorithm):
    """
    Decoupled Clip and Dynamic Sampling Policy Optimization.

    Removes KL penalty, uses asymmetric clipping for exploration,
    filters zero-variance groups, and normalizes loss by total tokens.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        config: Optional[DAPOConfig] = None,
        **kwargs,
    ):
        config = config or DAPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.beta,
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )

        self.config = config
        self.sampling_buffer = DynamicSamplingBuffer(config)
        self._validate_model()

    def _validate_model(self):
        """Ensure policy model has required interface."""
        if not hasattr(self.policy_model, "forward"):
            raise ValueError("policy_model must implement forward()")

    def get_log_probs(
        self,
        outputs: Union[dict[str, torch.Tensor], torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities.

        Args:
            outputs: Model outputs (dict with 'logits' or raw logits tensor)
            labels: Target token IDs [B, S], -100 for masked positions

        Returns:
            Log probabilities [B, S] with 0.0 at masked positions
        """
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        # Shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0

        token_log_probs = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

        # Zero out masked positions
        token_log_probs[shift_labels == -100] = 0.0

        # Pad to match original sequence length
        padding = torch.zeros(token_log_probs.size(0), 1, device=token_log_probs.device, dtype=token_log_probs.dtype)
        return torch.cat([token_log_probs, padding], dim=1)

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages (Equation 9).

        Note: With proper dynamic sampling, all groups should have non-zero
        variance, so we don't need to return a validity mask here.

        Args:
            rewards: Per-sample rewards [B] (R_i in paper notation)

        Returns:
            advantages: Normalized advantages [B]
        """
        cfg = self.config
        batch_size = rewards.size(0)
        num_groups = batch_size // cfg.group_size

        # Reshape to groups: [num_groups, group_size]
        grouped = rewards.view(num_groups, cfg.group_size)

        # Group statistics (Equation 9: A_i,t = (R_i - mean) / std)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True, unbiased=False)

        # Safe normalization (should be safe if dynamic sampling worked)
        std_safe = std.clamp(min=cfg.advantage_eps)
        advantages = (grouped - mean) / std_safe

        # Flatten back to batch dimension
        return advantages.view(-1)

    def compute_overlong_penalty(self, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Soft overlong punishment (Equation 13).

        Penalty schedule:
            |y| <= L_max - L_cache      -> 0 (no penalty)
            L_max - L_cache < |y| <= L_max -> linear ramp to -1
            |y| > L_max                 -> -1 (full penalty)
        """
        cfg = self.config
        penalties = torch.zeros_like(seq_lengths, dtype=torch.float32)

        soft_start = cfg.max_len - cfg.cache_len

        # Linear penalty in buffer zone: (L_max - L_cache - |y|) / L_cache
        in_buffer = (seq_lengths > soft_start) & (seq_lengths <= cfg.max_len)
        if in_buffer.any():
            penalties[in_buffer] = (soft_start - seq_lengths[in_buffer].float()) / cfg.cache_len

        # Full penalty beyond max
        over_max = seq_lengths > cfg.max_len
        if over_max.any():
            penalties[over_max] = -1.0

        return penalties

    @torch.no_grad()
    def compute_rollout_log_probs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log probs for rollout (frozen for all epochs).

        This implements the θ_old in Algorithm 1, which stays fixed
        across all μ optimization iterations.
        """
        self.policy_model.eval()

        outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        return self.get_log_probs(outputs, batch["labels"])

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute DAPO policy gradient loss (Equation 12).

        Args:
            batch: Training batch containing input_ids, labels, rewards.
                   Must also contain 'old_log_probs' injected by training_step.
        """
        cfg = self.config
        device = batch["input_ids"].device

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"].float()

        # Retrieve old_log_probs from batch (injected by training_step)
        if "old_log_probs" not in batch:
            raise ValueError("old_log_probs not found in batch. Ensure training_step is called correctly.")
        old_log_probs = batch["old_log_probs"]

        # Forward pass with current policy
        self.policy_model.train()
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        log_probs = self.get_log_probs(outputs, labels)

        # Apply overlong punishment
        if cfg.use_overlong_punishment:
            seq_lengths = attention_mask.sum(dim=-1)
            penalties = self.compute_overlong_penalty(seq_lengths)
            rewards = rewards + penalties.to(device)

        # Compute advantages
        advantages = self.compute_advantages(rewards)

        # Token mask: only compute loss on target tokens
        token_mask = labels != -100

        # Importance sampling ratio: π_θ / π_θ_old
        ratio = torch.exp(log_probs - old_log_probs)

        # Broadcast advantages to token level [B] -> [B, S]
        adv_expanded = advantages.unsqueeze(1).expand_as(ratio)

        # Asymmetric clipping (Clip-Higher, Section 3.1)
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - cfg.epsilon_low,
            1.0 + cfg.epsilon_high,
        )

        # Surrogate objectives
        surr_unclipped = ratio * adv_expanded
        surr_clipped = ratio_clipped * adv_expanded
        surr_min = torch.min(surr_unclipped, surr_clipped)

        # Token-level aggregation (Equation 12): 1 / Σ|o_i|
        num_tokens = token_mask.sum().float().clamp(min=1.0)
        policy_loss = -surr_min[token_mask].sum() / num_tokens

        # Optional entropy bonus
        entropy_loss = torch.tensor(0.0, device=device)
        if cfg.entropy_coeff > 0:
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            entropy = self._compute_entropy(logits, token_mask)
            entropy_loss = -cfg.entropy_coeff * entropy

        total_loss = policy_loss + entropy_loss

        # Diagnostics
        with torch.no_grad():
            metrics = self._compute_diagnostics(
                ratio=ratio,
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=adv_expanded,
                token_mask=token_mask,
                rewards=rewards,
            )

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            **metrics,
        }

    def _compute_entropy(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean entropy over valid tokens."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return (entropy * mask).sum() / mask.sum().clamp(min=1.0)

    def _compute_diagnostics(
        self,
        ratio: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        token_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training diagnostics with correct clip fraction accounting.

        FIX: Clip activation depends on advantage sign:
        - When A > 0 (increase prob): high clip activates if ratio > 1 + ε_high
        - When A < 0 (decrease prob): low clip activates if ratio < 1 - ε_low

        The asymmetric clipping means we care about different bounds
        depending on the direction of the policy update.
        """
        cfg = self.config

        # Masks for advantage direction
        positive_adv = advantages > 0
        negative_adv = advantages < 0

        # Clipping conditions (accounting for advantage sign)
        # High clip: trying to increase prob (A > 0) but ratio exceeds upper bound
        clipped_high = (ratio > (1.0 + cfg.epsilon_high)) & positive_adv

        # Low clip: trying to decrease prob (A < 0) but ratio below lower bound
        clipped_low = (ratio < (1.0 - cfg.epsilon_low)) & negative_adv

        # Combined clip mask
        clipped = clipped_high | clipped_low

        # Compute fractions over valid tokens
        valid_tokens = token_mask.sum().float().clamp(min=1.0)
        clip_frac = clipped[token_mask].float().sum() / valid_tokens
        clip_frac_high = clipped_high[token_mask].float().sum() / valid_tokens
        clip_frac_low = clipped_low[token_mask].float().sum() / valid_tokens

        # Approximate KL divergence: E[r - 1 - log(r)]
        log_ratio = log_probs - old_log_probs
        approx_kl = (ratio - 1 - log_ratio)[token_mask].mean()

        # Advantage statistics (per-sample, not per-token)
        sample_advantages = advantages[:, 0]  # First token has the sample advantage

        # Ratio statistics for monitoring
        ratio_masked = ratio[token_mask]

        return {
            "clip_frac": clip_frac,
            "clip_frac_high": clip_frac_high,
            "clip_frac_low": clip_frac_low,
            "approx_kl": approx_kl,
            "advantage_mean": sample_advantages.mean(),
            "advantage_std": sample_advantages.std() if sample_advantages.numel() > 1 else torch.tensor(0.0),
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std() if rewards.numel() > 1 else torch.tensor(0.0),
            "ratio_mean": ratio_masked.mean(),
            "ratio_std": ratio_masked.std() if ratio_masked.numel() > 1 else torch.tensor(0.0),
            "ratio_max": ratio_masked.max(),
            "ratio_min": ratio_masked.min(),
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        old_log_probs: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Training batch
            old_log_probs: Pre-computed log probs from rollout (required for multi-epoch)
        """
        # Compute old_log_probs if not provided (single epoch case)
        if old_log_probs is None:
            old_log_probs = self.compute_rollout_log_probs(batch)

        # Inject old_log_probs into batch for compute_loss to find it
        # This fixes the signature mismatch with BaseRLHFAlgorithm
        batch["old_log_probs"] = old_log_probs

        self.policy_model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.clip_grad_norm,
        )
        self.optimizer.step()

        # Sync to inference engine if applicable
        if hasattr(self, "use_vllm") and self.use_vllm:
            self.sync_vllm_weights()

        # Convert to Python floats for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["grad_norm"] = grad_norm.item()

        return metrics

    def train_on_rollout(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[dict[str, float]]:
        """
        Multi-epoch training on a single rollout batch (Algorithm 1, lines 10-11).

        Computes old_log_probs once and reuses across all epochs.

        Args:
            batch: Rollout batch with input_ids, attention_mask, labels, rewards

        Returns:
            List of metrics dicts, one per epoch
        """
        # Compute old_log_probs ONCE before any updates (θ_old frozen)
        old_log_probs = self.compute_rollout_log_probs(batch)

        all_metrics = []
        for epoch in range(self.config.n_epochs):
            # Pass old_log_probs to training_step which will inject it into batch
            metrics = self.training_step(batch, old_log_probs)
            metrics["epoch"] = epoch
            all_metrics.append(metrics)

            # Optional: early stopping if KL too high
            if metrics.get("approx_kl", 0) > 0.1:
                logger.warning(f"KL divergence too high ({metrics['approx_kl']:.4f}), stopping early")
                break

        return all_metrics

    def process_rollout_with_dynamic_sampling(
        self,
        sample_fn,
        batch_size: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        """
        Implements proper dynamic sampling (Section 3.2).

        Over-samples until we have enough valid (non-zero-variance) groups,
        then returns a fixed-size batch.

        Args:
            sample_fn: Callable that returns (samples_dict, rewards_tensor)
            batch_size: Target batch size

        Returns:
            Batch with exactly batch_size samples, all with effective gradients
        """
        cfg = self.config
        self.sampling_buffer.clear()

        for attempt in range(cfg.max_sampling_attempts):
            # Sample a batch
            samples, rewards = sample_fn()

            # Add to buffer (filters zero-variance groups)
            valid_added = self.sampling_buffer.add_samples(samples, rewards)

            logger.debug(
                f"Sampling attempt {attempt + 1}: added {valid_added} valid samples, "
                f"buffer size: {self.sampling_buffer.get_stats()['buffer_size']}"
            )

            # Check if we have enough
            batch = self.sampling_buffer.get_batch(batch_size)
            if batch is not None:
                stats = self.sampling_buffer.get_stats()
                logger.info(
                    f"Dynamic sampling complete: filter_ratio={stats['filter_ratio']:.2%}, "
                    f"total_sampled={stats['total_sampled']}"
                )
                return batch

        # Failed to collect enough samples
        logger.warning(
            f"Failed to collect {batch_size} valid samples after "
            f"{cfg.max_sampling_attempts} attempts. "
            f"Buffer has {self.sampling_buffer.get_stats()['buffer_size']} samples."
        )
        return None


# Convenience function for creating configured DAPO
def create_dapo(
    policy_model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 1e-6,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
    group_size: int = 16,
    n_epochs: int = 1,
    **kwargs,
) -> DAPOAlgorithm:
    """Factory function for DAPO with common defaults."""
    config = DAPOConfig(
        learning_rate=learning_rate,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        group_size=group_size,
        n_epochs=n_epochs,
        **kwargs,
    )

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    return DAPOAlgorithm(
        policy_model=policy_model,
        optimizer=optimizer,
        config=config,
    )
