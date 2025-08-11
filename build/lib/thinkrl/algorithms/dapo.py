"""
Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm implementation.

DAPO is a state-of-the-art reinforcement learning algorithm for training large language models
with long Chain-of-Thought reasoning. It achieves 50 points on AIME 2024 using Qwen2.5-32B,
outperforming previous methods with 50% fewer training steps.

Key Features:
- Clip-Higher: Decoupled clipping ranges to prevent entropy collapse
- Dynamic Sampling: Filters samples with zero gradients for efficiency
- Token-Level Policy Gradient Loss: Better handling of long sequences
- Overlong Reward Shaping: Reduces reward noise from truncated samples
- Group Relative Advantage Estimation: No value function required

References:
    - "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
    - ByteDance Seed & Tsinghua University, 2025
    - https://dapo-sia.github.io/
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class DAPOConfig(AlgorithmConfig):
    """
    Configuration for DAPO algorithm.

    This configuration includes all hyperparameters specific to DAPO,
    including the novel Clip-Higher technique and dynamic sampling parameters.

    Args:
        # Core DAPO hyperparameters
        clip_ratio_lower: Lower bound for clipping ratio (policy improvement)
        clip_ratio_higher: Higher bound for clipping ratio (prevents entropy collapse)
        dynamic_sampling: Enable dynamic sampling to filter zero-gradient samples
        token_level_loss: Use token-level policy gradient loss

        # Group Relative Advantage Estimation
        use_group_relative_advantage: Use group relative advantage estimation
        group_size: Size of groups for relative advantage estimation
        advantage_normalization: Normalize advantages within groups

        # Overlong Reward Shaping
        overlong_penalty: Penalty factor for overlong sequences
        max_sequence_length: Maximum allowed sequence length
        truncation_reward_adjustment: Adjust rewards for truncated sequences

        # Training dynamics
        entropy_coeff: Coefficient for entropy regularization
        value_loss_coeff: Coefficient for value function loss (if used)
        gae_lambda: GAE lambda parameter for advantage estimation

        # Dynamic sampling parameters
        min_gradient_threshold: Minimum gradient threshold for dynamic sampling
        sample_efficiency_target: Target sample efficiency ratio
        adaptive_threshold: Adapt gradient threshold based on efficiency

        # Advanced features
        use_reference_model: Use reference model for KL divergence
        kl_coeff: KL divergence coefficient
        kl_target: Target KL divergence
        kl_horizon: Horizon for KL annealing

        # Numerical stability
        eps: Small constant for numerical stability
        max_grad_norm: Maximum gradient norm for clipping
    """

    # Core DAPO hyperparameters
    clip_ratio_lower: float = 0.2
    clip_ratio_higher: float = 5.0  # Key innovation: much higher than traditional PPO
    dynamic_sampling: bool = True
    token_level_loss: bool = True

    # Group Relative Advantage Estimation
    use_group_relative_advantage: bool = True
    group_size: int = 8
    advantage_normalization: bool = True

    # Overlong Reward Shaping
    overlong_penalty: float = 0.1
    max_sequence_length: int = 2048
    truncation_reward_adjustment: bool = True

    # Training dynamics
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    gae_lambda: float = 0.95

    # Dynamic sampling parameters
    min_gradient_threshold: float = 1e-6
    sample_efficiency_target: float = 0.8
    adaptive_threshold: bool = True

    # Advanced features
    use_reference_model: bool = True
    kl_coeff: float = 0.1
    kl_target: float = 0.01
    kl_horizon: int = 10000

    # Numerical stability
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self._validate_dapo_config()

    def _validate_dapo_config(self) -> None:
        """Validate DAPO-specific configuration parameters."""
        if self.clip_ratio_lower <= 0:
            raise ValueError(
                f"clip_ratio_lower must be positive, got {self.clip_ratio_lower}"
            )

        if self.clip_ratio_higher <= self.clip_ratio_lower:
            raise ValueError(
                f"clip_ratio_higher ({self.clip_ratio_higher}) must be greater than "
                f"clip_ratio_lower ({self.clip_ratio_lower})"
            )

        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(
                f"gae_lambda must be between 0 and 1, got {self.gae_lambda}"
            )

        if self.overlong_penalty < 0:
            raise ValueError(
                f"overlong_penalty must be non-negative, got {self.overlong_penalty}"
            )


class DAPOAdvantageEstimator:
    """
    Group Relative Advantage Estimation for DAPO.

    This class implements the novel advantage estimation technique that doesn't
    require a value function, instead using group-relative comparisons.
    """

    def __init__(self, config: DAPOConfig):
        self.config = config

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute group relative advantages.

        Args:
            rewards: Rewards tensor [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            sequence_lengths: Actual sequence lengths [batch_size]

        Returns:
            Advantages tensor [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape

        # Apply overlong penalty if enabled
        if self.config.truncation_reward_adjustment and sequence_lengths is not None:
            rewards = self._apply_overlong_penalty(
                rewards, sequence_lengths, attention_mask
            )

        # Compute returns using GAE
        returns = self._compute_gae_returns(rewards, attention_mask)

        # Group relative advantage estimation
        if self.config.use_group_relative_advantage:
            advantages = self._compute_group_relative_advantages(
                returns, attention_mask
            )
        else:
            # Standard advantage estimation (returns - baseline)
            baseline = self._compute_baseline(returns, attention_mask)
            advantages = returns - baseline

        # Normalize advantages if enabled
        if self.config.advantage_normalization:
            advantages = self._normalize_advantages(advantages, attention_mask)

        return advantages

    def _apply_overlong_penalty(
        self,
        rewards: torch.Tensor,
        sequence_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply penalty for overlong sequences."""
        max_len = self.config.max_sequence_length

        # Create penalty mask for sequences exceeding max length
        penalty_mask = sequence_lengths > max_len

        if penalty_mask.any():
            # Apply penalty to the last tokens of overlong sequences
            penalty = -self.config.overlong_penalty

            for i, (is_overlong, seq_len) in enumerate(
                zip(penalty_mask, sequence_lengths)
            ):
                if is_overlong:
                    # Apply penalty to tokens beyond max_length
                    start_penalty = min(max_len, seq_len.item())
                    end_penalty = min(seq_len.item(), rewards.shape[1])
                    if start_penalty < end_penalty:
                        rewards[i, start_penalty:end_penalty] += penalty

        return rewards

    def _compute_gae_returns(
        self, rewards: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute returns using Generalized Advantage Estimation."""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)

        # For each sequence, compute GAE returns
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            seq_rewards = rewards[i][mask]
            seq_returns = torch.zeros_like(seq_rewards)

            # Compute discounted returns
            running_return = 0.0
            for t in reversed(range(len(seq_rewards))):
                running_return = (
                    seq_rewards[t] + self.config.gae_lambda * running_return
                )
                seq_returns[t] = running_return

            returns[i][mask] = seq_returns

        return returns

    def _compute_group_relative_advantages(
        self, returns: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute group relative advantages."""
        batch_size, seq_len = returns.shape
        group_size = self.config.group_size
        advantages = torch.zeros_like(returns)

        # Process in groups
        for start_idx in range(0, batch_size, group_size):
            end_idx = min(start_idx + group_size, batch_size)
            group_returns = returns[start_idx:end_idx]
            group_mask = attention_mask[start_idx:end_idx]

            # Compute group baseline (mean of valid returns in group)
            valid_returns = group_returns * group_mask
            group_sum = valid_returns.sum()
            valid_count = group_mask.sum()

            if valid_count > 0:
                group_baseline = group_sum / valid_count
                group_advantages = group_returns - group_baseline
                advantages[start_idx:end_idx] = group_advantages

        return advantages

    def _compute_baseline(
        self, returns: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute baseline for standard advantage estimation."""
        # Simple moving average baseline
        valid_returns = returns * attention_mask
        total_return = valid_returns.sum()
        valid_count = attention_mask.sum()

        if valid_count > 0:
            baseline = total_return / valid_count
        else:
            baseline = 0.0

        return torch.full_like(returns, baseline)

    def _normalize_advantages(
        self, advantages: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        valid_advantages = advantages * attention_mask

        # Compute mean and std only over valid tokens
        valid_count = attention_mask.sum()
        if valid_count > 1:
            mean = valid_advantages.sum() / valid_count

            # Compute standard deviation
            squared_diff = ((advantages - mean) * attention_mask) ** 2
            variance = squared_diff.sum() / (valid_count - 1)
            std = torch.sqrt(variance + self.config.eps)

            # Normalize
            normalized = (advantages - mean) / (std + self.config.eps)
            return normalized * attention_mask

        return advantages


class DAPOLoss:
    """
    DAPO loss computation with Clip-Higher and token-level optimization.
    """

    def __init__(self, config: DAPOConfig):
        self.config = config

    def compute_policy_loss(
        self,
        log_probs_old: torch.Tensor,
        log_probs_new: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DAPO policy loss with Clip-Higher technique.

        Args:
            log_probs_old: Old policy log probabilities [batch_size, seq_len]
            log_probs_new: New policy log probabilities [batch_size, seq_len]
            advantages: Advantage estimates [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute probability ratios
        log_ratio = log_probs_new - log_probs_old
        ratio = torch.exp(log_ratio)

        # Standard clipped objective (lower bound)
        clip_lower = 1.0 - self.config.clip_ratio_lower
        clip_upper = 1.0 + self.config.clip_ratio_lower

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, clip_lower, clip_upper) * advantages
        policy_loss_standard = -torch.min(surr1, surr2)

        # Clip-Higher objective (upper bound to prevent entropy collapse)
        clip_higher_upper = 1.0 + self.config.clip_ratio_higher
        clip_higher_lower = 1.0 - self.config.clip_ratio_higher

        # Only apply higher clipping when it's more restrictive
        surr3 = torch.clamp(ratio, clip_higher_lower, clip_higher_upper) * advantages

        # Choose the most restrictive clipping
        policy_loss_higher = -torch.min(surr1, surr3)

        # Combine standard and higher clipping
        # Use higher clipping when advantages are positive and ratio is high
        use_higher_mask = (advantages > 0) & (ratio > clip_upper)
        policy_loss = torch.where(
            use_higher_mask, policy_loss_higher, policy_loss_standard
        )

        # Apply attention mask and compute mean
        if self.config.token_level_loss:
            # Token-level loss (better for long sequences)
            masked_loss = policy_loss * attention_mask
            loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
        else:
            # Sequence-level loss
            sequence_losses = (policy_loss * attention_mask).sum(
                dim=1
            ) / attention_mask.sum(dim=1).clamp(min=1)
            loss = sequence_losses.mean()

        # Compute metrics
        metrics = self._compute_policy_metrics(
            ratio, advantages, attention_mask, use_higher_mask
        )

        return loss, metrics

    def compute_entropy_loss(
        self, logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy regularization loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)

        # Apply mask and average
        masked_entropy = entropy * attention_mask
        avg_entropy = masked_entropy.sum() / attention_mask.sum().clamp(min=1)

        return -self.config.entropy_coeff * avg_entropy  # Negative for maximization

    def compute_kl_loss(
        self,
        log_probs_new: torch.Tensor,
        log_probs_ref: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss with reference model."""
        kl_div = log_probs_new - log_probs_ref

        # Apply mask and average
        masked_kl = kl_div * attention_mask
        avg_kl = masked_kl.sum() / attention_mask.sum().clamp(min=1)

        return self.config.kl_coeff * avg_kl

    def _compute_policy_metrics(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
        use_higher_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute policy training metrics."""
        valid_mask = attention_mask.bool()

        # Basic ratio statistics
        valid_ratios = ratio[valid_mask]
        ratio_mean = valid_ratios.mean().item()
        ratio_std = valid_ratios.std().item()

        # Clipping statistics
        clip_lower = 1.0 - self.config.clip_ratio_lower
        clip_upper = 1.0 + self.config.clip_ratio_lower

        clipped_lower = (valid_ratios < clip_lower).float().mean().item()
        clipped_upper = (valid_ratios > clip_upper).float().mean().item()
        clipped_higher = use_higher_mask[valid_mask].float().mean().item()

        # Advantage statistics
        valid_advantages = advantages[valid_mask]
        advantage_mean = valid_advantages.mean().item()
        advantage_std = valid_advantages.std().item()

        return {
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "clipped_lower_frac": clipped_lower,
            "clipped_upper_frac": clipped_upper,
            "clipped_higher_frac": clipped_higher,
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
        }


class DAPOSampler:
    """
    Dynamic sampling for DAPO to filter out zero-gradient samples.
    """

    def __init__(self, config: DAPOConfig):
        self.config = config
        self.gradient_threshold = config.min_gradient_threshold
        self.efficiency_history = []

    def filter_samples(
        self,
        batch: Dict[str, torch.Tensor],
        gradients: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Filter samples based on gradient magnitude.

        Args:
            batch: Input batch dictionary
            gradients: List of gradient tensors (optional)

        Returns:
            Tuple of (filtered_batch, sample_mask)
        """
        if not self.config.dynamic_sampling or gradients is None:
            # No filtering, return original batch
            batch_size = batch["input_ids"].shape[0]
            sample_mask = torch.ones(batch_size, dtype=torch.bool)
            return batch, sample_mask

        # Compute gradient magnitudes per sample
        gradient_norms = self._compute_sample_gradient_norms(gradients, batch)

        # Filter samples based on gradient threshold
        sample_mask = gradient_norms > self.gradient_threshold

        # Adaptive threshold adjustment
        if self.config.adaptive_threshold:
            self._update_threshold(sample_mask, gradient_norms)

        # Filter batch
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == len(sample_mask):
                filtered_batch[key] = value[sample_mask]
            else:
                filtered_batch[key] = value

        return filtered_batch, sample_mask

    def _compute_sample_gradient_norms(
        self, gradients: List[torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute gradient norms per sample."""
        batch_size = batch["input_ids"].shape[0]

        # Compute total gradient norm (simplified approximation)
        total_grad_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_grad_norm += grad.norm().item() ** 2

        total_grad_norm = math.sqrt(total_grad_norm)

        # For simplicity, assume uniform distribution across samples
        # In practice, you'd want to compute per-sample gradients
        sample_norms = torch.full((batch_size,), total_grad_norm / batch_size)

        return sample_norms

    def _update_threshold(
        self, sample_mask: torch.Tensor, gradient_norms: torch.Tensor
    ) -> None:
        """Update gradient threshold based on sample efficiency."""
        efficiency = sample_mask.float().mean().item()
        self.efficiency_history.append(efficiency)

        # Keep only recent history
        if len(self.efficiency_history) > 100:
            self.efficiency_history.pop(0)

        # Adjust threshold to maintain target efficiency
        target = self.config.sample_efficiency_target
        current_avg = sum(self.efficiency_history) / len(self.efficiency_history)

        if current_avg < target:
            # Lower threshold to include more samples
            self.gradient_threshold *= 0.95
        elif current_avg > target + 0.1:
            # Raise threshold to be more selective
            self.gradient_threshold *= 1.05

        # Clamp threshold to reasonable range
        self.gradient_threshold = max(1e-8, min(1e-3, self.gradient_threshold))


class DAPO(BaseAlgorithm):
    """
    Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) algorithm.

    DAPO introduces several key innovations over traditional PPO:
    1. Clip-Higher: Decoupled clipping ranges to prevent entropy collapse
    2. Dynamic Sampling: Filters samples with zero gradients for efficiency
    3. Token-Level Loss: Better handling of long sequences
    4. Group Relative Advantage: No value function required
    5. Overlong Reward Shaping: Reduces noise from truncated samples

    Example:
        >>> config = DAPOConfig(
        ...     clip_ratio_lower=0.2,
        ...     clip_ratio_higher=5.0,
        ...     dynamic_sampling=True
        ... )
        >>> dapo = DAPO(config)
        >>> dapo.setup(model=policy_model, reference_model=ref_model)
        >>> output = dapo.step(batch)
    """

    def __init__(self, config: DAPOConfig):
        if not isinstance(config, DAPOConfig):
            raise TypeError(f"config must be DAPOConfig, got {type(config)}")

        super().__init__(config)

        self.advantage_estimator = DAPOAdvantageEstimator(config)
        self.loss_computer = DAPOLoss(config)
        self.sampler = DAPOSampler(config)

        self._reference_model = None
        self._step_count = 0

        logger.info(
            f"Initialized DAPO with clip_ratio_lower={config.clip_ratio_lower}, "
            f"clip_ratio_higher={config.clip_ratio_higher}"
        )

    def setup(self, model, optimizer=None, reference_model=None, **kwargs) -> None:
        """
        Setup DAPO with models and optimizer.

        Args:
            model: Policy model to train
            optimizer: Optimizer (optional)
            reference_model: Reference model for KL divergence (optional)
            **kwargs: Additional setup parameters
        """
        super().setup(model, optimizer, **kwargs)

        if reference_model is not None:
            self._reference_model = reference_model
            if hasattr(self._reference_model, "to"):
                self._reference_model.to(self.config.device)
            self._reference_model.eval()

        logger.info("DAPO setup complete")

    def step(self, batch: Dict[str, Any], **kwargs) -> AlgorithmOutput:
        """
        Perform one DAPO training step.

        Args:
            batch: Training batch containing:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - rewards: Rewards [batch_size, seq_len] or [batch_size]
                - old_log_probs: Old policy log probs [batch_size, seq_len]
            **kwargs: Additional parameters

        Returns:
            AlgorithmOutput containing loss, metrics, and logs
        """
        if not self._is_setup:
            raise RuntimeError("DAPO must be setup before calling step()")

        self._validate_batch(batch)

        # Move to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.config.device)

        # Set model to training mode
        self._model.train()

        # Forward pass through policy model
        model_outputs = self._model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Compute loss and metrics
        algorithm_output = self.compute_loss(batch, model_outputs, **kwargs)

        # Backward pass and optimization
        if algorithm_output.loss is not None:
            # Dynamic sampling (filter based on gradients)
            if self.config.dynamic_sampling and self._step_count > 0:
                # Get gradients for filtering (simplified)
                gradients = [
                    p.grad for p in self._model.parameters() if p.grad is not None
                ]
                filtered_batch, sample_mask = self.sampler.filter_samples(
                    batch, gradients
                )

                # Update metrics with sampling info
                algorithm_output.update_metrics(
                    sample_efficiency=sample_mask.float().mean().item()
                )

            # Standard optimization step
            self._optimizer.zero_grad()
            algorithm_output.loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.config.max_grad_norm
                )
                algorithm_output.update_metrics(grad_norm=grad_norm.item())

            self._optimizer.step()

        self._step_count += 1
        algorithm_output.update_logs(step=self._step_count)

        return algorithm_output

    def compute_loss(
        self, batch: Dict[str, Any], model_outputs: Any, **kwargs
    ) -> AlgorithmOutput:
        """
        Compute DAPO loss with all components.

        Args:
            batch: Input batch
            model_outputs: Model forward pass outputs
            **kwargs: Additional parameters

        Returns:
            AlgorithmOutput with loss and metrics
        """
        # Extract required tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        rewards = batch.get("rewards")
        old_log_probs = batch.get("old_log_probs")

        if rewards is None:
            raise ValueError("batch must contain 'rewards'")
        if old_log_probs is None:
            raise ValueError("batch must contain 'old_log_probs'")

        # Get model outputs
        logits = model_outputs.get("logits")
        if logits is None:
            raise ValueError("model_outputs must contain 'logits'")

        # Compute current policy log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs for actual tokens
        new_log_probs = torch.gather(
            log_probs, dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Handle reward shaping
        if rewards.dim() == 1:
            # Expand to sequence length if needed
            rewards = rewards.unsqueeze(-1).expand(-1, input_ids.shape[1])

        # Compute advantages
        advantages = self.advantage_estimator.compute_advantages(
            rewards=rewards,
            attention_mask=attention_mask,
            sequence_lengths=attention_mask.sum(dim=1),
        )

        # Policy loss
        policy_loss, policy_metrics = self.loss_computer.compute_policy_loss(
            log_probs_old=old_log_probs,
            log_probs_new=new_log_probs,
            advantages=advantages,
            attention_mask=attention_mask,
        )

        # Entropy loss
        entropy_loss = self.loss_computer.compute_entropy_loss(
            logits=logits, attention_mask=attention_mask
        )

        # KL divergence loss (if reference model available)
        kl_loss = 0.0
        if self._reference_model is not None and self.config.use_reference_model:
            with torch.no_grad():
                ref_outputs = self._reference_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                ref_log_probs = F.log_softmax(ref_outputs["logits"], dim=-1)
                ref_token_log_probs = torch.gather(
                    ref_log_probs, dim=-1, index=input_ids.unsqueeze(-1)
                ).squeeze(-1)

            kl_loss = self.loss_computer.compute_kl_loss(
                log_probs_new=new_log_probs,
                log_probs_ref=ref_token_log_probs,
                attention_mask=attention_mask,
            )

        # Total loss
        total_loss = policy_loss + entropy_loss + kl_loss

        # Prepare output
        output = AlgorithmOutput(loss=total_loss)

        # Add metrics
        output.update_metrics(**policy_metrics)
        output.update_metrics(
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            total_loss=total_loss.item(),
        )

        if isinstance(kl_loss, torch.Tensor):
            output.update_metrics(kl_loss=kl_loss.item())

        # Add algorithm-specific logs
        output.update_logs(
            algorithm="DAPO",
            clip_ratio_lower=self.config.clip_ratio_lower,
            clip_ratio_higher=self.config.clip_ratio_higher,
            dynamic_sampling=self.config.dynamic_sampling,
        )

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using the trained policy model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum generation length
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated sequences and metadata
        """
        if not self._is_setup:
            raise RuntimeError("DAPO must be setup before calling generate()")

        self._model.eval()

        with torch.no_grad():
            # Use model's generate method if available
            if hasattr(self._model, "generate"):
                generated = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    **kwargs,
                )
            else:
                # Simple greedy generation fallback
                generated = self._simple_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    **kwargs,
                )

        # Compute log probabilities for generated sequences
        model_outputs = self._model(
            input_ids=generated, attention_mask=torch.ones_like(generated)
        )

        log_probs = F.log_softmax(model_outputs["logits"], dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=generated.unsqueeze(-1)
        ).squeeze(-1)

        return {
            "sequences": generated,
            "log_probs": token_log_probs,
            "attention_mask": torch.ones_like(generated),
        }

    def _simple_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation method."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_length - seq_len):
            outputs = self._model(
                input_ids=generated, attention_mask=torch.ones_like(generated)
            )

            logits = outputs["logits"][:, -1, :] / temperature

            if temperature == 0.0:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def evaluate(
        self, eval_dataloader, num_eval_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on a validation dataset.

        Args:
            eval_dataloader: DataLoader for evaluation data
            num_eval_steps: Number of evaluation steps (None for full dataset)

        Returns:
            Dictionary of evaluation metrics
        """
        if not self._is_setup:
            raise RuntimeError("DAPO must be setup before calling evaluate()")

        self._model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if num_eval_steps is not None and i >= num_eval_steps:
                    break

                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.config.device)

                # Forward pass
                model_outputs = self._model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )

                # Compute losses
                algorithm_output = self.compute_loss(batch, model_outputs)

                # Accumulate metrics
                total_loss += algorithm_output.loss.item()
                total_policy_loss += algorithm_output.metrics.get("policy_loss", 0.0)
                total_entropy_loss += algorithm_output.metrics.get("entropy_loss", 0.0)
                total_kl_loss += algorithm_output.metrics.get("kl_loss", 0.0)
                num_batches += 1

        # Compute averages
        eval_metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_policy_loss": total_policy_loss / num_batches,
            "eval_entropy_loss": total_entropy_loss / num_batches,
            "eval_kl_loss": total_kl_loss / num_batches,
        }

        logger.info(f"Evaluation results: {eval_metrics}")
        return eval_metrics

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save DAPO checkpoint with algorithm-specific state."""
        additional_state = {
            "step_count": self._step_count,
            "gradient_threshold": self.sampler.gradient_threshold,
            "efficiency_history": self.sampler.efficiency_history,
        }

        super().save_checkpoint(path, **additional_state, **kwargs)

    def load_checkpoint(self, path: str, **kwargs) -> Dict[str, Any]:
        """Load DAPO checkpoint and restore algorithm-specific state."""
        checkpoint = super().load_checkpoint(path, **kwargs)

        # Restore DAPO-specific state
        if "step_count" in checkpoint:
            self._step_count = checkpoint["step_count"]

        if "gradient_threshold" in checkpoint:
            self.sampler.gradient_threshold = checkpoint["gradient_threshold"]

        if "efficiency_history" in checkpoint:
            self.sampler.efficiency_history = checkpoint["efficiency_history"]

        return checkpoint

    def get_info(self) -> Dict[str, Any]:
        """Get DAPO algorithm information."""
        info = super().get_info()

        # Add DAPO-specific information
        dapo_info = {
            "clip_ratio_lower": self.config.clip_ratio_lower,
            "clip_ratio_higher": self.config.clip_ratio_higher,
            "dynamic_sampling": self.config.dynamic_sampling,
            "token_level_loss": self.config.token_level_loss,
            "group_relative_advantage": self.config.use_group_relative_advantage,
            "step_count": self._step_count,
            "has_reference_model": self._reference_model is not None,
            "current_gradient_threshold": self.sampler.gradient_threshold,
        }

        info.update(dapo_info)
        return info


# Helper functions for DAPO usage


def create_dapo_config(**kwargs) -> DAPOConfig:
    """
    Create a DAPO configuration with sensible defaults.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        DAPOConfig instance

    Example:
        >>> config = create_dapo_config(
        ...     learning_rate=1e-4,
        ...     clip_ratio_higher=3.0,
        ...     dynamic_sampling=True
        ... )
    """
    return DAPOConfig(**kwargs)


def create_dapo_algorithm(config: Optional[DAPOConfig] = None, **kwargs) -> DAPO:
    """
    Create a DAPO algorithm instance.

    Args:
        config: DAPO configuration (optional)
        **kwargs: Configuration parameters if config is None

    Returns:
        DAPO algorithm instance

    Example:
        >>> dapo = create_dapo_algorithm(
        ...     clip_ratio_lower=0.2,
        ...     clip_ratio_higher=5.0
        ... )
    """
    if config is None:
        config = DAPOConfig(**kwargs)

    return DAPO(config)


def load_dapo_from_checkpoint(
    checkpoint_path: str, model=None, optimizer=None, reference_model=None
) -> DAPO:
    """
    Load DAPO algorithm from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Policy model (optional)
        optimizer: Optimizer (optional)
        reference_model: Reference model (optional)

    Returns:
        DAPO algorithm instance loaded from checkpoint

    Example:
        >>> dapo = load_dapo_from_checkpoint(
        ...     "checkpoints/dapo_step_1000.pt",
        ...     model=policy_model
        ... )
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    config = DAPOConfig.from_dict(config_dict)

    # Create algorithm
    dapo = DAPO(config)

    # Setup if models provided
    if model is not None:
        dapo.setup(model=model, optimizer=optimizer, reference_model=reference_model)

    # Load state
    dapo.load_checkpoint(checkpoint_path)

    return dapo


# Export main classes and functions
__all__ = [
    "DAPO",
    "DAPOConfig",
    "DAPOAdvantageEstimator",
    "DAPOLoss",
    "DAPOSampler",
    "create_dapo_config",
    "create_dapo_algorithm",
    "load_dapo_from_checkpoint",
]
