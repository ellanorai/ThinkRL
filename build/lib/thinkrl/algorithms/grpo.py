
"""
Group Relative Policy Optimization (GRPO) algorithm implementation.

GRPO is a reinforcement learning algorithm designed for efficient training of large language models
with RLHF. It uses group-based advantage normalization to reduce variance and improve stability
during training, making it particularly effective for tasks with sparse rewards.

Key Features:
- Group-based reward normalization across responses with the same prompt
- No value function required (value-free approach)
- Token-level policy gradient optimization
- Entropy regularization for exploration
- Efficient batching strategy for memory optimization

References:
    - "Group Relative Policy Optimization" (2024)
    - Used in production systems for LLM alignment
"""

import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig(AlgorithmConfig):
    """
    Configuration for GRPO algorithm.
    
    This configuration includes all hyperparameters specific to GRPO,
    including group-based normalization and token-level optimization parameters.
    
    Args:
        # Core GRPO hyperparameters
        use_group_normalization: Enable group-based reward normalization
        group_min_size: Minimum size for a group to apply normalization
        reward_clip_range: Range for clipping normalized rewards
        
        # Token-level optimization
        token_level_loss: Use token-level policy gradient loss
        only_reward_generated: Only apply rewards to generated tokens
        
        # Entropy regularization
        entropy_coeff: Coefficient for entropy regularization
        entropy_decay: Decay factor for entropy coefficient over time
        min_entropy_coeff: Minimum entropy coefficient
        
        # Training dynamics
        advantage_normalization: Normalize advantages within batch
        gae_lambda: GAE lambda parameter (if using value function)
        
        # Optimization
        micro_batch_size: Size of micro-batches for gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
        
        # Numerical stability
        eps: Small constant for numerical stability
        reward_scale: Scale factor for rewards before normalization
    """
    
    # Core GRPO hyperparameters
    use_group_normalization: bool = True
    group_min_size: int = 2
    reward_clip_range: Tuple[float, float] = (-5.0, 5.0)
    
    # Token-level optimization
    token_level_loss: bool = True
    only_reward_generated: bool = True
    
    # Entropy regularization
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.999
    min_entropy_coeff: float = 0.001
    
    # Training dynamics
    advantage_normalization: bool = True
    gae_lambda: float = 0.95
    
    # Optimization
    micro_batch_size: int = 4
    max_grad_norm: float = 1.0
    
    # Numerical stability
    eps: float = 1e-8
    reward_scale: float = 1.0
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_grpo_config()
    
    def _validate_grpo_config(self) -> None:
        """Validate GRPO-specific configuration parameters."""
        if self.group_min_size < 1:
            raise ValueError(f"group_min_size must be at least 1, got {self.group_min_size}")
        
        if self.reward_clip_range[0] >= self.reward_clip_range[1]:
            raise ValueError(
                f"reward_clip_range must be (min, max) with min < max, "
                f"got {self.reward_clip_range}"
            )
        
        if self.entropy_coeff < 0:
            raise ValueError(f"entropy_coeff must be non-negative, got {self.entropy_coeff}")
        
        if not 0 < self.entropy_decay <= 1:
            raise ValueError(f"entropy_decay must be in (0, 1], got {self.entropy_decay}")
        
        if self.min_entropy_coeff < 0:
            raise ValueError(
                f"min_entropy_coeff must be non-negative, got {self.min_entropy_coeff}"
            )
        
        if self.micro_batch_size <= 0:
            raise ValueError(
                f"micro_batch_size must be positive, got {self.micro_batch_size}"
            )


class GRPORewardNormalizer:
    """
    Group-based reward normalization for GRPO.
    
    This class implements the core innovation of GRPO: normalizing rewards
    within groups of responses that share the same prompt, reducing variance
    while preserving relative differences.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
    
    def normalize_rewards(
        self,
        rewards: torch.Tensor,
        prompts: List[str],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normalize rewards within groups defined by prompts.
        
        Args:
            rewards: Rewards tensor [batch_size] or [batch_size, seq_len]
            prompts: List of prompt strings for grouping
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Normalized rewards tensor with same shape as input
        """
        if not self.config.use_group_normalization:
            return rewards
        
        # Handle both sequence-level and token-level rewards
        original_shape = rewards.shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        
        batch_size, seq_len = rewards.shape
        normalized_rewards = torch.zeros_like(rewards)
        
        # Group by prompts
        groups = defaultdict(list)
        for idx, prompt in enumerate(prompts):
            groups[prompt].append(idx)
        
        # Normalize within each group
        for prompt, indices in groups.items():
            if len(indices) < self.config.group_min_size:
                # Skip normalization for small groups
                for idx in indices:
                    normalized_rewards[idx] = rewards[idx]
                continue
            
            # Extract group rewards
            group_rewards = rewards[indices]  # [group_size, seq_len]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                group_mask = attention_mask[indices]
                masked_rewards = group_rewards * group_mask
                
                # Compute statistics only over valid tokens
                valid_count = group_mask.sum()
                if valid_count > 0:
                    mean = masked_rewards.sum() / valid_count
                    
                    # Compute variance
                    centered = (group_rewards - mean) * group_mask
                    variance = (centered ** 2).sum() / valid_count
                    std = torch.sqrt(variance + self.config.eps)
                else:
                    mean = 0.0
                    std = 1.0
            else:
                # No mask, use all values
                mean = group_rewards.mean()
                std = group_rewards.std() + self.config.eps
            
            # Normalize and clip
            normalized = (group_rewards - mean) / std
            normalized = torch.clamp(
                normalized,
                self.config.reward_clip_range[0],
                self.config.reward_clip_range[1]
            )
            
            # Apply reward scale
            normalized = normalized * self.config.reward_scale
            
            # Store normalized rewards
            for i, idx in enumerate(indices):
                normalized_rewards[idx] = normalized[i]
        
        # Restore original shape
        if original_shape == rewards.squeeze(-1).shape:
            normalized_rewards = normalized_rewards.squeeze(-1)
        
        return normalized_rewards


class GRPOLoss:
    """
    GRPO loss computation with token-level policy gradient and entropy regularization.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.current_entropy_coeff = config.entropy_coeff
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO policy gradient loss.
        
        Args:
            log_probs: Log probabilities of actions [batch_size, seq_len]
            advantages: Advantage estimates [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generated_mask: Mask for generated tokens [batch_size, seq_len]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Apply masks
        if self.config.only_reward_generated and generated_mask is not None:
            # Only apply loss to generated tokens
            loss_mask = attention_mask * generated_mask
        else:
            loss_mask = attention_mask
        
        # Normalize advantages if enabled
        if self.config.advantage_normalization:
            advantages = self._normalize_advantages(advantages, loss_mask)
        
        # Compute policy gradient loss
        pg_loss = -log_probs * advantages
        
        if self.config.token_level_loss:
            # Token-level averaging
            masked_loss = pg_loss * loss_mask
            loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)
        else:
            # Sequence-level averaging
            seq_losses = (pg_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
            loss = seq_losses.mean()
        
        # Compute metrics
        metrics = self._compute_policy_metrics(log_probs, advantages, loss_mask)
        
        return loss, metrics
    
    def compute_entropy_loss(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute entropy regularization loss."""
        # Compute entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Apply masks
        if self.config.only_reward_generated and generated_mask is not None:
            entropy_mask = attention_mask * generated_mask
        else:
            entropy_mask = attention_mask
        
        # Average entropy
        masked_entropy = entropy * entropy_mask
        avg_entropy = masked_entropy.sum() / entropy_mask.sum().clamp(min=1)
        
        # Negative because we want to maximize entropy
        return -self.current_entropy_coeff * avg_entropy
    
    def update_entropy_coeff(self) -> None:
        """Update entropy coefficient with decay."""
        self.current_entropy_coeff = max(
            self.current_entropy_coeff * self.config.entropy_decay,
            self.config.min_entropy_coeff
        )
    
    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        # Compute statistics only over valid positions
        valid_advantages = advantages * mask
        valid_count = mask.sum()
        
        if valid_count > 1:
            mean = valid_advantages.sum() / valid_count
            
            # Compute variance
            centered = (advantages - mean) * mask
            variance = (centered ** 2).sum() / valid_count
            std = torch.sqrt(variance + self.config.eps)
            
            # Normalize
            normalized = (advantages - mean) / (std + self.config.eps)
            return normalized * mask
        
        return advantages
    
    def _compute_policy_metrics(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute policy training metrics."""
        valid_mask = mask.bool()
        
        # Advantage statistics
        valid_advantages = advantages[valid_mask]
        advantage_mean = valid_advantages.mean().item()
        advantage_std = valid_advantages.std().item()
        
        # Log probability statistics
        valid_log_probs = log_probs[valid_mask]
        log_prob_mean = valid_log_probs.mean().item()
        
        # Entropy coefficient
        entropy_coeff = self.current_entropy_coeff
        
        return {
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "log_prob_mean": log_prob_mean,
            "entropy_coeff": entropy_coeff,
        }


class GRPOBatcher:
    """
    Efficient batching strategy for GRPO to handle variable-length sequences.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
    
    def create_micro_batches(
        self,
        episodes: List[Dict[str, Any]],
        pad_token_id: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create micro-batches from episodes for gradient accumulation.
        
        Args:
            episodes: List of episode dictionaries
            pad_token_id: Padding token ID
            
        Returns:
            List of micro-batch dictionaries
        """
        # Sort episodes by total length for efficient packing
        sorted_episodes = sorted(
            episodes,
            key=lambda x: len(x['prefix_tokens']) + len(x['generated_tokens'])
        )
        
        micro_batches = []
        
        for i in range(0, len(sorted_episodes), self.config.micro_batch_size):
            batch_episodes = sorted_episodes[i:i + self.config.micro_batch_size]
            
            # Find maximum length in this micro-batch
            max_length = max(
                len(ep['prefix_tokens']) + len(ep['generated_tokens'])
                for ep in batch_episodes
            )
            
            # Create padded tensors
            batch_size = len(batch_episodes)
            input_ids = torch.full(
                (batch_size, max_length),
                pad_token_id,
                dtype=torch.long
            )
            attention_mask = torch.zeros(
                (batch_size, max_length),
                dtype=torch.long
            )
            generated_mask = torch.zeros(
                (batch_size, max_length),
                dtype=torch.long
            )
            rewards = torch.zeros(
                (batch_size, max_length),
                dtype=torch.float32
            )
            
            prompts = []
            
            for j, episode in enumerate(batch_episodes):
                prefix_len = len(episode['prefix_tokens'])
                generated_len = len(episode['generated_tokens'])
                total_len = prefix_len + generated_len
                
                # Fill input_ids
                input_ids[j, :total_len] = torch.tensor(
                    episode['prefix_tokens'] + episode['generated_tokens']
                )
                
                # Fill attention mask
                attention_mask[j, :total_len] = 1
                
                # Fill generated mask
                generated_mask[j, prefix_len:total_len] = 1
                
                # Fill rewards (expanded to token level if needed)
                if isinstance(episode['reward'], (int, float)):
                    # Sequence-level reward
                    rewards[j, prefix_len:total_len] = episode['reward']
                else:
                    # Token-level rewards
                    rewards[j, :len(episode['reward'])] = torch.tensor(episode['reward'])
                
                prompts.append(episode['prompt'])
            
            micro_batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'generated_mask': generated_mask,
                'rewards': rewards,
                'prompts': prompts,
            }
            
            micro_batches.append(micro_batch)
        
        return micro_batches


class GRPO(BaseAlgorithm):
    """
    Group Relative Policy Optimization (GRPO) algorithm.
    
    GRPO is designed for efficient RLHF training with the following key features:
    1. Group-based reward normalization for variance reduction
    2. No value function required (simpler than PPO)
    3. Token-level policy gradient optimization
    4. Efficient batching for variable-length sequences
    5. Entropy regularization with decay
    
    Example:
        >>> config = GRPOConfig(
        ...     use_group_normalization=True,
        ...     entropy_coeff=0.01,
        ...     micro_batch_size=4
        ... )
        >>> grpo = GRPO(config)
        >>> grpo.setup(model=policy_model)
        >>> output = grpo.step(batch)
    """
    
    def __init__(self, config: GRPOConfig):
        if not isinstance(config, GRPOConfig):
            raise TypeError(f"config must be GRPOConfig, got {type(config)}")
        
        super().__init__(config)
        
        self.reward_normalizer = GRPORewardNormalizer(config)
        self.loss_computer = GRPOLoss(config)
        self.batcher = GRPOBatcher(config)
        
        self._step_count = 0
        
        logger.info(
            f"Initialized GRPO with group_normalization={config.use_group_normalization}, "
            f"entropy_coeff={config.entropy_coeff}"
        )
    
    def step(self, batch: Dict[str, Any], **kwargs) -> AlgorithmOutput:
        """
        Perform one GRPO training step.
        
        Args:
            batch: Training batch containing:
                - episodes: List of episode dictionaries with:
                    - prefix_tokens: Prompt token IDs
                    - generated_tokens: Generated token IDs
                    - reward: Reward value(s)
                    - prompt: Prompt string for grouping
                - pad_token_id: Padding token ID
            **kwargs: Additional parameters
            
        Returns:
            AlgorithmOutput containing loss, metrics, and logs
        """
        if not self._is_setup:
            raise RuntimeError("GRPO must be setup before calling step()")
        
        self._validate_batch(batch)
        
        # Extract episodes and create micro-batches
        episodes = batch.get('episodes', [])
        pad_token_id = batch.get('pad_token_id', 0)
        
        if not episodes:
            raise ValueError("batch must contain 'episodes'")
        
        # Create micro-batches for gradient accumulation
        micro_batches = self.batcher.create_micro_batches(episodes, pad_token_id)
        
        # Initialize accumulators
        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_tokens = 0
        
        # Set model to training mode
        self._model.train()
        
        # Process micro-batches
        for micro_batch in micro_batches:
            # Move to device
            for key, value in micro_batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batch[key] = value.to(self.config.device)
            
            # Forward pass
            model_outputs = self._model(
                input_ids=micro_batch['input_ids'],
                attention_mask=micro_batch['attention_mask']
            )
            
            # Compute loss for this micro-batch
            micro_output = self._compute_micro_batch_loss(
                micro_batch, model_outputs, **kwargs
            )
            
            # Accumulate loss and metrics
            if micro_output.loss is not None:
                # Scale loss by micro-batch contribution
                batch_tokens = micro_batch['generated_mask'].sum().item()
                loss_scale = batch_tokens / max(1, len(episodes))
                
                scaled_loss = micro_output.loss * loss_scale
                scaled_loss.backward()
                
                total_loss += micro_output.loss.item() * batch_tokens
                num_tokens += batch_tokens
                
                # Accumulate metrics
                for key, value in micro_output.metrics.items():
                    total_metrics[key] += value * batch_tokens
        
        # Average metrics
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
            avg_metrics = {k: v / num_tokens for k, v in total_metrics.items()}
        else:
            avg_loss = 0.0
            avg_metrics = {}
        
        # Gradient clipping and optimization
        if self.config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.config.max_grad_norm
            )
            avg_metrics['grad_norm'] = grad_norm.item()
        
        self._optimizer.step()
        self._optimizer.zero_grad()
        
        # Update entropy coefficient
        self.loss_computer.update_entropy_coeff()
        
        # Update step count
        self._step_count += 1
        
        # Create output
        output = AlgorithmOutput(
            loss=torch.tensor(avg_loss),
            metrics=avg_metrics,
            logs={
                'step': self._step_count,
                'algorithm': 'GRPO',
                'entropy_coeff': self.loss_computer.current_entropy_coeff
            }
        )
        
        return output