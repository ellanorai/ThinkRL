"""
Proximal Policy Optimization (PPO) algorithm implementation.

PPO is a policy gradient method that uses a clipped surrogate objective to ensure
stable training by preventing large policy updates. This implementation includes
value function learning, advantage estimation via GAE, and entropy regularization.

Key Features:
- Clipped surrogate objective for stable training
- Generalized Advantage Estimation (GAE)
- Value function learning with critic network
- Entropy regularization for exploration
- Multiple epochs of mini-batch updates

References:
    - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    - OpenAI implementation and best practices
"""

import logging
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig(AlgorithmConfig):
    """
    Configuration for PPO algorithm.

    This config includes all hyperparameters specific to PPO,
    including clipping parameters, GAE settings, and training dynamics.

    Args:
        # Core PPO hyperparameters
        clip_ratio: Clipping parameter for the policy loss
        value_loss_coeff: Coefficient for value function loss
        entropy_coeff: Coefficient for entropy regularization

        # Advantage estimation
        gamma: Discount factor for rewards
        gae_lambda: GAE lambda parameter
        use_gae: Whether to use Generalized Advantage Estimation
        advantage_normalization: Whether to normalize advantages

        # Training dynamics
        ppo_epochs: Number of PPO epochs per update
        num_mini_batches: Number of mini-batches per epoch
        target_kl: KL divergence threshold for early stopping

        # Value function
        use_value_clipping: Whether to clip value function updates
        value_clip_range: Clipping range for value function
        value_loss_weight: Weight for combining clipped vs unclipped value loss

        # Learning rates (can be different for actor and critic)
        critic_learning_rate: Learning rate for critic (None = same as actor)

        # Numerical stability
        eps: Small constant for numerical stability
        max_grad_norm: Maximum gradient norm for clipping

        # Reproducibility
        shuffle_mini_batches: Whether to shuffle mini-batches
    """

    # Core PPO hyperparameters
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Advantage estimation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_gae: bool = True
    advantage_normalization: bool = True

    # Training dynamics
    ppo_epochs: int = 4
    num_mini_batches: int = 4
    target_kl: Optional[float] = None

    # Value function
    use_value_clipping: bool = True
    value_clip_range: float = 0.2
    value_loss_weight: float = 0.5

    # Learning rates
    critic_learning_rate: Optional[float] = None

    # Numerical stability
    eps: float = 1e-8
    max_grad_norm: float = 0.5

    # Reproducibility
    shuffle_mini_batches: bool = True

    # Token-level vs sequence-level loss
    token_level_loss: bool = True

    # Early stopping
    enable_early_stopping: bool = True
    kl_threshold: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        self._validate_ppo_config()

    def _validate_ppo_config(self) -> None:
        """Validate PPO-specific configuration parameters."""
        if self.clip_ratio <= 0:
            raise ValueError(f"clip_ratio must be positive, got {self.clip_ratio}")

        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")

        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be between 0 and 1, got {self.gae_lambda}")

        if self.ppo_epochs <= 0:
            raise ValueError(f"ppo_epochs must be positive, got {self.ppo_epochs}")

        if self.num_mini_batches <= 0:
            raise ValueError(f"num_mini_batches must be positive, got {self.num_mini_batches}")

        if self.value_loss_coeff < 0:
            raise ValueError(f"value_loss_coeff must be non-negative, got {self.value_loss_coeff}")

        if self.entropy_coeff < 0:
            raise ValueError(f"entropy_coeff must be non-negative, got {self.entropy_coeff}")


class PPOAdvantageEstimator:
    """
    Generalized Advantage Estimation (GAE) for PPO.
    
    Computes advantages using the value function and GAE for bias-variance trade-off.
    """

    def __init__(self, config: PPOConfig):
        self.config = config

    def compute_advantages_and_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        next_values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Rewards [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            next_values: Next step values [batch_size, seq_len] (optional)
            dones: Done flags [batch_size, seq_len] (optional)

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape

        if next_values is None:
            # Use next time step values, assuming zero for final step
            next_values = torch.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]

        if self.config.use_gae:
            advantages, returns = self._compute_gae(
                rewards, values, next_values, attention_mask, dones
            )
        else:
            # Simple TD residual
            advantages, returns = self._compute_td_residual(
                rewards, values, next_values, attention_mask, dones
            )

        # Normalize advantages if enabled
        if self.config.advantage_normalization:
            advantages = self._normalize_advantages(advantages, attention_mask)

        return advantages, returns

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        attention_mask: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Process each sequence independently
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            seq_len_i = mask.sum().item()

            if seq_len_i == 0:
                continue

            seq_rewards = rewards[i][:seq_len_i]
            seq_values = values[i][:seq_len_i]
            seq_next_values = next_values[i][:seq_len_i]
            seq_dones = dones[i][:seq_len_i] if dones is not None else torch.zeros(seq_len_i)

            # Compute TD errors
            if dones is not None:
                td_targets = seq_rewards + self.config.gamma * seq_next_values * (1 - seq_dones)
            else:
                td_targets = seq_rewards + self.config.gamma * seq_next_values

            td_errors = td_targets - seq_values

            # Compute GAE advantages
            seq_advantages = torch.zeros_like(seq_rewards)
            running_gae = 0.0

            for t in reversed(range(seq_len_i)):
                if t == seq_len_i - 1:
                    next_advantage = 0.0
                else:
                    next_advantage = seq_advantages[t + 1]

                delta = td_errors[t]
                if dones is not None and seq_dones[t]:
                    running_gae = delta
                else:
                    running_gae = delta + self.config.gamma * self.config.gae_lambda * next_advantage

                seq_advantages[t] = running_gae

            # Compute returns
            seq_returns = seq_advantages + seq_values

            # Store results
            advantages[i][:seq_len_i] = seq_advantages
            returns[i][:seq_len_i] = seq_returns

        return advantages, returns

    def _compute_td_residual(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        attention_mask: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute simple TD residual advantages."""
        if dones is not None:
            returns = rewards + self.config.gamma * next_values * (1 - dones)
        else:
            returns = rewards + self.config.gamma * next_values

        advantages = returns - values

        # Apply attention mask
        advantages = advantages * attention_mask
        returns = returns * attention_mask

        return advantages, returns

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize advantages to have zero mean and unit variance."""
        valid_advantages = advantages * attention_mask
        valid_count = attention_mask.sum()

        if valid_count > 1:
            mean = valid_advantages.sum() / valid_count

            # Compute variance
            centered = (advantages - mean) * attention_mask
            variance = (centered ** 2).sum() / valid_count
            std = torch.sqrt(variance + self.config.eps)

            # Normalize
            normalized = (advantages - mean) / (std + self.config.eps)
            return normalized * attention_mask

        return advantages


class PPOValueFunction:
    """
    Value function for PPO critic.
    
    Can be either a separate network or use the model's hidden states.
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.value_head = None
        self.value_optimizer = None
        self._is_setup = False

    def setup(self, hidden_size: int, device: str) -> None:
        """Setup value function network."""
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        ).to(device)

        # Setup optimizer for value function
        lr = self.config.critic_learning_rate or self.config.learning_rate
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=lr,
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )

        self._is_setup = True
        logger.info(f"Setup PPO value function with hidden_size={hidden_size}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through value function."""
        if not self._is_setup:
            raise RuntimeError("Value function must be setup before calling forward()")

        values = self.value_head(hidden_states).squeeze(-1)
        return values

    def compute_value_loss(
        self,
        predicted_values: torch.Tensor,
        target_returns: torch.Tensor,
        old_values: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute value function loss with optional clipping."""
        if self.config.use_value_clipping:
            # Clipped value loss (similar to policy clipping)
            value_clip_range = self.config.value_clip_range
            
            # Unclipped loss
            value_loss_unclipped = (predicted_values - target_returns) ** 2
            
            # Clipped values
            clipped_values = old_values + torch.clamp(
                predicted_values - old_values,
                -value_clip_range,
                value_clip_range
            )
            
            # Clipped loss
            value_loss_clipped = (clipped_values - target_returns) ** 2
            
            # Take maximum (most conservative)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            # Standard MSE loss
            value_loss = (predicted_values - target_returns) ** 2

        # Apply attention mask and average
        masked_loss = value_loss * attention_mask
        loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)

        # Compute metrics
        metrics = {
            "value_loss": loss.item(),
            "predicted_values_mean": predicted_values.mean().item(),
            "target_returns_mean": target_returns.mean().item(),
            "value_mse": F.mse_loss(predicted_values, target_returns).item()
        }

        return loss, metrics

    def get_state_dict(self) -> Optional[Dict[str, Any]]:
        """Get value function state dict."""
        if self.value_head is not None:
            return {
                'value_head': self.value_head.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict()
            }
        return None

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load value function state dict."""
        if self.value_head is not None and 'value_head' in state_dict:
            self.value_head.load_state_dict(state_dict['value_head'])
        
        if self.value_optimizer is not None and 'value_optimizer' in state_dict:
            self.value_optimizer.load_state_dict(state_dict['value_optimizer'])


class PPOLoss:
    """
    PPO loss computation with clipped surrogate objective.
    """

    def __init__(self, config: PPOConfig):
        self.config = config

    def compute_policy_loss(
        self,
        log_probs_old: torch.Tensor,
        log_probs_new: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO clipped policy loss.

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

        # Clipped surrogate objective
        clip_range = self.config.clip_ratio
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        
        # Take minimum (most conservative)
        policy_loss = -torch.min(surr1, surr2)

        # Apply attention mask and compute average
        if self.config.token_level_loss:
            # Token-level loss
            masked_loss = policy_loss * attention_mask
            loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
        else:
            # Sequence-level loss
            seq_losses = (policy_loss * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            loss = seq_losses.mean()

        # Compute metrics
        metrics = self._compute_policy_metrics(ratio, advantages, attention_mask, log_ratio)

        return loss, metrics

    def compute_entropy_loss(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor
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

    def _compute_policy_metrics(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
        log_ratio: torch.Tensor
    ) -> Dict[str, float]:
        """Compute policy training metrics."""
        valid_mask = attention_mask.bool()

        # Basic ratio statistics
        valid_ratios = ratio[valid_mask]
        valid_log_ratios = log_ratio[valid_mask]
        valid_advantages = advantages[valid_mask]

        # Clipping statistics
        clip_range = self.config.clip_ratio
        clipped_lower = (valid_ratios < (1 - clip_range)).float().mean().item()
        clipped_upper = (valid_ratios > (1 + clip_range)).float().mean().item()
        
        # KL divergence approximation
        approx_kl = valid_log_ratios.mean().item()

        metrics = {
            "ratio_mean": valid_ratios.mean().item(),
            "ratio_std": valid_ratios.std().item(),
            "clipped_lower_frac": clipped_lower,
            "clipped_upper_frac": clipped_upper,
            "approx_kl": approx_kl,
            "advantage_mean": valid_advantages.mean().item(),
            "advantage_std": valid_advantages.std().item(),
        }

        return metrics


class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent destructively large policy updates while maintaining sample efficiency.

    Key features:
    1. Clipped surrogate objective for stable training
    2. Value function learning with GAE
    3. Multiple epochs of mini-batch updates
    4. Entropy regularization
    5. Early stopping based on KL divergence

    Example:
        >>> config = PPOConfig(
        ...     clip_ratio=0.2,
        ...     ppo_epochs=4,
        ...     num_mini_batches=4
        ... )
        >>> ppo = PPO(config)
        >>> ppo.setup(model=policy_model)
        >>> output = ppo.step(batch)
    """

    def __init__(self, config: PPOConfig):
        if not isinstance(config, PPOConfig):
            raise TypeError(f"config must be PPOConfig, got {type(config)}")

        super().__init__(config)

        self.advantage_estimator = PPOAdvantageEstimator(config)
        self.value_function = PPOValueFunction(config)
        self.loss_computer = PPOLoss(config)

        self._step_count = 0
        self._is_value_function_setup = False

        logger.info(
            f"Initialized PPO with clip_ratio={config.clip_ratio}, "
            f"ppo_epochs={config.ppo_epochs}, num_mini_batches={config.num_mini_batches}"
        )

    def setup(self, model, optimizer=None, **kwargs) -> None:
        """
        Setup PPO with model and optimizer.

        Args:
            model: Policy model to train
            optimizer: Optimizer (optional)
            **kwargs: Additional setup parameters
        """
        super().setup(model, optimizer, **kwargs)

        # Setup value function
        if hasattr(self._model, 'config') and hasattr(self._model.config, 'hidden_size'):
            hidden_size = self._model.config.hidden_size
        else:
            # Assume reasonable default
            hidden_size = 768
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.value_function.setup(hidden_size, self.config.device)
        self._is_value_function_setup = True

        logger.info("PPO setup complete")

    def step(self, batch: Dict[str, Any], **kwargs) -> AlgorithmOutput:
        """
        Perform one PPO training step.

        Args:
            batch: Training batch containing:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - rewards: Rewards [batch_size, seq_len] or [batch_size]
                - old_log_probs: Old policy log probs [batch_size, seq_len] (optional)
                - old_values: Old value estimates [batch_size, seq_len] (optional)
            **kwargs: Additional parameters

        Returns:
            AlgorithmOutput containing loss, metrics, and logs
        """
        if not self._is_setup:
            raise RuntimeError("PPO must be setup before calling step()")

        if not self._is_value_function_setup:
            raise RuntimeError("Value function must be setup before training")

        self._validate_batch(batch)

        # Move to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.config.device)

        # Set model to training mode
        self._model.train()

        # Collect old policy data if not provided
        if 'old_log_probs' not in batch or 'old_values' not in batch:
            with torch.no_grad():
                old_outputs = self._collect_old_policy_data(batch)
                if 'old_log_probs' not in batch:
                    batch['old_log_probs'] = old_outputs['old_log_probs']
                if 'old_values' not in batch:
                    batch['old_values'] = old_outputs['old_values']

        # PPO training with multiple epochs and mini-batches
        total_loss = 0.0
        total_metrics = {}
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            mini_batches = self._create_mini_batches(batch)

            epoch_kl = 0.0
            epoch_updates = 0

            for mini_batch in mini_batches:
                # Forward pass and compute loss
                algorithm_output = self._compute_ppo_loss(mini_batch, **kwargs)

                if algorithm_output.loss is not None:
                    # Backward pass
                    self._optimizer.zero_grad()
                    if self.value_function.value_optimizer is not None:
                        self.value_function.value_optimizer.zero_grad()

                    algorithm_output.loss.backward()

                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            self.config.max_grad_norm
                        )
                        if self.value_function.value_head is not None:
                            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.value_function.value_head.parameters(),
                                self.config.max_grad_norm
                            )
                        algorithm_output.update_metrics(
                            grad_norm=grad_norm.item(),
                            value_grad_norm=value_grad_norm.item() if 'value_grad_norm' in locals() else 0.0
                        )

                    # Optimization step
                    self._optimizer.step()
                    if self.value_function.value_optimizer is not None:
                        self.value_function.value_optimizer.step()

                    # Accumulate metrics
                    total_loss += algorithm_output.loss.item()
                    for key, value in algorithm_output.metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value

                    # Track KL for early stopping
                    epoch_kl += algorithm_output.metrics.get('approx_kl', 0.0)
                    epoch_updates += 1
                    num_updates += 1

            # Early stopping based on KL divergence
            if (self.config.enable_early_stopping and 
                self.config.target_kl is not None and 
                epoch_updates > 0):
                
                avg_kl = epoch_kl / epoch_updates
                if avg_kl > self.config.target_kl:
                    logger.info(f"Early stopping at epoch {epoch} due to KL divergence: {avg_kl:.4f}")
                    break

        # Average metrics
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            avg_metrics = {k: v / num_updates for k, v in total_metrics.items()}
        else:
            avg_loss = 0.0
            avg_metrics = {}

        self._step_count += 1

        # Create final output
        output = AlgorithmOutput(
            loss=torch.tensor(avg_loss),
            metrics=avg_metrics,
            logs={
                'step': self._step_count,
                'algorithm': 'PPO',
                'ppo_epochs_completed': min(epoch + 1, self.config.ppo_epochs),
                'num_ppo_updates': num_updates
            }
        )

        return output

    def _collect_old_policy_data(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Collect old policy log probabilities and values."""
        with torch.no_grad():
            # Forward pass through policy model
            model_outputs = self._model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            logits = model_outputs['logits']
            hidden_states = model_outputs.get('hidden_states', model_outputs.get('last_hidden_state'))

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=batch['input_ids'].unsqueeze(-1)
            ).squeeze(-1)

            # Compute values
            old_values = self.value_function.forward(hidden_states)

            return {
                'old_log_probs': old_log_probs,
                'old_values': old_values
            }

    def _create_mini_batches(self, batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Create mini-batches for PPO updates."""
        batch_size = batch['input_ids'].shape[0]
        mini_batch_size = max(1, batch_size // self.config.num_mini_batches)

        # Create indices
        indices = list(range(batch_size))
        if self.config.shuffle_mini_batches:
            random.shuffle(indices)

        mini_batches = []
        for i in range(0, batch_size, mini_batch_size):
            mini_batch_indices = indices[i:i + mini_batch_size]

            mini_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    mini_batch[key] = value[mini_batch_indices]
                else:
                    mini_batch[key] = value

            mini_batches.append(mini_batch)

        return mini_batches

    def _compute_ppo_loss(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> AlgorithmOutput:
        """Compute PPO loss for a mini-batch."""
        # Extract required tensors
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        rewards = batch.get('rewards')
        old_log_probs = batch.get('old_log_probs')
        old_values = batch.get('old_values')

        if rewards is None:
            raise ValueError("batch must contain 'rewards'")
        if old_log_probs is None:
            raise ValueError("batch must contain 'old_log_probs'")
        if old_values is None:
            raise ValueError("batch must contain 'old_values'")

        # Forward pass through policy model
        model_outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = model_outputs['logits']
        hidden_states = model_outputs.get('hidden_states', model_outputs.get('last_hidden_state'))

        # Compute current log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        new_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Compute current values
        new_values = self.value_function.forward(hidden_states)

        # Handle reward dimensions
        if rewards.dim() == 1:
            # Expand to sequence length if needed
            rewards = rewards.unsqueeze(-1).expand(-1, input_ids.shape[1])

        # Compute advantages and returns
        advantages, returns = self.advantage_estimator.compute_advantages_and_returns(
            rewards=rewards,
            values=old_values,  # Use old values for advantage computation
            attention_mask=attention_mask
        )

        # Policy loss
        policy_loss, policy_metrics = self.loss_computer.compute_policy_loss(
            log_probs_old=old_log_probs,
            log_probs_new=new_log_probs,
            advantages=advantages,
            attention_mask=attention_mask
        )

        # Value loss
        value_loss, value_metrics = self.value_function.compute_value_loss(
            predicted_values=new_values,
            target_returns=returns,
            old_values=old_values,
            attention_mask=attention_mask
        )

        # Entropy loss
        entropy_loss = self.loss_computer.compute_entropy_loss(
            logits=logits,
            attention_mask=attention_mask
        )

        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_loss_coeff * value_loss + 
            entropy_loss
        )

        # Prepare output
        output = AlgorithmOutput(loss=total_loss)

        # Add metrics
        output.update_metrics(**policy_metrics)
        output.update_metrics(**value_metrics)
        output.update_metrics(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            total_loss=total_loss.item()
        )

        return output

    def compute_loss(
        self,
        batch: Dict[str, Any],
        model_outputs: Any,
        **kwargs
    ) -> AlgorithmOutput:
        """
        Compute PPO loss (required by base class).
        
        Note: This is primarily for compatibility. The main training
        happens in step() with multiple epochs and mini-batches.
        """
        return self._compute_ppo_loss(batch, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate sequences using the trained policy model."""
        if not self._is_setup:
            raise RuntimeError("PPO must be setup before calling generate()")

        self._model.eval()

        with torch.no_grad():
            # Use model's generate method if available
            if hasattr(self._model, 'generate'):
                generated = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    **kwargs
                )
            else:
                # Simple greedy generation fallback
                generated = self._simple_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    **kwargs
                )

        # Compute log probabilities for generated sequences
        model_outputs = self._model(
            input_ids=generated,
            attention_mask=torch.ones_like(generated)
        )

        log_probs = F.log_softmax(model_outputs['logits'], dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=generated.unsqueeze(-1)
        ).squeeze(-1)

        return {
            'sequences': generated,
            'log_probs': token_log_probs,
            'attention_mask': torch.ones_like(generated)
        }

    def _simple_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Simple greedy/sampling generation method."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_length - seq_len):
            outputs = self._model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated)
            )

            logits = outputs['logits'][:, -1, :] / temperature

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
        self,
        eval_dataloader,
        num_eval_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate the model on a validation dataset."""
        if not self._is_setup:
            raise RuntimeError("PPO must be setup before calling evaluate()")

        self._model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if num_eval_steps is not None and i >= num_eval_steps:
                    break

                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.config.device)

                # Ensure old policy data is available
                if 'old_log_probs' not in batch or 'old_values' not in batch:
                    old_outputs = self._collect_old_policy_data(batch)
                    batch.update(old_outputs)

                # Compute losses
                algorithm_output = self._compute_ppo_loss(batch)

                # Accumulate metrics
                total_loss += algorithm_output.loss.item()
                total_policy_loss += algorithm_output.metrics.get('policy_loss', 0.0)
                total_value_loss += algorithm_output.metrics.get('value_loss', 0.0)
                total_entropy_loss += algorithm_output.metrics.get('entropy_loss', 0.0)
                num_batches += 1

        # Compute averages
        eval_metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_policy_loss': total_policy_loss / num_batches,
            'eval_value_loss': total_value_loss / num_batches,
            'eval_entropy_loss': total_entropy_loss / num_batches,
        }

        logger.info(f"Evaluation results: {eval_metrics}")
        return eval_metrics

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save PPO checkpoint with algorithm-specific state."""
        additional_state = {
            'step_count': self._step_count,
            'value_function_state': self.value_function.get_state_dict(),
        }

        super().save_checkpoint(path, **additional_state, **kwargs)

    def load_checkpoint(self, path: str, **kwargs) -> Dict[str, Any]:
        """Load PPO checkpoint and restore algorithm-specific state."""
        checkpoint = super().load_checkpoint(path, **kwargs)

        # Restore PPO-specific state
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']

        if 'value_function_state' in checkpoint and checkpoint['value_function_state'] is not None:
            self.value_function.load_state_dict(checkpoint['value_function_state'])

        return checkpoint

    def get_info(self) -> Dict[str, Any]:
        """Get PPO algorithm information."""
        info = super().get_info()

        # Add PPO-specific information
        ppo_info = {
            'clip_ratio': self.config.clip_ratio,
            'ppo_epochs': self.config.ppo_epochs,
            'num_mini_batches': self.config.num_mini_batches,
            'use_gae': self.config.use_gae,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'step_count': self._step_count,
            'value_function_setup': self._is_value_function_setup,
        }

        info.update(ppo_info)
        return info


# Helper functions for PPO usage


def create_ppo_config(**kwargs) -> PPOConfig:
    """
    Create a PPO configuration with sensible defaults.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        PPOConfig instance

    Example:
        >>> config = create_ppo_config(
        ...     clip_ratio=0.2,
        ...     ppo_epochs=4,
        ...     num_mini_batches=4
        ... )
    """
    return PPOConfig(**kwargs)


def create_ppo_algorithm(
    config: Optional[PPOConfig] = None,
    **kwargs
) -> PPO:
    """
    Create a PPO algorithm instance.

    Args:
        config: PPO configuration (optional)
        **kwargs: Configuration parameters if config is None

    Returns:
        PPO algorithm instance

    Example:
        >>> ppo = create_ppo_algorithm(
        ...     clip_ratio=0.2,
        ...     ppo_epochs=4
        ... )
    """
    if config is None:
        config = PPOConfig(**kwargs)

    return PPO(config)


def load_ppo_from_checkpoint(
    checkpoint_path: str,
    model=None,
    optimizer=None
) -> PPO:
    """
    Load PPO algorithm from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Policy model (optional)
        optimizer: Optimizer (optional)

    Returns:
        PPO algorithm instance loaded from checkpoint

    Example:
        >>> ppo = load_ppo_from_checkpoint(
        ...     "checkpoints/ppo_step_1000.pt",
        ...     model=policy_model
        ... )
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    config = PPOConfig.from_dict(config_dict)

    # Create algorithm
    ppo = PPO(config)

    # Setup if models provided
    if model is not None:
        ppo.setup(model=model, optimizer=optimizer)

    # Load state
    ppo.load_checkpoint(checkpoint_path)

    return ppo


# Export main classes and functions
__all__ = [
    'PPO',
    'PPOConfig',
    'PPOAdvantageEstimator',
    'PPOValueFunction',
    'PPOLoss',
    'create_ppo_config',
    'create_ppo_algorithm',
    'load_ppo_from_checkpoint'
]