"""
REINFORCE (REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility) algorithm implementation.

REINFORCE is a fundamental policy gradient algorithm that learns by directly optimizing the policy
using Monte Carlo estimates of the policy gradient. It's the foundational algorithm for many
modern policy gradient methods including PPO and DAPO.

Key Features:
- Pure policy gradient method (no value function required)
- Monte Carlo return estimation
- Optional baseline for variance reduction
- Entropy regularization for exploration
- Support for both episodic and continuing tasks
- Variance reduction techniques (baseline, reward normalization)

References:
    - "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (Sutton et al., 2000)
    - "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)
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
class REINFORCEConfig(AlgorithmConfig):
    """
    Configuration for REINFORCE algorithm.
    
    This configuration includes all hyperparameters specific to REINFORCE,
    including return computation, baseline options, and variance reduction techniques.
    
    Args:
        # Core REINFORCE hyperparameters
        gamma: Discount factor for future rewards
        use_baseline: Whether to use a baseline to reduce variance
        baseline_type: Type of baseline ('moving_average', 'value_function', 'none')
        normalize_returns: Whether to normalize returns across batch
        
        # Return computation
        return_normalization: Method for normalizing returns ('batch', 'running', 'none')
        reward_scaling: Scale factor for rewards before computing returns
        reward_clipping: Whether to clip rewards
        reward_clip_range: Range for clipping rewards
        
        # Baseline parameters (if using baseline)
        baseline_learning_rate: Learning rate for baseline learning
        baseline_momentum: Momentum for moving average baseline
        baseline_update_frequency: How often to update baseline
        
        # Entropy regularization
        entropy_coeff: Coefficient for entropy regularization
        entropy_decay: Decay factor for entropy coefficient
        min_entropy_coeff: Minimum entropy coefficient
        
        # Variance reduction
        use_reward_to_go: Use reward-to-go instead of full episode returns
        use_causality_mask: Only include future rewards (causality principle)
        advantage_normalization: Normalize advantages within batch
        
        # Training dynamics
        sequence_level_loss: Use sequence-level averaging instead of token-level
        only_final_reward: Only apply reward to final token (for episodic tasks)
        
        # Numerical stability
        eps: Small constant for numerical stability
        max_grad_norm: Maximum gradient norm for clipping
        return_clip_range: Range for clipping returns
        
        # Logging and debugging
        log_return_stats: Whether to log return statistics
        track_baseline_performance: Track baseline prediction accuracy
    """
    
    # Core REINFORCE hyperparameters
    gamma: float = 0.99
    use_baseline: bool = True
    baseline_type: str = "moving_average"  # 'moving_average', 'value_function', 'none'
    normalize_returns: bool = True
    
    # Return computation
    return_normalization: str = "batch"  # 'batch', 'running', 'none'
    reward_scaling: float = 1.0
    reward_clipping: bool = False
    reward_clip_range: Tuple[float, float] = (-10.0, 10.0)
    
    # Baseline parameters
    baseline_learning_rate: Optional[float] = None  # None = same as main lr
    baseline_momentum: float = 0.9
    baseline_update_frequency: int = 1
    
    # Entropy regularization
    entropy_coeff: float = 0.01
    entropy_decay: float = 1.0  # No decay by default
    min_entropy_coeff: float = 0.0
    
    # Variance reduction
    use_reward_to_go: bool = True
    use_causality_mask: bool = True
    advantage_normalization: bool = True
    
    # Training dynamics
    sequence_level_loss: bool = False
    only_final_reward: bool = False
    
    # Numerical stability
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    return_clip_range: Optional[Tuple[float, float]] = None
    
    # Logging and debugging
    log_return_stats: bool = True
    track_baseline_performance: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_reinforce_config()
    
    def _validate_reinforce_config(self) -> None:
        """Validate REINFORCE-specific configuration parameters."""
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")
        
        if self.baseline_type not in ["moving_average", "value_function", "none"]:
            raise ValueError(
                f"baseline_type must be one of ['moving_average', 'value_function', 'none'], "
                f"got {self.baseline_type}"
            )
        
        if self.return_normalization not in ["batch", "running", "none"]:
            raise ValueError(
                f"return_normalization must be one of ['batch', 'running', 'none'], "
                f"got {self.return_normalization}"
            )
        
        if self.reward_scaling <= 0:
            raise ValueError(f"reward_scaling must be positive, got {self.reward_scaling}")
        
        if self.baseline_momentum < 0 or self.baseline_momentum > 1:
            raise ValueError(
                f"baseline_momentum must be between 0 and 1, got {self.baseline_momentum}"
            )
        
        if self.entropy_coeff < 0:
            raise ValueError(f"entropy_coeff must be non-negative, got {self.entropy_coeff}")
        
        if not 0 < self.entropy_decay <= 1:
            raise ValueError(f"entropy_decay must be in (0, 1], got {self.entropy_decay}")


class REINFORCEReturns:
    """
    Return computation for REINFORCE algorithm.
    
    Handles Monte Carlo return estimation with various variance reduction techniques.
    """
    
    def __init__(self, config: REINFORCEConfig):
        self.config = config
        self.running_mean = 0.0
        self.running_std = 1.0
        self.return_count = 0
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Monte Carlo returns with optional variance reduction.
        
        Args:
            rewards: Rewards tensor [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len] (optional)
            
        Returns:
            Returns tensor [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        
        # Apply reward scaling and clipping if enabled
        scaled_rewards = rewards * self.config.reward_scaling
        if self.config.reward_clipping:
            scaled_rewards = torch.clamp(
                scaled_rewards,
                self.config.reward_clip_range[0],
                self.config.reward_clip_range[1]
            )
        
        # Compute returns
        if self.config.use_reward_to_go:
            returns = self._compute_reward_to_go(scaled_rewards, attention_mask, dones)
        else:
            returns = self._compute_full_episode_returns(scaled_rewards, attention_mask, dones)
        
        # Apply causality mask if enabled
        if self.config.use_causality_mask:
            returns = self._apply_causality_mask(returns, attention_mask)
        
        # Handle only final reward case
        if self.config.only_final_reward:
            returns = self._apply_final_reward_only(returns, attention_mask)
        
        # Normalize returns if enabled
        if self.config.normalize_returns:
            returns = self._normalize_returns(returns, attention_mask)
        
        # Clip returns if specified
        if self.config.return_clip_range is not None:
            returns = torch.clamp(
                returns,
                self.config.return_clip_range[0],
                self.config.return_clip_range[1]
            )
        
        return returns
    
    def _compute_reward_to_go(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reward-to-go (discounted sum of future rewards)."""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            seq_rewards = rewards[i][mask]
            seq_dones = dones[i][mask] if dones is not None else None
            
            seq_returns = torch.zeros_like(seq_rewards)
            running_return = 0.0
            
            # Compute returns in reverse order
            for t in reversed(range(len(seq_rewards))):
                if seq_dones is not None and seq_dones[t]:
                    running_return = 0.0  # Reset at episode boundaries
                
                running_return = seq_rewards[t] + self.config.gamma * running_return
                seq_returns[t] = running_return
            
            returns[i][mask] = seq_returns
        
        return returns
    
    def _compute_full_episode_returns(
        self,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute full episode returns (same return for all timesteps in episode)."""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            seq_rewards = rewards[i][mask]
            
            # Compute discounted sum of all rewards in episode
            episode_return = 0.0
            for t, reward in enumerate(seq_rewards):
                episode_return += (self.config.gamma ** t) * reward
            
            # Assign same return to all timesteps
            returns[i][mask] = episode_return
        
        return returns
    
    def _apply_causality_mask(
        self,
        returns: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply causality mask to ensure only future rewards influence current action."""
        # This is already handled in reward-to-go computation
        # For full episode returns, we might want to modify this
        return returns
    
    def _apply_final_reward_only(
        self,
        returns: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply reward only to the final token in each sequence."""
        final_returns = torch.zeros_like(returns)
        
        for i in range(returns.shape[0]):
            mask = attention_mask[i].bool()
            if mask.any():
                # Find last valid position
                last_pos = mask.nonzero(as_tuple=True)[0][-1]
                final_returns[i, last_pos] = returns[i, last_pos]
        
        return final_returns
    
    def _normalize_returns(
        self,
        returns: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize returns according to configuration."""
        if self.config.return_normalization == "none":
            return returns
        
        valid_returns = returns * attention_mask
        valid_count = attention_mask.sum()
        
        if valid_count == 0:
            return returns
        
        if self.config.return_normalization == "batch":
            # Batch normalization
            mean = valid_returns.sum() / valid_count
            variance = ((returns - mean) * attention_mask).pow(2).sum() / valid_count
            std = torch.sqrt(variance + self.config.eps)
            
            normalized = (returns - mean) / (std + self.config.eps)
            return normalized * attention_mask
        
        elif self.config.return_normalization == "running":
            # Running normalization
            batch_mean = valid_returns.sum() / valid_count
            batch_var = ((returns - batch_mean) * attention_mask).pow(2).sum() / valid_count
            
            # Update running statistics
            self.return_count += 1
            alpha = 1.0 / self.return_count
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean.item()
            self.running_std = (1 - alpha) * self.running_std + alpha * torch.sqrt(batch_var + self.config.eps).item()
            
            normalized = (returns - self.running_mean) / (self.running_std + self.config.eps)
            return normalized * attention_mask
        
        return returns


class REINFORCEBaseline:
    """
    Baseline estimation for REINFORCE to reduce variance.
    
    Supports different baseline types including moving average and value function.
    """
    
    def __init__(self, config: REINFORCEConfig):
        self.config = config
        self.baseline_type = config.baseline_type
        
        # Moving average baseline
        self.moving_average = 0.0
        self.update_count = 0
        
        # Value function baseline (if used)
        self.value_function = None
        self.value_optimizer = None
        
        # Performance tracking
        self.baseline_mse = 0.0
        self.tracking_count = 0
    
    def setup_value_function(self, hidden_size: int) -> None:
        """Setup value function for baseline estimation."""
        if self.baseline_type == "value_function":
            self.value_function = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
            lr = self.config.baseline_learning_rate or self.config.learning_rate
            self.value_optimizer = torch.optim.AdamW(
                self.value_function.parameters(),
                lr=lr,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
    
    def estimate_baseline(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate baseline values.
        
        Args:
            hidden_states: Hidden states from model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Baseline estimates [batch_size, seq_len]
        """
        if not self.config.use_baseline or self.baseline_type == "none":
            if hidden_states is not None:
                return torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
            else:
                raise ValueError("Need hidden_states or attention_mask for baseline shape")
        
        if self.baseline_type == "moving_average":
            # Simple moving average baseline
            if hidden_states is not None:
                batch_size, seq_len = hidden_states.shape[:2]
                device = hidden_states.device
            else:
                batch_size, seq_len = attention_mask.shape
                device = attention_mask.device
            
            return torch.full(
                (batch_size, seq_len),
                self.moving_average,
                device=device
            )
        
        elif self.baseline_type == "value_function":
            # Value function baseline
            if hidden_states is None:
                raise ValueError("Value function baseline requires hidden_states")
            
            if self.value_function is None:
                raise RuntimeError("Value function not initialized. Call setup_value_function first.")
            
            # Move value function to same device as hidden states
            if next(self.value_function.parameters()).device != hidden_states.device:
                self.value_function.to(hidden_states.device)
            
            values = self.value_function(hidden_states).squeeze(-1)
            return values
        
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def update_baseline(
        self,
        returns: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update baseline estimates.
        
        Args:
            returns: Computed returns [batch_size, seq_len]
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of baseline metrics
        """
        if not self.config.use_baseline or self.baseline_type == "none":
            return {}
        
        metrics = {}
        
        if self.baseline_type == "moving_average":
            # Update moving average
            if attention_mask is not None:
                valid_returns = returns * attention_mask
                valid_count = attention_mask.sum()
                if valid_count > 0:
                    batch_mean = valid_returns.sum() / valid_count
                    
                    self.update_count += 1
                    momentum = self.config.baseline_momentum
                    self.moving_average = momentum * self.moving_average + (1 - momentum) * batch_mean.item()
                    
                    metrics["baseline_value"] = self.moving_average
                    metrics["baseline_batch_mean"] = batch_mean.item()
            
        elif self.baseline_type == "value_function":
            # Update value function
            if hidden_states is None or self.value_function is None:
                return metrics
            
            predicted_values = self.value_function(hidden_states).squeeze(-1)
            
            # Compute value loss
            if attention_mask is not None:
                value_loss = ((predicted_values - returns) ** 2) * attention_mask
                value_loss = value_loss.sum() / attention_mask.sum().clamp(min=1)
            else:
                value_loss = F.mse_loss(predicted_values, returns)
            
            # Update value function
            if self.update_count % self.config.baseline_update_frequency == 0:
                self.value_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)
                self.value_optimizer.step()
            
            metrics["value_loss"] = value_loss.item()
            metrics["predicted_values_mean"] = predicted_values.mean().item()
            
            # Track baseline performance if enabled
            if self.config.track_baseline_performance:
                with torch.no_grad():
                    if attention_mask is not None:
                        mse = ((predicted_values - returns) ** 2 * attention_mask).sum() / attention_mask.sum().clamp(min=1)
                    else:
                        mse = F.mse_loss(predicted_values, returns)
                    
                    self.tracking_count += 1
                    alpha = 1.0 / self.tracking_count
                    self.baseline_mse = (1 - alpha) * self.baseline_mse + alpha * mse.item()
                    
                    metrics["baseline_mse"] = self.baseline_mse
        
        self.update_count += 1
        return metrics


class REINFORCELoss:
    """
    REINFORCE loss computation with policy gradient and entropy regularization.
    """
    
    def __init__(self, config: REINFORCEConfig):
        self.config = config
        self.current_entropy_coeff = config.entropy_coeff
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        baseline: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute REINFORCE policy gradient loss.
        
        Args:
            log_probs: Log probabilities [batch_size, seq_len]
            returns: Monte Carlo returns [batch_size, seq_len]
            baseline: Baseline estimates [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute advantages
        advantages = returns - baseline
        
        # Normalize advantages if enabled
        if self.config.advantage_normalization:
            advantages = self._normalize_advantages(advantages, attention_mask)
        
        # REINFORCE loss: -log_prob * advantage
        policy_loss = -log_probs * advantages
        
        # Apply attention mask and compute average
        if self.config.sequence_level_loss:
            # Sequence-level loss
            seq_losses = (policy_loss * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            loss = seq_losses.mean()
        else:
            # Token-level loss
            masked_loss = policy_loss * attention_mask
            loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
        
        # Compute metrics
        metrics = self._compute_policy_metrics(log_probs, returns, advantages, attention_mask)
        
        return loss, metrics
    
    def compute_entropy_loss(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy regularization loss."""
        # Compute entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Apply mask and average
        masked_entropy = entropy * attention_mask
        avg_entropy = masked_entropy.sum() / attention_mask.sum().clamp(min=1)
        
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
    
    def _compute_policy_metrics(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute policy training metrics."""
        valid_mask = attention_mask.bool()
        
        # Basic statistics
        valid_log_probs = log_probs[valid_mask]
        valid_returns = returns[valid_mask]
        valid_advantages = advantages[valid_mask]
        
        metrics = {
            "log_prob_mean": valid_log_probs.mean().item(),
            "log_prob_std": valid_log_probs.std().item(),
            "return_mean": valid_returns.mean().item(),
            "return_std": valid_returns.std().item(),
            "advantage_mean": valid_advantages.mean().item(),
            "advantage_std": valid_advantages.std().item(),
            "entropy_coeff": self.current_entropy_coeff,
        }
        
        return metrics


class REINFORCE(BaseAlgorithm):
    """
    REINFORCE (Williams, 1992) policy gradient algorithm.
    
    REINFORCE is a Monte Carlo policy gradient method that learns by:
    1. Collecting complete episodes or sequences
    2. Computing returns using Monte Carlo estimation
    3. Using policy gradient theorem to update parameters
    4. Optional baseline to reduce variance
    
    Key features:
    - Pure policy gradient (no value function required for core algorithm)
    - Monte Carlo return estimation
    - Optional baseline for variance reduction
    - Entropy regularization
    - Support for both episodic and continuing tasks
    
    Example:
        >>> config = REINFORCEConfig(
        ...     gamma=0.99,
        ...     use_baseline=True,
        ...     baseline_type="moving_average"
        ... )
        >>> reinforce = REINFORCE(config)
        >>> reinforce.setup(model=policy_model)
        >>> output = reinforce.step(batch)
    """
    
    def __init__(self, config: REINFORCEConfig):
        if not isinstance(config, REINFORCEConfig):
            raise TypeError(f"config must be REINFORCEConfig, got {type(config)}")
        
        super().__init__(config)
        
        self.returns_computer = REINFORCEReturns(config)
        self.baseline = REINFORCEBaseline(config)
        self.loss_computer = REINFORCELoss(config)
        
        self._step_count = 0
        self._is_value_function_setup = False
        
        logger.info(
            f"Initialized REINFORCE with gamma={config.gamma}, "
            f"baseline={config.baseline_type}, use_reward_to_go={config.use_reward_to_go}"
        )
    
    def setup(self, model, optimizer=None, **kwargs) -> None:
        """
        Setup REINFORCE with model and optimizer.
        
        Args:
            model: Policy model to train
            optimizer: Optimizer (optional)
            **kwargs: Additional setup parameters
        """
        super().setup(model, optimizer, **kwargs)
        
        # Setup value function baseline if needed
        if (self.config.baseline_type == "value_function" and 
            hasattr(self._model, 'config') and 
            hasattr(self._model.config, 'hidden_size')):
            
            hidden_size = self._model.config.hidden_size
            self.baseline.setup_value_function(hidden_size)
            self._is_value_function_setup = True
            
            logger.info(f"Setup value function baseline with hidden_size={hidden_size}")
        
        logger.info("REINFORCE setup complete")
    
    def step(self, batch: Dict[str, Any], **kwargs) -> AlgorithmOutput:
        """
        Perform one REINFORCE training step.
        
        Args:
            batch: Training batch containing:
                - input_ids: Token IDs [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - rewards: Rewards [batch_size, seq_len] or [batch_size]
                - dones: Done flags [batch_size, seq_len] (optional)
            **kwargs: Additional parameters
            
        Returns:
            AlgorithmOutput containing loss, metrics, and logs
        """
        if not self._is_setup:
            raise RuntimeError("REINFORCE must be setup before calling step()")
        
        self._validate_batch(batch)
        
        # Move to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.config.device)
        
        # Set model to training mode
        self._model.train()
        
        # Forward pass through policy model
        model_outputs = self._model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Compute loss and metrics
        algorithm_output = self.compute_loss(batch, model_outputs, **kwargs)
        
        # Backward pass and optimization
        if algorithm_output.loss is not None:
            self._optimizer.zero_grad()
            algorithm_output.loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self.config.max_grad_norm
                )
                algorithm_output.update_metrics(grad_norm=grad_norm.item())
            
            self._optimizer.step()
        
        # Update entropy coefficient
        self.loss_computer.update_entropy_coeff()
        
        self._step_count += 1
        algorithm_output.update_logs(step=self._step_count)
        
        return algorithm_output
    
    def compute_loss(
        self,
        batch: Dict[str, Any],
        model_outputs: Any,
        **kwargs
    ) -> AlgorithmOutput:
        """
        Compute REINFORCE loss with all components.
        
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
        dones = batch.get("dones")
        
        if rewards is None:
            raise ValueError("batch must contain 'rewards'")
        
        # Get model outputs
        logits = model_outputs.get("logits")
        hidden_states = model_outputs.get("hidden_states")
        
        if logits is None:
            raise ValueError("model_outputs must contain 'logits'")
        
        # Compute log probabilities for taken actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Handle reward dimensions
        if rewards.dim() == 1:
            # Expand sequence-level rewards to token level
            rewards = rewards.unsqueeze(-1).expand(-1, input_ids.shape[1])
        
        # Compute returns
        returns = self.returns_computer.compute_returns(
            rewards=rewards,
            attention_mask=attention_mask,
            dones=dones
        )
        
        # Estimate baseline
        baseline = self.baseline.estimate_baseline(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Policy loss
        policy_loss, policy_metrics = self.loss_computer.compute_policy_loss(
            log_probs=action_log_probs,
            returns=returns,
            baseline=baseline,
            attention_mask=attention_mask
        )
        
        # Entropy loss
        entropy_loss = self.loss_computer.compute_entropy_loss(
            logits=logits,
            attention_mask=attention_mask
        )
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Update baseline
        baseline_metrics = self.baseline.update_baseline(
            returns=returns,
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Prepare output
        output = AlgorithmOutput(loss=total_loss)
        
        # Add metrics
        output.update_metrics(**policy_metrics)
        output.update_metrics(**baseline_metrics)
        output.update_metrics(
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            total_loss=total_loss.item()
        )
        
        # Add return statistics if enabled
        if self.config.log_return_stats:
            valid_returns = returns * attention_mask
            valid_count = attention_mask.sum()
            if valid_count > 0:
                output.update_metrics(
                    returns_mean=valid_returns.sum().item() / valid_count.item(),
                    returns_std=((returns - valid_returns.sum() / valid_count) ** 2 * attention_mask).sum().sqrt().item() / valid_count.item(),
                    returns_min=returns[attention_mask.bool()].min().item(),
                    returns_max=returns[attention_mask.bool()].max().item()
                )
        
        # Add algorithm-specific logs
        output.update_logs(
            algorithm="REINFORCE",
            gamma=self.config.gamma,
            baseline_type=self.config.baseline_type,
            use_reward_to_go=self.config.use_reward_to_go
        )
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        **kwargs
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
            raise RuntimeError("REINFORCE must be setup before calling generate()")
        
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
        """
        Evaluate the model on a validation dataset.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            num_eval_steps: Number of evaluation steps (None for full dataset)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self._is_setup:
            raise RuntimeError("REINFORCE must be setup before calling evaluate()")
        
        self._model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_return = 0.0
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
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Compute losses
                algorithm_output = self.compute_loss(batch, model_outputs)
                
                # Accumulate metrics
                total_loss += algorithm_output.loss.item()
                total_policy_loss += algorithm_output.metrics.get('policy_loss', 0.0)
                total_entropy_loss += algorithm_output.metrics.get('entropy_loss', 0.0)
                total_return += algorithm_output.metrics.get('returns_mean', 0.0)
                num_batches += 1
        
        # Compute averages
        eval_metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_policy_loss': total_policy_loss / num_batches,
            'eval_entropy_loss': total_entropy_loss / num_batches,
            'eval_return_mean': total_return / num_batches,
        }
        
        logger.info(f"Evaluation results: {eval_metrics}")
        return eval_metrics
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save REINFORCE checkpoint with algorithm-specific state."""
        additional_state = {
            'step_count': self._step_count,
            'entropy_coeff': self.loss_computer.current_entropy_coeff,
            'baseline_moving_average': self.baseline.moving_average,
            'baseline_update_count': self.baseline.update_count,
            'returns_running_mean': self.returns_computer.running_mean,
            'returns_running_std': self.returns_computer.running_std,
            'returns_count': self.returns_computer.return_count,
        }
        
        # Save value function state if using value function baseline
        if (self.config.baseline_type == "value_function" and 
            self.baseline.value_function is not None):
            additional_state['value_function_state'] = self.baseline.value_function.state_dict()
            if self.baseline.value_optimizer is not None:
                additional_state['value_optimizer_state'] = self.baseline.value_optimizer.state_dict()
        
        super().save_checkpoint(path, **additional_state, **kwargs)
    
    def load_checkpoint(self, path: str, **kwargs) -> Dict[str, Any]:
        """Load REINFORCE checkpoint and restore algorithm-specific state."""
        checkpoint = super().load_checkpoint(path, **kwargs)
        
        # Restore REINFORCE-specific state
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']
        
        if 'entropy_coeff' in checkpoint:
            self.loss_computer.current_entropy_coeff = checkpoint['entropy_coeff']
        
        if 'baseline_moving_average' in checkpoint:
            self.baseline.moving_average = checkpoint['baseline_moving_average']
        
        if 'baseline_update_count' in checkpoint:
            self.baseline.update_count = checkpoint['baseline_update_count']
        
        if 'returns_running_mean' in checkpoint:
            self.returns_computer.running_mean = checkpoint['returns_running_mean']
        
        if 'returns_running_std' in checkpoint:
            self.returns_computer.running_std = checkpoint['returns_running_std']
        
        if 'returns_count' in checkpoint:
            self.returns_computer.return_count = checkpoint['returns_count']
        
        # Restore value function state if present
        if ('value_function_state' in checkpoint and 
            self.baseline.value_function is not None):
            self.baseline.value_function.load_state_dict(checkpoint['value_function_state'])
        
        if ('value_optimizer_state' in checkpoint and 
            self.baseline.value_optimizer is not None):
            self.baseline.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        
        return checkpoint
    
    def get_info(self) -> Dict[str, Any]:
        """Get REINFORCE algorithm information."""
        info = super().get_info()
        
        # Add REINFORCE-specific information
        reinforce_info = {
            'gamma': self.config.gamma,
            'baseline_type': self.config.baseline_type,
            'use_baseline': self.config.use_baseline,
            'use_reward_to_go': self.config.use_reward_to_go,
            'normalize_returns': self.config.normalize_returns,
            'current_entropy_coeff': self.loss_computer.current_entropy_coeff,
            'step_count': self._step_count,
            'baseline_moving_average': self.baseline.moving_average,
            'has_value_function': self.baseline.value_function is not None,
        }
        
        info.update(reinforce_info)
        return info


# Helper functions for REINFORCE usage


def create_reinforce_config(**kwargs) -> REINFORCEConfig:
    """
    Create a REINFORCE configuration with sensible defaults.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        REINFORCEConfig instance
        
    Example:
        >>> config = create_reinforce_config(
        ...     gamma=0.95,
        ...     use_baseline=True,
        ...     baseline_type="moving_average"
        ... )
    """
    return REINFORCEConfig(**kwargs)


def create_reinforce_algorithm(
    config: Optional[REINFORCEConfig] = None,
    **kwargs
) -> REINFORCE:
    """
    Create a REINFORCE algorithm instance.
    
    Args:
        config: REINFORCE configuration (optional)
        **kwargs: Configuration parameters if config is None
        
    Returns:
        REINFORCE algorithm instance
        
    Example:
        >>> reinforce = create_reinforce_algorithm(
        ...     gamma=0.99,
        ...     use_baseline=True
        ... )
    """
    if config is None:
        config = REINFORCEConfig(**kwargs)
    
    return REINFORCE(config)


def load_reinforce_from_checkpoint(
    checkpoint_path: str,
    model=None,
    optimizer=None
) -> REINFORCE:
    """
    Load REINFORCE algorithm from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Policy model (optional)
        optimizer: Optimizer (optional)
        
    Returns:
        REINFORCE algorithm instance loaded from checkpoint
        
    Example:
        >>> reinforce = load_reinforce_from_checkpoint(
        ...     "checkpoints/reinforce_step_1000.pt",
        ...     model=policy_model
        ... )
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    config = REINFORCEConfig.from_dict(config_dict)
    
    # Create algorithm
    reinforce = REINFORCE(config)
    
    # Setup if models provided
    if model is not None:
        reinforce.setup(model=model, optimizer=optimizer)
    
    # Load state
    reinforce.load_checkpoint(checkpoint_path)
    
    return reinforce


# Export main classes and functions
__all__ = [
    'REINFORCE',
    'REINFORCEConfig',
    'REINFORCEReturns',
    'REINFORCEBaseline',
    'REINFORCELoss',
    'create_reinforce_config',
    'create_reinforce_algorithm',
    'load_reinforce_from_checkpoint'
]