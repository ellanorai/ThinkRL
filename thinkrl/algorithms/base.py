"""
ThinkRL Base Algorithm
======================

Abstract base class for all Reinforcement Learning algorithms in ThinkRL.
Defines the common interface for PPO, GRPO, DAPO, VAPO, and other custom algorithms.

Author: Archit Sood @ EllanorAI
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from thinkrl.utils.datasets import to_device
from thinkrl.utils.logging import get_logger
from thinkrl.utils.metrics import MetricsTracker

logger = get_logger(__name__)


class RLAlgorithm(ABC):
    """
    Abstract base class for Reinforcement Learning algorithms.

    This class defines the standard interface that all RL algorithms must implement
    to be compatible with ThinkRL's training loop and infrastructure.

    Attributes:
        model (nn.Module): The policy model (and optionally value model) being trained.
        tokenizer (Any): Tokenizer for processing text.
        optimizer (Optional[Optimizer]): Optimizer for model updates.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        device (torch.device): Compute device.
        metrics (MetricsTracker): Tracker for training metrics.
        config (Dict[str, Any]): Algorithm-specific configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RL algorithm.

        Args:
            model: The main policy model.
            tokenizer: Tokenizer instance.
            optimizer: Optimizer (if handled by algorithm).
            scheduler: LR scheduler (optional).
            device: Device to run on (auto-detected if None).
            config: Algorithm-specific configuration dictionary.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.metrics = MetricsTracker()
        self._step_count = 0
        self._epoch_count = 0

        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training update step.

        This method should handle:
        1. Forward pass (policy generation or evaluation)
        2. Loss computation (PPO loss, GRPO loss, etc.)
        3. Backward pass and optimization (if not handled externally)

        Args:
            batch: A batch of data (containing input_ids, attention_mask, etc.)

        Returns:
            Dictionary of metrics from the update step (e.g., "loss", "reward", "kl_div").
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'update' method. "
            "This method should perform forward pass, loss computation, and optimization."
        )

    @abstractmethod
    def generate(
        self,
        prompts: Union[List[str], torch.Tensor],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses from the policy model.

        Used for the rollout phase in RLHF.

        Args:
            prompts: Input prompts (text strings or token IDs).
            **kwargs: Generation parameters (max_length, temperature, top_p, etc.).

        Returns:
            Dictionary containing generated sequences and associated metadata.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'generate' method. "
            "This method should generate responses from the policy model for rollouts."
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor of shape (batch_size, seq_len).
            values: Value estimates of shape (batch_size, seq_len).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.

        Returns:
            Tuple of (advantages, returns) tensors.
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=rewards.device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values
        return advantages, returns

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.

        Args:
            log_probs: Log probabilities from current policy.
            ref_log_probs: Log probabilities from reference policy.
            mask: Optional mask for valid tokens.

        Returns:
            Mean KL divergence.
        """
        kl = log_probs - ref_log_probs

        if mask is not None:
            kl = kl * mask
            return kl.sum() / mask.sum().clamp(min=1)

        return kl.mean()

    def train(self):
        """Set model to training mode."""
        self.model.train()
        logger.debug(f"{self.__class__.__name__} set to training mode")

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        logger.debug(f"{self.__class__.__name__} set to evaluation mode")

    def to(self, device: Union[str, torch.device]) -> "RLAlgorithm":
        """
        Move algorithm components to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        logger.debug(f"Moved {self.__class__.__name__} to {self.device}")
        return self

    def zero_grad(self):
        """Zero out gradients in optimizer."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.model.zero_grad()

    def step_optimizer(self, max_grad_norm: Optional[float] = None):
        """
        Perform optimizer step with optional gradient clipping.

        Args:
            max_grad_norm: Maximum gradient norm for clipping (None to skip).
        """
        if self.optimizer is None:
            raise RuntimeError(
                "Cannot step optimizer: no optimizer configured. "
                "Pass an optimizer to __init__ or handle optimization externally."
            )

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()
        self._step_count += 1

        if self.scheduler is not None:
            self.scheduler.step()

    def step_scheduler(self):
        """Step the learning rate scheduler if configured."""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_learning_rate(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate from optimizer or scheduler.
        """
        if self.optimizer is None:
            return 0.0

        return self.optimizer.param_groups[0]["lr"]

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary for saving.

        Override this if the algorithm maintains state outside of the model/optimizer
        (e.g., target networks, buffers).

        Returns:
            Dictionary containing algorithm state.
        """
        state = {
            "step_count": self._step_count,
            "epoch_count": self._epoch_count,
            "config": self.config,
            "metrics_history": self.metrics.get_history() if hasattr(self.metrics, "get_history") else {},
        }

        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load algorithm state.

        Args:
            state_dict: Dictionary containing algorithm state.
        """
        self._step_count = state_dict.get("step_count", 0)
        self._epoch_count = state_dict.get("epoch_count", 0)

        loaded_config = state_dict.get("config", {})
        if loaded_config:
            self.config.update(loaded_config)

        if "optimizer_state_dict" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            logger.debug("Loaded optimizer state")

        if "scheduler_state_dict" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            logger.debug("Loaded scheduler state")

        logger.info(
            f"Restored {self.__class__.__name__} state: "
            f"step={self._step_count}, epoch={self._epoch_count}"
        )

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch contents to the correct device.

        Args:
            batch: Input batch dictionary.

        Returns:
            Batch with tensors moved to self.device.
        """
        return to_device(batch, self.device)

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log metrics to the metrics tracker.

        Args:
            metrics: Dictionary of metric name to value.
            prefix: Optional prefix for metric names.
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.metrics.log(full_name, value, step=self._step_count)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"steps={self._step_count}, "
            f"epochs={self._epoch_count})"
        )
