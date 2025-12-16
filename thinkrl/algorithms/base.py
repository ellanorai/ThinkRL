"""
ThinkRL Base Algorithm
======================

Abstract base class for RLHF algorithms providing:
- Common functionality and interfaces
- vLLM integration for high-throughput generation
- Distributed training support (DDP/FSDP)
- Reward processing and advantage computation
- KL divergence computation
- Metrics tracking and logging

All RLHF algorithms (PPO, VAPO, DAPO, GRPO, etc.) inherit from this base class.

Author: Archit Sood @ EllanorAI
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.utils.logging import get_logger
from thinkrl.utils.metrics import (
    MetricsTracker,
    compute_advantages,
    compute_kl_divergence,
    compute_reward,
)


# Optional vLLM integration
if TYPE_CHECKING:
    from thinkrl.integration.vllm_client import VLLMClient

    _VLLM_AVAILABLE = True
else:
    try:
        from thinkrl.integration.vllm_client import VLLMClient

        _VLLM_AVAILABLE = True
    except ImportError:
        VLLMClient = None
        _VLLM_AVAILABLE = False

logger = get_logger(__name__)


class BaseRLHFAlgorithm(ABC):
    """
    Abstract base class for RLHF algorithms.

    This class provides the common infrastructure needed by all RLHF algorithms:
    - Policy and reference model management
    - Optimizer configuration
    - KL divergence computation from reference policy
    - Reward normalization and processing
    - Advantage estimation (GAE)
    - Integration with vLLM for efficient generation
    - Distributed training utilities
    - Metrics tracking

    Subclasses must implement:
        - compute_loss(): Algorithm-specific loss computation
        - training_step(): Complete training step with gradient updates
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        learning_rate: float = 1e-5,
        kl_coeff: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        normalize_rewards: bool = True,
        normalize_advantages: bool = True,
        clip_grad_norm: float = 1.0,
        use_vllm: bool = False,
        vllm_url: str = "http://localhost:8000",
        vllm_bridge_port: int = 51216,
        **kwargs,
    ):
        """
        Initialize the base RLHF algorithm.

        Args:
            policy_model: The policy model to train
            ref_model: Reference model for KL divergence (optional)
            optimizer: Optimizer for policy model
            learning_rate: Learning rate for optimizer
            kl_coeff: Coefficient for KL divergence penalty
            gamma: Discount factor for returns
            lambda_: GAE lambda parameter
            normalize_rewards: Whether to normalize rewards
            normalize_advantages: Whether to normalize advantages
            clip_grad_norm: Maximum gradient norm for clipping
            use_vllm: Whether to use vLLM for generation
            vllm_url: URL of vLLM server
            vllm_bridge_port: Port for NCCL weight synchronization bridge
            **kwargs: Additional algorithm-specific hyperparameters
        """
        self.policy_model = policy_model
        self.ref_model = ref_model

        # Hyperparameters
        self.learning_rate = learning_rate
        self.kl_coeff = kl_coeff
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.clip_grad_norm = clip_grad_norm

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        # Distributed training
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = self.rank == 0

        # vLLM integration
        self.use_vllm = use_vllm
        self.vllm_client: Optional["VLLMClient"] = None
        if use_vllm:
            if not _VLLM_AVAILABLE:
                raise ImportError("vLLM is required for generation. Install with: pip install vllm")
            # Ensure VLLMClient is not None before instantiation
            if self.is_main_process and VLLMClient is not None:
                self.vllm_client = VLLMClient(url=vllm_url, group_port=vllm_bridge_port)
                logger.info(f"Initialized vLLM client: {vllm_url}")

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Store additional kwargs
        self.config = kwargs

        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(lr={learning_rate}, kl_coeff={kl_coeff}, "
            f"distributed={self.is_distributed}, vllm={use_vllm})"
        )

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute the algorithm-specific loss.
        """
        pass

    @abstractmethod
    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform a single training step.
        """
        pass

    def compute_kl_penalty(self, batch: dict[str, torch.Tensor], reduction: str = "mean") -> torch.Tensor:
        """
        Compute KL divergence penalty from reference model.
        """
        if "ref_log_probs" not in batch:
            # Compute reference log probs if not provided
            if self.ref_model is None:
                logger.warning("No reference model available for KL penalty, returning 0")
                return torch.tensor(0.0, device=batch["input_ids"].device)

            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                ref_log_probs = self.get_log_probs(ref_outputs, batch["labels"])
        else:
            ref_log_probs = batch["ref_log_probs"]

        # Get policy log probs
        policy_outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        policy_log_probs = self.get_log_probs(policy_outputs, batch["labels"])

        # Compute KL divergence
        kl_div = compute_kl_divergence(policy_log_probs, ref_log_probs, reduction)

        return kl_div

    def process_rewards(self, rewards: torch.Tensor, normalize: bool | None = None) -> torch.Tensor:
        """
        Process and optionally normalize rewards.
        """
        should_normalize = normalize if normalize is not None else self.normalize_rewards
        return compute_reward(rewards, normalize=should_normalize)

    def compute_gae_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        should_normalize = normalize if normalize is not None else self.normalize_advantages
        return compute_advantages(
            rewards=rewards,
            values=values,
            gamma=self.gamma,
            lambda_=self.lambda_,
            normalize=should_normalize,
        )

    def get_log_probs(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract log probabilities of target tokens from model outputs.
        """
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        log_probs = torch.log_softmax(logits, dim=-1)

        target_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return target_log_probs

    def generate_rollouts(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **generation_kwargs,
    ) -> list[str]:
        """
        Generate text completions using vLLM (if enabled) or policy model.
        """
        # Assign to distinct local variable for strict type narrowing
        client = self.vllm_client

        if self.use_vllm and client is not None:
            if self.is_main_process:
                logger.debug(f"Generating {len(prompts)} rollouts via vLLM")

                generation_params = {
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "n": num_return_sequences,
                    **generation_kwargs,
                }

                completions = client.generate(prompts, generation_params)
                return completions
            else:
                return []
        else:
            return self._generate_with_policy_model(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                **generation_kwargs,
            )

    def _generate_with_policy_model(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **generation_kwargs,
    ) -> list[str]:
        """
        Generate using the policy model directly (fallback when vLLM not used).
        """
        raise NotImplementedError(
            "Direct policy model generation requires tokenizer. "
            "Either enable vLLM or implement _generate_with_policy_model in subclass."
        )

    def sync_vllm_weights(self):
        """
        Synchronize policy model weights to vLLM server.
        """
        # Assign to distinct local variable for strict type narrowing
        client = self.vllm_client
        if self.use_vllm and client is not None and self.is_main_process:
            logger.debug("Syncing weights to vLLM server")
            client.update_model_weights(self.policy_model)

    def init_vllm_weight_sync(self, device: torch.device):
        """
        Initialize the NCCL bridge for vLLM weight synchronization.
        """
        # Assign to distinct local variable for strict type narrowing
        client = self.vllm_client
        if self.use_vllm and client is not None and self.is_main_process:
            logger.info("Initializing vLLM weight sync bridge")
            client.init_weight_sync(device)

    def get_metrics(self) -> dict[str, float]:
        """
        Get current training metrics.
        Ensures the return type is always a dictionary, even if tracker returns int/float.
        """
        metrics = self.metrics_tracker.get_average()

        if isinstance(metrics, dict):
            return metrics

        # If metrics is an int (likely 0) or float, return an empty dict
        # or a wrapped dict if strictly needed.
        # Assuming int means "empty/no metrics"
        return {}

    def reset_metrics(self):
        """Reset metrics tracker."""
        self.metrics_tracker.reset()

    def state_dict(self) -> dict[str, Any]:
        """
        Get algorithm state dictionary for checkpointing.
        """
        state = {
            "policy_model": self.policy_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "learning_rate": self.learning_rate,
                "kl_coeff": self.kl_coeff,
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "normalize_rewards": self.normalize_rewards,
                "normalize_advantages": self.normalize_advantages,
                "clip_grad_norm": self.clip_grad_norm,
                **self.config,
            },
        }

        if self.ref_model is not None:
            state["ref_model"] = self.ref_model.state_dict()

        return state

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        """
        Load algorithm state from checkpoint.
        """
        self.policy_model.load_state_dict(state["policy_model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])

        if "ref_model" in state and self.ref_model is not None:
            self.ref_model.load_state_dict(state["ref_model"], strict=strict)

        # Load config
        if "config" in state:
            config = state["config"]
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.kl_coeff = config.get("kl_coeff", self.kl_coeff)
            self.gamma = config.get("gamma", self.gamma)
            self.lambda_ = config.get("lambda_", self.lambda_)
            self.normalize_rewards = config.get("normalize_rewards", self.normalize_rewards)
            self.normalize_advantages = config.get("normalize_advantages", self.normalize_advantages)
            self.clip_grad_norm = config.get("clip_grad_norm", self.clip_grad_norm)

        logger.info("Loaded algorithm state from checkpoint")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lr={self.learning_rate}, "
            f"kl_coeff={self.kl_coeff}, "
            f"gamma={self.gamma}, "
            f"lambda={self.lambda_}, "
            f"distributed={self.is_distributed}, "
            f"vllm={self.use_vllm})"
        )


# Public API
__all__ = ["BaseRLHFAlgorithm"]
