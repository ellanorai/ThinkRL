"""
Direct Preference Optimization (DPO) Algorithm
==============================================

Implementation of Direct Preference Optimization (DPO) as described in
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
(Rafailov et al., 2023).

DPO optimizes the policy directly from preferences without training an explicit
reward model or using PPO-style RL loops.

Author: EllanorAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import DPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DPOConfig:
    """
    Configuration for DPO training.
    """

    learning_rate: float = 1e-6
    beta: float = 0.1  # The KL (temperature) coefficient
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # Options: 'sigmoid', 'hinge', 'ipo'

    # Training stability
    clip_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        assert self.beta > 0, "Beta (KL coefficient) must be positive"
        assert self.loss_type in ["sigmoid", "hinge", "ipo"]


class DPOAlgorithm(BaseRLHFAlgorithm):
    """
    Direct Preference Optimization (DPO) algorithm.

    Optimizes a policy to maximize the likelihood of chosen responses over
    rejected responses relative to a reference model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: Optimizer | None = None,
        config: DPOConfig | None = None,
        **kwargs,
    ):
        """
        Initialize DPO Algorithm.

        Args:
            policy_model: The policy model to optimize (and implicit reward model)
            ref_model: The reference model (frozen) for KL regularization
            optimizer: Optimizer for policy model
            config: DPO configuration dataclass
            **kwargs: Additional arguments for BaseRLHFAlgorithm
        """
        config = config or DPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.beta,  # We map beta to kl_coeff for consistency
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )

        self.config = config

        # Type check to satisfy strict linters since base class defines it as Optional
        if self.ref_model is None:
            raise ValueError("DPOAlgorithm requires a reference model.")

        # Ensure reference model is in eval mode and disabled gradients
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Initialize Loss Function
        self.loss_fn = DPOLoss(
            beta=config.beta,
            label_smoothing=config.label_smoothing,
            ipo=(config.loss_type == "ipo"),
        )

        logger.info(
            f"Initialized DPO (beta={config.beta}, " f"loss_type={config.loss_type}, lr={config.learning_rate})"
        )

    def get_batch_log_probs(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sum of log probabilities for the completion.

        Args:
            outputs: Model outputs or logits
            labels: Labels tensor [B, S] (masked with -100)

        Returns:
            Sum of log probabilities for valid tokens per sequence [B]
        """
        # Reuse base class method to get per-token log probs
        # get_log_probs returns [B, S] with 0.0 at masked positions (where label == -100)
        token_log_probs = self.get_log_probs(outputs, labels)

        # Sum over the sequence dimension to get log(pi(y|x))
        # Since masked tokens are 0.0, sum works correctly
        return token_log_probs.sum(dim=-1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute DPO loss.

        Expected batch keys:
            - chosen_input_ids, chosen_attention_mask, chosen_labels
            - rejected_input_ids, rejected_attention_mask, rejected_labels

        Args:
            batch: Dictionary containing chosen and rejected tensors

        Returns:
            Dictionary with loss and metrics
        """
        # Ensure ref_model is available (for type checker)
        if self.ref_model is None:
            raise ValueError("Reference model is missing for DPO.")

        # Unpack batch
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        # Concatenate batches for efficient forward pass
        # Structure: [Chosen Batch... | Rejected Batch...]
        all_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        # 1. Forward pass policy model
        policy_outputs = self.policy_model(input_ids=all_input_ids, attention_mask=all_mask)
        policy_log_probs = self.get_batch_log_probs(policy_outputs, all_labels)

        # 2. Forward pass reference model (no grad)
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=all_input_ids, attention_mask=all_mask)
            ref_log_probs = self.get_batch_log_probs(ref_outputs, all_labels)

        # 3. Split logits back into chosen and rejected
        batch_size = chosen_input_ids.shape[0]

        policy_chosen_log_probs = policy_log_probs[:batch_size]
        policy_rejected_log_probs = policy_log_probs[batch_size:]

        ref_chosen_log_probs = ref_log_probs[:batch_size]
        ref_rejected_log_probs = ref_log_probs[batch_size:]

        # 4. Compute Loss using DPOLoss
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_log_probs,
            policy_rejected_logps=policy_rejected_log_probs,
            reference_chosen_logps=ref_chosen_log_probs,
            reference_rejected_logps=ref_rejected_log_probs,
        )

        # Add other metrics useful for debugging
        metrics["log_probs/chosen"] = policy_chosen_log_probs.detach().mean()
        metrics["log_probs/rejected"] = policy_rejected_log_probs.detach().mean()

        return {
            "loss": loss,
            **metrics,
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute single DPO training step on a batch of preference data.

        Args:
            batch: Dictionary containing chosen/rejected keys

        Returns:
            Dictionary of metrics
        """
        assert self.optimizer is not None, "Optimizer not initialized"

        self.policy_model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]

        loss.backward()

        grad_norm = 0.0
        if self.config.clip_grad_norm > 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.clip_grad_norm
            )
            grad_norm = float(grad_norm_tensor.item())

        self.optimizer.step()

        metrics = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
        metrics["grad_norm"] = grad_norm

        return metrics


def create_dpo(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-6,
    beta: float = 0.1,
    **kwargs,
) -> DPOAlgorithm:
    """
    Factory function to create DPOAlgorithm instance.

    Args:
        policy_model: The policy model
        ref_model: The reference model
        optimizer: Optimizer (optional)
        learning_rate: Learning rate
        beta: KL coefficient
        **kwargs: Additional args for DPOConfig/DPOAlgorithm

    Returns:
        Configured DPOAlgorithm instance
    """
    config = DPOConfig(learning_rate=learning_rate, beta=beta, **kwargs)

    if optimizer is None:
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    return DPOAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
    )


__all__ = ["DPOAlgorithm", "DPOConfig", "create_dpo"]
