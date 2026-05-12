"""
ThinkRL KTO Algorithm
=====================

KTO (Kahneman-Tversky Optimization) is a preference optimization method
based on prospect theory from behavioral economics. Adapted for Online RLHF/RLAIF
where the policy generates completions and a reward function determines if the
completion is desirable or undesirable.

Key features:
- Based on Kahneman-Tversky prospect theory
- Handles unbalanced preference data
- Asymmetric treatment of gains and losses
- Works with binary feedback (good/bad)

Reference:
- KTO: Model Alignment as Prospect Theoretic Optimization
- OpenRLHF implementation

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import KTOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class KTOConfig:
    """Configuration for KTO algorithm."""

    # Learning rate
    learning_rate: float = 1e-6

    # KTO-specific parameters
    beta: float = 0.1  # Temperature parameter

    # Prospect theory parameters
    lambda_d: float = 1.0  # Weight for desirable (positive) examples
    lambda_u: float = 1.0  # Weight for undesirable (negative) examples

    # Entropy bonus
    entropy_coef: float = 0.0

    # Training
    n_epochs: int = 1
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Gradient clipping
    clip_grad_norm: float = 1.0

    # Execution
    use_vllm: bool = False

    def __post_init__(self):
        assert self.beta >= 0, "beta must be non-negative"
        assert self.lambda_d > 0, "lambda_d must be positive"
        assert self.lambda_u > 0, "lambda_u must be positive"


class KTOAlgorithm(BaseRLHFAlgorithm):
    """
    KTO (Kahneman-Tversky Optimization) Algorithm.

    Applies prospect theory to preference optimization, treating
    desirable and undesirable examples asymmetrically.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: KTOConfig | None = None,
        **kwargs,
    ):
        config = config or KTOConfig()

        if reference_model is None:
            raise ValueError("KTO requires a reference model.")

        super().__init__(
            policy_model=policy_model,
            ref_model=reference_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            clip_grad_norm=config.clip_grad_norm,
            use_vllm=config.use_vllm,
            **kwargs,
        )

        self.config: KTOConfig = config

        # Initialize Loss Function
        self.loss_fn = KTOLoss(
            beta=config.beta,
            desirable_weight=config.lambda_d,
            undesirable_weight=config.lambda_u,
        )

        if config.entropy_coef > 0:
            from thinkrl.models.loss import EntropyLoss

            self.entropy_fn = EntropyLoss(coef=config.entropy_coef)
        else:
            self.entropy_fn = None

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute KTO Loss.

        Args:
            batch: Dict containing:
                - input_ids: [B, S]
                - attention_mask: [B, S]
                - labels: [B, S] (with -100 for prompt)
                - binarized_labels: [B] (1 for desirable, 0 for undesirable)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bin_labels = batch["binarized_labels"]

        # 1. Forward pass policy model (pi_theta)
        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        # 2. Compute reference model log probabilities
        with torch.inference_mode():
            self.ref_model.eval()
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = self.get_log_probs(ref_outputs, labels)

        # 3. Token Mask (completion tokens only)
        token_mask = (labels != -100).float()

        # 4. Sequence-level log probabilities
        seq_log_probs = (log_probs * token_mask).sum(dim=1)
        seq_ref_log_probs = (ref_log_probs * token_mask).sum(dim=1)

        # 5. Compute KTO Loss
        total_loss, metrics = self.loss_fn(
            policy_logps=seq_log_probs,
            reference_logps=seq_ref_log_probs,
            labels=bin_labels,
        )

        # 6. Optional Entropy Regularization
        if self.entropy_fn is not None:
            entropy_loss = self.entropy_fn(outputs, action_mask=token_mask)
            total_loss += entropy_loss
            metrics["entropy_loss"] = entropy_loss.detach()

        return {
            "loss": total_loss,
            **metrics,
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform a single training update.
        """
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

    def train_on_rollout(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[dict[str, float]]:
        """
        Execute the KTO inner loop (iterations 1..n_epochs).

        Args:
            batch: Rollout batch with input_ids, attention_mask, labels, binarized_labels

        Returns:
            List of metrics dicts, one per epoch
        """
        epoch_metrics = []
        for epoch in range(self.config.n_epochs):
            metrics = self.training_step(batch)
            metrics["epoch"] = epoch
            epoch_metrics.append(metrics)

        return epoch_metrics


def create_kto(
    policy_model: nn.Module,
    reference_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    config: KTOConfig | None = None,
    **kwargs,
) -> KTOAlgorithm:
    """Factory function to create KTO algorithm."""
    return KTOAlgorithm(
        policy_model=policy_model,
        reference_model=reference_model,
        optimizer=optimizer,
        config=config,
        **kwargs
    )


__all__ = ["KTOConfig", "KTOAlgorithm", "create_kto"]
