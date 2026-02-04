"""
ThinkRL IPO Algorithm
=====================

IPO (Identity Preference Optimization) is a variant of DPO that uses
identity mapping instead of the log-sigmoid loss.

Key features:
- Simpler loss function than DPO
- Identity preference mapping
- More stable gradients
- Better calibrated probabilities (with length normalization)

Reference:
- A General Theoretical Paradigm to Understand Learning from Human Preferences
- OpenRLHF implementation

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import IPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class IPOConfig:
    """
    Configuration for IPO algorithm.

    Note on Length Normalization:
    IPO is sensitive to response length differences because it operates on
    summed log-probabilities. It is strongly recommended to enable
    `length_normalization` unless your data has fixed-length responses.
    """

    # Learning rate
    learning_rate: float = 1e-6

    # IPO-specific parameters
    beta: float = 0.1  # Temperature parameter (used for reward scaling in metrics)
    tau: float = 0.05  # IPO regularization strength

    # Loss configuration
    loss_type: Literal["ipo", "ipo_hinge"] = "ipo"

    # Stability and Normalization
    length_normalization: bool = False

    # Training
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0


class IPOAlgorithm(BaseRLHFAlgorithm):
    """
    IPO (Identity Preference Optimization) Algorithm.

    Uses identity mapping for preference optimization, resulting in
    simpler gradients and better calibration than DPO.

    Loss (Standard): (log_ratio_chosen - log_ratio_rejected - 1/(2*tau))^2
    Loss (Hinge):    RELU(1/(2*tau) - (log_ratio_chosen - log_ratio_rejected))^2

    Assumes:
    - Identical prompt length for chosen/rejected (standard RLHF datasets).
    - If length_normalization is False: similar response lengths are preferred or length bias is acceptable.
    - Reference model is frozen and in eval mode.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: Optimizer | None = None,
        config: IPOConfig | None = None,
        **kwargs,
    ):
        """
        Initialize IPO Algorithm.

        Args:
            policy_model: The policy model to optimize
            ref_model: The reference model (required)
            optimizer: Optimizer for policy model
            config: IPO configuration
            **kwargs: Additional arguments for BaseRLHFAlgorithm
        """
        config = config or IPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=config.beta,  # Use beta as proxy for KL coeff in logging
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )

        self.config: IPOConfig = config
        self.accum_steps = 0  # Counter for gradient accumulation

        if self.ref_model is None:
            raise ValueError("IPOAlgorithm requires a reference model.")

        # Ensure reference model is in eval mode and disabled gradients
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Initialize Loss Function
        self.loss_fn: IPOLoss = IPOLoss(
            tau=config.tau,
            loss_type=config.loss_type,
        )

        logger.info(
            f"Initialized IPO (tau={config.tau}, beta={config.beta}, "
            f"loss_type={config.loss_type}, norm={config.length_normalization})"
        )

    def get_batch_log_probs(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for the completion.

        Handles:
        1. Causal shift (via BaseRLHFAlgorithm.get_log_probs)
        2. Masking of padding tokens (-100)
        3. Optional length normalization
        """
        # Base class get_log_probs returns [B, S] with 0.0 at masked positions.
        # It correctly handles the autoregressive shift.
        token_log_probs = self.get_log_probs(outputs, labels)

        sum_log_probs = token_log_probs.sum(dim=-1)

        if self.config.length_normalization:
            # Calculate length of completions (excluding masked parts)
            # labels structure: [prompt... response... pad...]
            # get_log_probs aligns with input, so we check labels for validity.
            # We must account for the shift: get_log_probs is based on predictions for labels[:, 1:]
            shift_labels = labels[:, 1:]

            # Count non-masked tokens per sequence
            lengths = (shift_labels != -100).float().sum(dim=-1)

            # Avoid division by zero
            return sum_log_probs / lengths.clamp(min=1)

        return sum_log_probs

    def compute_log_ratios(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability ratios for chosen and rejected.

        Returns:
            chosen_log_ratios: log(pi(yw|x)/ref(yw|x))
            rejected_log_ratios: log(pi(yl|x)/ref(yl|x))
        """
        # Unpack batch
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        # Concatenate batches for efficient forward pass
        all_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        # 1. Forward pass policy model
        policy_outputs = self.policy_model(input_ids=all_input_ids, attention_mask=all_mask)
        policy_log_probs = self.get_batch_log_probs(policy_outputs, all_labels)

        # 2. Forward pass reference model (no grad)
        with torch.no_grad():
            assert self.ref_model is not None, "Reference model required for IPO"
            ref_outputs = self.ref_model(input_ids=all_input_ids, attention_mask=all_mask)
            ref_log_probs = self.get_batch_log_probs(ref_outputs, all_labels)

        # 3. Calculate log ratios: log(pi) - log(ref)
        log_ratios = policy_log_probs - ref_log_probs

        # 4. Split back into chosen and rejected
        batch_size = chosen_input_ids.shape[0]
        chosen_log_ratios = log_ratios[:batch_size]
        rejected_log_ratios = log_ratios[batch_size:]

        return chosen_log_ratios, rejected_log_ratios

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Compute IPO loss.
        """
        # Unpack batch
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        # Concatenate batches for efficient forward pass
        all_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        # 1. Forward pass policy model
        policy_outputs = self.policy_model(input_ids=all_input_ids, attention_mask=all_mask)
        policy_log_probs = self.get_batch_log_probs(policy_outputs, all_labels)

        # 2. Forward pass reference model (no grad)
        with torch.no_grad():
            assert self.ref_model is not None, "Reference model required for IPO"
            ref_outputs = self.ref_model(input_ids=all_input_ids, attention_mask=all_mask)
            ref_log_probs = self.get_batch_log_probs(ref_outputs, all_labels)

        # 3. Split back into chosen and rejected
        batch_size = chosen_input_ids.shape[0]

        policy_chosen_log_probs = policy_log_probs[:batch_size]
        policy_rejected_log_probs = policy_log_probs[batch_size:]

        ref_chosen_log_probs = ref_log_probs[:batch_size]
        ref_rejected_log_probs = ref_log_probs[batch_size:]

        # 4. Compute Loss using IPOLoss
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_log_probs,
            policy_rejected_logps=policy_rejected_log_probs,
            ref_chosen_logps=ref_chosen_log_probs,
            ref_rejected_logps=ref_rejected_log_probs,
        )

        metrics["log_probs/chosen"] = policy_chosen_log_probs.detach().mean()
        metrics["log_probs/rejected"] = policy_rejected_log_probs.detach().mean()

        # Add explicit reward metrics scaled by beta for consistency
        with torch.no_grad():
            chosen_logratios = policy_chosen_log_probs - ref_chosen_log_probs
            rejected_logratios = policy_rejected_log_probs - ref_rejected_log_probs

            chosen_rewards = self.config.beta * chosen_logratios
            rejected_rewards = self.config.beta * rejected_logratios

            metrics["rewards/chosen"] = chosen_rewards.mean()
            metrics["rewards/rejected"] = rejected_rewards.mean()
            metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss,
            **metrics,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a single training step with gradient accumulation support.
        """
        assert self.optimizer is not None, "Optimizer not initialized"

        self.policy_model.train()

        # Normalize loss for gradient accumulation
        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
        loss.backward()

        self.accum_steps += 1
        grad_norm = 0.0

        # Optimization Step
        if self.accum_steps % self.config.gradient_accumulation_steps == 0:
            if self.config.clip_grad_norm > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.config.clip_grad_norm
                )
                grad_norm = float(grad_norm_tensor.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Reset accumulation counter if needed, or just let it grow
            # (Resetting effectively handles the modulo check)
            self.accum_steps = 0

        metrics = {k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
        # Only report grad norm on stepping steps, otherwise 0.0
        metrics["grad_norm"] = grad_norm

        return metrics


def create_ipo(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-6,
    beta: float = 0.1,
    tau: float = 0.05,
    **kwargs,
) -> IPOAlgorithm:
    """
    Factory function to create IPO algorithm.

    Args:
        policy_model: Policy model
        ref_model: Reference model
        optimizer: Optimizer (optional)
        learning_rate: Learning rate
        beta: Reward scaling parameter
        tau: IPO regularization parameter
        **kwargs: Additional args for IPOConfig or IPOAlgorithm
    """
    # Extract config-specific args from kwargs if present
    config_args = {k: v for k, v in kwargs.items() if hasattr(IPOConfig, k)}

    config = IPOConfig(learning_rate=learning_rate, beta=beta, tau=tau, **config_args)

    # Pass remaining kwargs to Algorithm init
    algo_kwargs = {k: v for k, v in kwargs.items() if k not in config_args}

    if optimizer is None:
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    return IPOAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
        **algo_kwargs,
    )


__all__ = ["IPOConfig", "IPOAlgorithm", "create_ipo"]
