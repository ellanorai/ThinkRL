"""
PAPO Algorithm Implementation
=============================

Perception-Aware Policy Optimization (PAPO) for Multimodal Reasoning.

PAPO builds upon Group Relative Policy Optimization (GRPO) by introducing:
1. Implicit Perception Loss: Maximizes a bounded KL divergence between the policy's
   distribution on original inputs vs. masked/corrupted inputs. This encourages
   the model to attend to visual information (if masking removes it) while preventing
   hallucinations via the cap.
2. Double Entropy Loss: Regularizes the policy to prevent collapse, countering the
   maximization pressure of the perception term.

References:
    PAPO: https://arxiv.org/abs/2507.06448
    GRPO: https://arxiv.org/abs/2402.03300

Author: Archit Sood @ EllanorAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.models.loss import PAPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PAPOConfig(GRPOConfig):
    """
    Configuration for PAPO algorithm.
    Inherits from GRPOConfig.
    """

    # PAPO-specific hyperparameters
    gamma: float = 0.01  # Coefficient for Implicit Perception Loss
    eta: float = 0.03  # Coefficient for Double Entropy Loss
    kl_prcp_cap: float = 5.0  # Cap for perception KL divergence

    # Inherited defaults from GRPO (can be overridden)
    # beta: float = 0.04  # KL penalty coefficient
    # group_size: int = 64
    # clip_epsilon: float = 0.2


class PAPOAlgorithm(GRPOAlgorithm):
    """
    Perception-Aware Policy Optimization (PAPO).

    Extends GRPOAlgorithm to support Implicit Perception Loss and Double Entropy Loss.
    Requires the batch to contain masked versions of inputs ('masked_input_ids').
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: PAPOConfig | None = None,
        **kwargs,
    ):
        # Use default config if none provided
        config = config or PAPOConfig()

        # Initialize base GRPO algorithm
        # This sets up the optimizer, scheduler, etc.
        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config,
            **kwargs,
        )

        self.config = config

        # Initialize PAPO Loss Function
        # Note: PAPOLoss computes KL ref internally, unlike GRPOLoss
        self.loss_fn = PAPOLoss(
            gamma=config.gamma,
            eta=config.eta,
            clip_eps=config.clip_epsilon,
            beta=config.beta,
            kl_prcp_cap=config.kl_prcp_cap,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute PAPO Loss.

        Args:
            batch: Dict containing:
                - input_ids: [B, S]
                - masked_input_ids: [B, S] (Required for PAPO)
                - attention_mask: [B, S]
                - masked_attention_mask: [B, S] (Optional, defaults to attention_mask)
                - labels: [B, S] (with -100 for prompt)
                - rewards: [B]
                - old_log_probs: [B, S]
        """
        cfg = self.config

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"]
        old_log_probs = batch["old_log_probs"]

        # Check for masked inputs required by PAPO
        if "masked_input_ids" not in batch:
            raise ValueError(
                "PAPO algorithm requires 'masked_input_ids' in the batch to compute "
                "Implicit Perception Loss. Ensure your data loader provides this."
            )

        masked_input_ids = batch["masked_input_ids"]
        masked_attention_mask = batch.get("masked_attention_mask", attention_mask)

        # 1. Compute Advantages (Group Relative) - Reused from GRPO
        advantages = self.compute_advantages(rewards)
        # Broadcast advantages to [B, S]
        advantages_expanded = advantages.unsqueeze(1).expand_as(old_log_probs)

        # 2. Forward pass policy model on ORIGINAL inputs
        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        # 3. Forward pass policy model on MASKED inputs (for Perception Loss)
        # We need gradients here for the entropy maximization part of PAPO
        outputs_mask = self.policy_model(input_ids=masked_input_ids, attention_mask=masked_attention_mask)
        log_probs_mask = self.get_log_probs(outputs_mask, labels)

        # 4. Compute reference model log probabilities if needed
        ref_log_probs = None
        if self.ref_model is not None and cfg.beta > 0:
            with torch.inference_mode():
                self.ref_model.eval()
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_log_probs = self.get_log_probs(ref_outputs, labels)

        # 5. Token Mask (completion tokens only)
        token_mask = (labels != -100).float()

        # 6. Compute PAPO Loss
        # Update loss parameters in case they were changed (e.g. annealing)
        self.loss_fn.gamma = cfg.gamma
        self.loss_fn.beta = cfg.beta

        total_loss, metrics_loss = self.loss_fn(
            log_probs=log_probs,
            log_probs_mask=log_probs_mask,
            old_log_probs=old_log_probs,
            advantages=advantages_expanded,
            ref_log_probs=ref_log_probs,
            action_mask=token_mask,
        )

        # 7. Collect Metrics
        with torch.no_grad():
            metrics = {
                "loss": total_loss,
                "advantage_mean": advantages.mean(),
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                # Add PAPO-specific metrics from the loss function
                **metrics_loss,
            }

        return metrics
