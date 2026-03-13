"""
STaR (Self-Taught Reasoner) Algorithm Implementation
=====================================================

STaR leverages a model's own reasoning capabilities to improve itself iterativey:
1. Bootstrapping: Sample multiple rationales for each problem.
2. Filtering: Select rationales that lead to the correct answer.
3. Rationalization: For failed problems, provide the correct answer as a hint
   to "reason backward" and generate a valid rationale.
4. Fine-tuning: Update the model on the combined set of successful rationales.

Reference:
    "STaR: Bootstrapping Reasoning with Reasoning"
    Zelikman et al., 2022. https://arxiv.org/abs/2203.14465

Author: Antigravity @ DeepMind
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import SFTLoss
from thinkrl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class STaRConfig:
    """Configuration for STaR algorithm."""

    learning_rate: float = 1e-6
    max_iterations: int = 40  # Number of outer loops
    
    # Training schedule from paper
    warmup_steps: int = 100
    base_training_steps: int = 40
    step_scaling_factor: float = 1.2
    
    # Rationalization
    rationalization_hint_format: str = "The correct answer is {answer}. Let's think step by step:"
    
    # Stability
    clip_grad_norm: float = 1.0
    
    # Dataset processing
    max_length: int = 512


class STaRAlgorithm(BaseRLHFAlgorithm):
    """
    STaR Algorithm (Self-Taught Reasoner).
    
    Essentially performs Supervised Fine-Tuning (SFT) over a dynamically 
    collected dataset of successful reasoning chains.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: STaRConfig | None = None,
        **kwargs,
    ):
        config = config or STaRConfig()
        
        # STaR doesn't strictly need a ref_model for KL during SFT 
        # (the paper uses reset-to-base model M instead), but we keep the interface.
        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )
        
        self.config: STaRConfig = config
        self.loss_fn = SFTLoss()

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute SFT Loss over reasoning chains.
        
        Args:
            batch: Dict containing:
                - input_ids: [B, S]
                - attention_mask: [B, S]
                - labels: [B, S] (with -100 for prompt tokens)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get log probs for SFTLoss
        log_probs = torch.log_softmax(logits, dim=-1)
        
        policy_log_probs = self.get_log_probs(outputs, labels)
        
        loss = self.loss_fn(policy_log_probs, labels, attention_mask=attention_mask)

        return {
            "loss": loss,
            "loss_val": loss.detach(),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform a single SFT training update.
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

        # Return scalars
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["grad_norm"] = grad_norm.item()

        return metrics


def create_star(
    policy_model: nn.Module,
    optimizer: Optimizer | None = None,
    **kwargs,
) -> STaRAlgorithm:
    """
    Factory function to create a STaRAlgorithm instance.
    """
    config = STaRConfig(**kwargs)
    return STaRAlgorithm(
        policy_model=policy_model,
        optimizer=optimizer,
        config=config,
    )
