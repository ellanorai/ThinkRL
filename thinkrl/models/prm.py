"""
ThinkRL Process Reward Model (PRM)
==================================

Process Reward Model for step-by-step reward modeling in reasoning tasks.

PRMs provide fine-grained feedback at each reasoning step, unlike outcome
reward models (ORMs) that only score final answers.

Key features:
- Step-level reward prediction
- Process supervision
- Compatible with PRIME and other process-based RL

References:
- Let's Verify Step by Step (OpenAI)
- Process Reward Models for reasoning

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PRMConfig:
    """Configuration for Process Reward Model."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B"

    # Architecture
    hidden_size: int | None = None  # Auto-detect from base model
    num_labels: int = 2  # Binary: correct/incorrect step
    dropout: float = 0.1

    # Step detection
    step_tag: str = "\n"  # Token/string that delimits steps
    use_last_token: bool = True  # Use last token of each step for prediction

    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Loss
    loss_type: str = "cross_entropy"  # cross_entropy, focal, weighted
    class_weights: list[float] | None = None
    focal_gamma: float = 2.0

    # Inference
    aggregation: str = "min"  # min, mean, product, last
    threshold: float = 0.5

    # Precision
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class ProcessRewardModel(nn.Module):
    """
    Process Reward Model for step-level reward prediction.

    Predicts the correctness of each reasoning step, enabling:
    - Process supervision during training
    - Step-level reward signals for RL
    - Early stopping on incorrect reasoning paths

    Architecture:
    - Base LLM encoder
    - Classification head per step
    - Optional pooling over steps

    TODO: Implement model
    """

    def __init__(
        self,
        base_model: Any = None,
        config: PRMConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        raise NotImplementedError("Process Reward Model not yet implemented")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        step_positions: list[list[int]] | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | tuple:
        """
        Forward pass for PRM.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            step_positions: Positions of step boundaries for each sample
            labels: Step-level labels [batch, num_steps]
            return_dict: Whether to return a dict

        Returns:
            Dict with:
                - step_rewards: Per-step reward predictions [batch, num_steps]
                - loss: Training loss (if labels provided)
                - aggregated_reward: Overall reward [batch]
        """
        raise NotImplementedError

    def predict_step_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        step_positions: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """
        Predict rewards for each reasoning step.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            step_positions: Positions of step boundaries

        Returns:
            Step rewards [batch, num_steps]
        """
        raise NotImplementedError

    def aggregate_rewards(
        self,
        step_rewards: torch.Tensor,
        method: str | None = None,
    ) -> torch.Tensor:
        """
        Aggregate step rewards into a single score.

        Args:
            step_rewards: Per-step rewards [batch, num_steps]
            method: Aggregation method (min, mean, product, last)

        Returns:
            Aggregated reward [batch]
        """
        raise NotImplementedError

    def detect_steps(
        self,
        input_ids: torch.Tensor,
        tokenizer: Any,
    ) -> list[list[int]]:
        """
        Automatically detect step boundaries in the input.

        Args:
            input_ids: Input token IDs
            tokenizer: Tokenizer for decoding

        Returns:
            List of step boundary positions for each sample
        """
        raise NotImplementedError

    def find_first_error(
        self,
        step_rewards: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Find the first incorrect step in the reasoning chain.

        Args:
            step_rewards: Per-step rewards
            threshold: Threshold for correct/incorrect

        Returns:
            Index of first error for each sample (-1 if all correct)
        """
        raise NotImplementedError


class PRMTrainer:
    """
    Trainer for Process Reward Models.

    Handles:
    - Step-level label processing
    - Class imbalance handling
    - Evaluation metrics (step accuracy, path accuracy)

    TODO: Implement trainer
    """

    def __init__(
        self,
        model: ProcessRewardModel,
        config: PRMConfig | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        raise NotImplementedError("PRM Trainer not yet implemented")

    def train(self) -> dict[str, Any]:
        """Run training."""
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Run evaluation."""
        raise NotImplementedError

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute PRM-specific metrics.

        Returns:
            - step_accuracy: Per-step accuracy
            - path_accuracy: Full path correct rate
            - first_error_position: Avg position of first error
        """
        raise NotImplementedError


def create_prm(
    model_name_or_path: str,
    config: PRMConfig | None = None,
    **kwargs,
) -> ProcessRewardModel:
    """
    Factory function to create a Process Reward Model.

    Args:
        model_name_or_path: Base model path
        config: PRM configuration
        **kwargs: Additional arguments

    Returns:
        ProcessRewardModel instance
    """
    if config is None:
        config = PRMConfig(model_name_or_path=model_name_or_path)
    else:
        config.model_name_or_path = model_name_or_path

    return ProcessRewardModel(config=config, **kwargs)


__all__ = [
    "PRMConfig",
    "ProcessRewardModel",
    "PRMTrainer",
    "create_prm",
]
