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
import logging
from typing import Any

import torch
import torch.nn as nn


# Optional imports
try:
    from transformers import AutoConfig, AutoModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class PRMConfig:
    """Configuration for Process Reward Model."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"

    # Architecture
    hidden_size: int | None = None  # Auto-detect from base model
    num_labels: int = 1  # 1 for scalar reward (logit)
    dropout: float = 0.1

    # Step detection
    step_tag: str = "\n"  # Token/string that delimits steps
    use_last_token: bool = True  # Use last token of each step for prediction

    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Loss
    loss_type: str = "binary_cross_entropy"  # or cross_entropy if num_labels > 1
    class_weights: list[float] | None = None
    focal_gamma: float = 2.0

    # Inference
    aggregation: str = "min"  # min, mean, product, last
    threshold: float = 0.5

    # Precision
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # LoRA
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None


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
    """

    def __init__(
        self,
        base_model: str | nn.Module | None = None,
        config: PRMConfig | None = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = PRMConfig(**kwargs)
        self.config = config

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        model_name = base_model or config.model_name_or_path

        # Load base model if string provided
        if isinstance(model_name, str):
            config_kwargs = {}
            if kwargs.get("use_flash_attention", True):
                config_kwargs["attn_implementation"] = "flash_attention_2"

            hf_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=kwargs.get("trust_remote_code", False),
                **config_kwargs,
            )

            load_kwargs = {
                "config": hf_config,
                "trust_remote_code": kwargs.get("trust_remote_code", False),
            }

            if config.bf16:
                load_kwargs["torch_dtype"] = torch.bfloat16

            if kwargs.get("device_map"):
                load_kwargs["device_map"] = kwargs.get("device_map")

            if config.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, skipping 4-bit loading")

            self.model = AutoModel.from_pretrained(
                model_name,
                **load_kwargs,
            )
            self.hidden_size = hf_config.hidden_size
            logger.info(f"Loaded PRM base model: {model_name}")
        else:
            self.model = base_model
            self.hidden_size = self.model.config.hidden_size

        # Classification/Reward head
        self.score_head = nn.Linear(self.hidden_size, config.num_labels, bias=False)
        nn.init.normal_(self.score_head.weight, std=0.02)

        # Apply LoRA if specified
        if config.lora_rank > 0:
            if not _PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules or self._get_default_target_modules(),
                task_type=TaskType.SEQ_CLS,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA to PRM with rank={config.lora_rank}")

    def _get_default_target_modules(self) -> list[str]:
        """Get default LoRA target modules."""
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
            step_positions: Positions of step boundaries for each sample.
                            If None, predicts for ALL tokens (dense).
            labels: Step-level labels [batch, num_steps] or [batch, seq_len]
            return_dict: Whether to return a dict

        Returns:
            Dict with:
                - step_rewards: Per-step reward predictions
                - loss: Training loss (if labels provided)
                - aggregated_reward: Overall reward [batch]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Project to scores
        logits = self.score_head(hidden_states)  # [batch, seq_len, num_labels]

        # If using scalar reward (num_labels=1), squeeze last dim
        if self.config.num_labels == 1:
            logits = logits.squeeze(-1)  # [batch, seq_len]

        # Gather step rewards if positions provided
        step_rewards = None
        if step_positions is not None:
            # This logic assumes step_positions lists the indices of step endings
            # We need to gather logits at these specific indices
            batch_size = input_ids.size(0)
            max_steps = max(len(pos) for pos in step_positions)

            # Initialize with padding value (e.g. -inf or 0)
            gathered_logits = torch.full(
                (batch_size, max_steps), fill_value=float("-inf"), device=input_ids.device, dtype=logits.dtype
            )

            for b in range(batch_size):
                pos = step_positions[b]
                if len(pos) > 0:
                    indices = torch.tensor(pos, device=input_ids.device)
                    # Indices must be within bounds [0, seq_len-1]
                    indices = torch.clamp(indices, max=logits.size(1) - 1)
                    gathered_logits[b, : len(pos)] = logits[b, indices]

            step_rewards = gathered_logits
        else:
            step_rewards = logits  # Dense rewards for all tokens

        loss = None
        if labels is not None:
            # Logic depends on whether labels are per-step or per-token
            if step_positions is not None:
                # Labels should match step_rewards shape [batch, max_steps]
                # We need to mask out padded steps
                # Check shapes
                if labels.shape != step_rewards.shape:
                    # Try to align if simple mismatch
                    if labels.shape[1] > step_rewards.shape[1]:
                        labels = labels[:, : step_rewards.shape[1]]
                    elif labels.shape[1] < step_rewards.shape[1]:
                        # Pad labels? Or just slice rewards?
                        step_rewards = step_rewards[:, : labels.shape[1]]

                # Create mask for valid steps (where reward is not -inf)
                mask = step_rewards != float("-inf")

                if self.config.loss_type == "binary_cross_entropy":
                    # Assuming labels are 0/1 and predictions are logits
                    loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                    loss = loss_fct(step_rewards, labels.float())
                    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                else:
                    # MSE
                    loss_fct = nn.MSELoss(reduction="none")
                    loss = loss_fct(step_rewards, labels.float())
                    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            else:
                # Dense supervision
                # Labels [batch, seq_len]
                if self.config.loss_type == "binary_cross_entropy":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(step_rewards, labels.float())

        # Aggregate rewards
        if step_positions is not None:
            mask = step_rewards != float("-inf")
            # Filter for aggregation
            valid_rewards = torch.where(mask, step_rewards, torch.tensor(0.0, device=step_rewards.device))
            # Note: aggregations like 'min' need careful handling of padding values

            if self.config.aggregation == "min":
                # Replace padding with inf for min
                min_input = torch.where(mask, step_rewards, torch.tensor(float("inf"), device=step_rewards.device))
                aggregated_reward, _ = min_input.min(dim=1)
                # If all were padding/inf, set to 0
                aggregated_reward = torch.where(
                    aggregated_reward == float("inf"),
                    torch.tensor(0.0, device=aggregated_reward.device),
                    aggregated_reward,
                )
            elif self.config.aggregation == "mean":
                aggregated_reward = valid_rewards.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            elif self.config.aggregation == "last":
                # Get last valid step
                last_indices = mask.sum(dim=1).long() - 1
                last_indices = torch.clamp(last_indices, min=0)
                aggregated_reward = step_rewards.gather(1, last_indices.unsqueeze(1)).squeeze(1)
            else:
                # Default to mean
                aggregated_reward = valid_rewards.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            # Dense aggregation? specific to last token usually
            if attention_mask is not None:
                last_indices = attention_mask.sum(dim=1) - 1
                aggregated_reward = step_rewards.gather(1, last_indices.unsqueeze(1)).squeeze(1)
            else:
                aggregated_reward = step_rewards[:, -1]

        if return_dict:
            return {
                "step_rewards": step_rewards,
                "loss": loss,
                "aggregated_reward": aggregated_reward,
            }
        return step_rewards, loss, aggregated_reward

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
        output = self.forward(input_ids, attention_mask, step_positions, return_dict=True)
        return output["step_rewards"]

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
        # Simple helper re-implementing aggregation logic on tensor
        method = method or self.config.aggregation

        # Assuming -inf is padding
        mask = step_rewards != float("-inf")

        if method == "min":
            min_input = torch.where(mask, step_rewards, torch.tensor(float("inf"), device=step_rewards.device))
            agg, _ = min_input.min(dim=1)
            return torch.where(agg == float("inf"), torch.tensor(0.0), agg)
        elif method == "mean":
            valid = torch.where(mask, step_rewards, torch.tensor(0.0, device=step_rewards.device))
            return valid.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        elif method == "last":
            last_indices = mask.sum(dim=1).long() - 1
            last_indices = torch.clamp(last_indices, min=0)
            return step_rewards.gather(1, last_indices.unsqueeze(1)).squeeze(1)

        return step_rewards.mean(dim=1)

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
        step_tag = self.config.step_tag
        step_token_id = tokenizer.encode(step_tag, add_special_tokens=False)

        # If multiple tokens, take the last one? Or perform string matching?
        # Simple token matching for now
        target_id = step_token_id[-1] if step_token_id else tokenizer.eos_token_id

        batch_positions = []
        for seq in input_ids:
            # Find indices where token == target_id
            indices = (seq == target_id).nonzero(as_tuple=True)[0]
            batch_positions.append(indices.tolist())

        return batch_positions

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
        thresh = threshold or self.config.threshold

        # Identify errors (reward < threshold)
        # Assuming sigmoid applied if binary? Or logits?
        # If logits, thresh usually 0. If prob, 0.5

        # Let's assume input is logits for flexibility, convert to probs if needed
        # But commonly PRM outputs logits.

        is_error = step_rewards < thresh

        # Mask out padding
        mask = step_rewards != float("-inf")
        is_error = is_error & mask

        # Find first true
        # We can use (is_error.int().argmax(dim=1))
        # but if no error, argmax returns 0. Need to check if any error exists.

        has_error = is_error.any(dim=1)
        first_error_idx = is_error.int().argmax(dim=1)

        # Set to -1 where no error
        result = torch.where(has_error, first_error_idx, torch.tensor(-1, device=step_rewards.device))
        return result


class PRMTrainer:
    """
    Trainer for Process Reward Models.

    Handles:
    - Step-level label processing
    - Class imbalance handling
    - Evaluation metrics (step accuracy, path accuracy)

    TODO: Implement full trainer logic in future iteration.
    For now, this is a placeholder.
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
        self.model = model
        self.config = config or model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def train(self) -> dict[str, Any]:
        """Run training."""
        logger.warning("PRMTrainer.train not fully implemented.")
        return {}

    def evaluate(self) -> dict[str, float]:
        """Run evaluation."""
        return {}

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
        return {}


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
        config = PRMConfig(model_name_or_path=model_name_or_path, **kwargs)
    else:
        config.model_name_or_path = model_name_or_path
        # Override config with kwargs if needed
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    return ProcessRewardModel(config=config, **kwargs)


__all__ = [
    "PRMConfig",
    "ProcessRewardModel",
    "PRMTrainer",
    "create_prm",
]
