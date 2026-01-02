"""
ThinkRL Critic Model
=====================

Critic/Value model for advantage estimation in RLHF.
Aligned with OpenRLHF patterns.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# Optional imports
try:
    from transformers import AutoConfig, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class Critic(nn.Module):
    """
    Critic/Value model for advantage estimation.

    Wraps a transformer model with a value head for predicting
    state values in RLHF training.

    Example:
        ```python
        critic = Critic(
            pretrained_model="meta-llama/Llama-2-7b-hf",
            lora_rank=8,
        )

        # Get values for states
        values = critic(input_ids, attention_mask, action_mask)
        ```
    """

    def __init__(
        self,
        pretrained_model: str | nn.Module,
        use_flash_attention: bool = True,
        bf16: bool = True,
        load_in_4bit: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        device_map: str | dict | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize the Critic model.

        Args:
            pretrained_model: Model name or pre-loaded model
            use_flash_attention: Use Flash Attention 2 if available
            bf16: Use bfloat16 precision
            load_in_4bit: Load model in 4-bit quantization
            lora_rank: LoRA rank (0 to disable)
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            target_modules: LoRA target modules
            device_map: Device placement strategy
            trust_remote_code: Trust remote code for custom models
        """
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        # Load base model if string provided
        if isinstance(pretrained_model, str):
            config_kwargs = {}
            if use_flash_attention:
                config_kwargs["attn_implementation"] = "flash_attention_2"

            config = AutoConfig.from_pretrained(
                pretrained_model,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )

            load_kwargs = {
                "config": config,
                "trust_remote_code": trust_remote_code,
            }

            if bf16:
                load_kwargs["torch_dtype"] = torch.bfloat16

            if device_map is not None:
                load_kwargs["device_map"] = device_map

            if load_in_4bit:
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
                pretrained_model,
                **load_kwargs,
            )
            self.hidden_size = config.hidden_size
            logger.info(f"Loaded model: {pretrained_model}")
        else:
            self.model = pretrained_model
            self.hidden_size = self.model.config.hidden_size

        # Value head: projects hidden states to scalar values
        self.value_head = nn.Linear(self.hidden_size, 1, bias=False)

        # Initialize value head
        nn.init.normal_(self.value_head.weight, std=0.02)

        # Apply LoRA if specified
        if lora_rank > 0:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required for LoRA. Install with: pip install peft"
                )

            if target_modules is None:
                target_modules = self._get_default_target_modules()

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.SEQ_CLS,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with rank={lora_rank}")

            # Cast non-LoRA layers to bf16 if needed
            if bf16:
                for name, param in self.model.named_parameters():
                    if "lora_" not in name:
                        param.data = param.data.to(torch.bfloat16)

        self.lora_rank = lora_rank

    def _get_default_target_modules(self) -> list[str]:
        """Get default LoRA target modules based on model architecture."""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "query_key_value",
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Forward pass through the critic model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            action_mask: Mask for action tokens [batch, seq_len]
            return_output: Whether to return full model output

        Returns:
            Values tensor [batch, seq_len-1] or (values, output)
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get last hidden states
        hidden_states = output.last_hidden_state

        # Compute values through value head
        # We exclude the last token (like in OpenRLHF)
        values = self.value_head(hidden_states[:, :-1, :]).squeeze(-1).float()

        # Apply action mask if provided
        if action_mask is not None:
            # Adjust mask to match value dimensions
            if action_mask.shape[1] == input_ids.shape[1]:
                action_mask = action_mask[:, :-1]
            values = values * action_mask

        if return_output:
            return values, output
        return values

    def get_value_at_position(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Get value at a specific position.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position: Position to get value (-1 for last valid position)

        Returns:
            Value tensor [batch]
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = output.last_hidden_state

        if position == -1:
            # Get last valid position for each sequence
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
                last_hidden = hidden_states[batch_indices, seq_lengths]
            else:
                last_hidden = hidden_states[:, -1]
        else:
            last_hidden = hidden_states[:, position]

        values = self.value_head(last_hidden).squeeze(-1).float()
        return values

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        """Enable gradient checkpointing for memory efficiency."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        """Print trainable parameter count."""
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logger.info(
                f"Trainable: {trainable:,} / {total:,} = {100 * trainable / total:.2f}%"
            )

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model."""
        self.model.save_pretrained(save_directory, **kwargs)
        # Save value head separately
        torch.save(
            self.value_head.state_dict(),
            f"{save_directory}/value_head.pt"
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Critic":
        """Load a saved Critic model."""
        critic = cls(pretrained_model=model_path, **kwargs)
        # Try to load value head if it exists
        import os
        value_head_path = os.path.join(model_path, "value_head.pt")
        if os.path.exists(value_head_path):
            critic.value_head.load_state_dict(torch.load(value_head_path))
        return critic
