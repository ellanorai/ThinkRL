"""
ThinkRL Reward Model
=====================

Reward model for preference learning in RLHF.
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
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class RewardModel(nn.Module):
    """
    Reward model for preference learning.

    Wraps a transformer model with a reward head for scoring
    responses in RLHF training.

    Example:
        ```python
        rm = RewardModel(
            pretrained_model="meta-llama/Llama-2-7b-hf",
            lora_rank=8,
        )

        # Score responses
        rewards = rm(input_ids, attention_mask)
        ```
    """

    def __init__(
        self,
        pretrained_model: str | nn.Module,
        use_flash_attention: bool = True,
        bf16: bool = True,
        fp16: bool = False,
        load_in_4bit: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_init_type: str = "default",
        target_modules: list[str] | None = None,
        device_map: str | dict | None = None,
        trust_remote_code: bool = False,
        normalize_reward: bool = False,
        **kwargs,
    ):
        """
        Initialize the Reward model.

        Args:
            pretrained_model: Model name or pre-loaded model
            use_flash_attention: Use Flash Attention 2 if available
            bf16: Use bfloat16 precision
            fp16: Use float16 precision
            load_in_4bit: Load model in 4-bit quantization
            lora_rank: LoRA rank (0 to disable)
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout rate
            target_modules: LoRA target modules
            device_map: Device placement strategy
            trust_remote_code: Trust remote code for custom models
            normalize_reward: Apply reward normalization
        """
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        self.normalize_reward = normalize_reward

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
            elif fp16:
                load_kwargs["torch_dtype"] = torch.float16

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

        # Reward head: projects hidden states to scalar reward
        self.reward_head = nn.Linear(self.hidden_size, 1, bias=False)

        # Initialize reward head
        nn.init.normal_(self.reward_head.weight, std=0.02)

        # Normalization buffers (running mean and std)
        if normalize_reward:
            self.register_buffer("reward_mean", torch.zeros(1))
            self.register_buffer("reward_std", torch.ones(1))

        # Apply LoRA if specified
        if lora_rank > 0:
            if not _PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")

            if target_modules is None:
                target_modules = self._get_default_target_modules()

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.SEQ_CLS,
                bias="none",
                init_lora_weights=True if lora_init_type == "default" else lora_init_type,
            )

            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with rank={lora_rank}")

            # Cast non-LoRA layers to target precision if needed
            if bf16:
                for name, param in self.model.named_parameters():
                    if "lora_" not in name:
                        param.data = param.data.to(torch.bfloat16)
            elif fp16:
                for name, param in self.model.named_parameters():
                    if "lora_" not in name:
                        param.data = param.data.to(torch.float16)

        self.lora_rank = lora_rank

    def _get_default_target_modules(self) -> list[str]:
        """Get default LoRA target modules based on model architecture."""
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "query_key_value",
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        Forward pass through the reward model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_output: Whether to return full model output

        Returns:
            Rewards tensor [batch] or (rewards, output)
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get last hidden states
        hidden_states = output.last_hidden_state

        # Get representation at last valid position (EOS token)
        if attention_mask is not None:
            # Find last non-padded position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1]

        # Compute rewards through reward head
        rewards = self.reward_head(last_hidden).squeeze(-1).float()

        # Apply normalization during inference
        if self.normalize_reward and not self.training:
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)

        if return_output:
            return rewards, output
        return rewards

    def compute_pairwise_rewards(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards for preference pairs.

        Args:
            chosen_ids: Chosen response token IDs
            chosen_mask: Chosen response attention mask
            rejected_ids: Rejected response token IDs
            rejected_mask: Rejected response attention mask

        Returns:
            Tuple of (chosen_rewards, rejected_rewards)
        """
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)
        return chosen_rewards, rejected_rewards

    def update_normalization(
        self,
        rewards: torch.Tensor,
        momentum: float = 0.1,
    ):
        """
        Update reward normalization statistics.

        Args:
            rewards: Batch of rewards to update statistics with
            momentum: Exponential moving average momentum
        """
        if not self.normalize_reward:
            return

        with torch.no_grad():
            batch_mean = rewards.mean()
            batch_std = rewards.std()

            self.reward_mean = (1 - momentum) * self.reward_mean + momentum * batch_mean
            self.reward_std = (1 - momentum) * self.reward_std + momentum * batch_std

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
            logger.info(f"Trainable: {trainable:,} / {total:,} = {100 * trainable / total:.2f}%")

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model."""
        self.model.save_pretrained(save_directory, **kwargs)
        # Save reward head and normalization stats
        torch.save(
            {
                "reward_head": self.reward_head.state_dict(),
                "reward_mean": self.reward_mean if self.normalize_reward else None,
                "reward_std": self.reward_std if self.normalize_reward else None,
            },
            f"{save_directory}/reward_head.pt",
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> RewardModel:
        """Load a saved RewardModel."""
        rm = cls(pretrained_model=model_path, **kwargs)
        # Try to load reward head if it exists
        import os

        reward_head_path = os.path.join(model_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            state = torch.load(reward_head_path)
            rm.reward_head.load_state_dict(state["reward_head"])
            if state.get("reward_mean") is not None:
                rm.reward_mean = state["reward_mean"]
                rm.reward_std = state["reward_std"]
        return rm
