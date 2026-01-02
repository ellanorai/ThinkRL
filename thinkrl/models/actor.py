"""
ThinkRL Actor Model
====================

Actor model wrapper for policy learning in RLHF.
Aligned with OpenRLHF patterns.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# Optional imports
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class Actor(nn.Module):
    """
    Actor model wrapper for policy learning.

    Wraps a causal language model for use in RLHF training.
    Supports LoRA fine-tuning, gradient checkpointing, and
    distributed training.

    Example:
        ```python
        actor = Actor(
            pretrained_model="meta-llama/Llama-2-7b-hf",
            lora_rank=8,
            lora_alpha=16,
        )

        # Forward pass
        log_probs, output = actor(
            input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
        )
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
        Initialize the Actor model.

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

        # Load model if string provided
        if isinstance(pretrained_model, str):
            config_kwargs = {}
            if use_flash_attention:
                config_kwargs["attn_implementation"] = "flash_attention_2"

            config = AutoConfig.from_pretrained(
                pretrained_model,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )

            # Disable KV cache for training
            config.use_cache = False

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

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                **load_kwargs,
            )
            logger.info(f"Loaded model: {pretrained_model}")
        else:
            self.model = pretrained_model

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
                task_type=TaskType.CAUSAL_LM,
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
        # Common target modules for different architectures
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP (Llama-style)
            "query_key_value",  # Some models use fused QKV
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_output: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward pass through the actor model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            action_mask: Mask for action tokens [batch, seq_len]
            return_output: Whether to return full model output
            temperature: Temperature for log probability scaling

        Returns:
            Tuple of (log_probs, output) or just log_probs
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get logits and compute log probabilities
        logits = output.logits.float()  # Cast to float32 for stability

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens (shifted by 1)
        # logits[:, :-1] predicts tokens[:, 1:]
        gathered_log_probs = log_probs[:, :-1, :].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # Apply action mask if provided
        if action_mask is not None:
            # Action mask should match the shifted sequence
            if action_mask.shape[1] == input_ids.shape[1]:
                action_mask = action_mask[:, 1:]
            gathered_log_probs = gathered_log_probs * action_mask

        if return_output:
            return gathered_log_probs, output
        return (gathered_log_probs,)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences from the actor model.

        Args:
            input_ids: Input prompt token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample (vs greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 to disable)
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Enable KV cache for generation
        self.model.config.use_cache = True

        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                pad_token_id=self.model.config.eos_token_id,
                **kwargs,
            )
        finally:
            # Disable KV cache after generation
            self.model.config.use_cache = False

        return outputs

    def compute_entropy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Entropy tensor
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = output.logits.float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy

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

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "Actor":
        """Load a saved Actor model."""
        return cls(pretrained_model=model_path, **kwargs)
