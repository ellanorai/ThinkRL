"""
ThinkRL PAPO Trainer
======================

Dedicated trainer for PAPO (Perception-Aware Policy Optimization).

Author: EllanorAI
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.papo import PAPOAlgorithm, PAPOConfig
from thinkrl.training.grpo_trainer import GRPOTrainer


class PAPOTrainer(GRPOTrainer):
    """
    Trainer for PAPO.

    PAPOAlgorithm.compute_loss hard-requires `masked_input_ids` in the rollout batch
    for its Implicit Perception Loss (#194) -- a second forward pass on a corrupted
    input, KL-penalized against the original, that teaches the policy to rely on the
    input rather than hallucinate past it. The paper corrupts the *visual* input;
    ThinkRL's dataset/rollout pipeline is text-only (#194's finding still holds, no
    multimodal pipeline exists), so this trainer corrupts the text analogue instead --
    randomly masking a fraction of the *prompt* tokens (never the generated completion)
    in `make_experience`'s output before `train_on_rollout` sees it. This is a real,
    running Implicit Perception Loss, not the paper's vision-occlusion term; `mask_prob`
    is exposed (`--mask-prob` in cli/papo.py) so it's a visible, tunable knob rather than
    a silently-approximated one. A true multimodal PAPO still needs the vision pipeline
    #194 originally found missing.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        config: PAPOConfig | None = None,
        optimizer: Optimizer | None = None,
        mask_prob: float = 0.15,
        **kwargs,
    ):
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
        self.mask_prob = mask_prob

        algorithm = PAPOAlgorithm(
            policy_model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config or PAPOConfig(),
        )
        super().__init__(algorithm=algorithm, **kwargs)

    def make_experience(self, batch_prompts: dict[str, Any]) -> dict[str, torch.Tensor]:
        rollout_data = super().make_experience(batch_prompts)
        rollout_data["masked_input_ids"] = self._mask_prompt_tokens(
            rollout_data["input_ids"], rollout_data["labels"]
        )
        return rollout_data

    def _mask_prompt_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Replace a random `mask_prob` fraction of prompt tokens (labels == -100)
        with the tokenizer's mask token, leaving the generated completion untouched."""
        assert self.tokenizer is not None, "PAPOTrainer requires a tokenizer to build masked_input_ids"
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            mask_token_id = self.tokenizer.pad_token_id

        masked_input_ids = input_ids.clone()
        prompt_region = labels == -100
        drop = prompt_region & (torch.rand(input_ids.shape, device=input_ids.device) < self.mask_prob)
        masked_input_ids[drop] = mask_token_id
        return masked_input_ids
