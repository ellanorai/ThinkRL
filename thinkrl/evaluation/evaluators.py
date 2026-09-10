"""Minimal evaluation loop for a trained policy.

Generates completions for a set of prompts and scores them, either with a reward function
of the same shape the trainers use, or against reference answers, or both. It exists so a
policy can be measured without writing the loop by hand; benchmark harnesses belong in
:mod:`thinkrl.evaluation.benchmarks`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from thinkrl.evaluation.metrics import contains_match, exact_match, mean
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Outcome of one evaluation pass."""

    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    completions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        scores = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.metrics.items()))
        return f"EvalResult(n={self.num_samples}, {scores})"


class Evaluator:
    """Run a policy over prompts and score the completions.

    Args:
        model: Policy model exposing ``generate``
        tokenizer: Tokenizer for the model
        reward_fn: Optional callable taking (prompts, completions) and returning a tensor or
            sequence of per-sample rewards, matching the trainers' reward contract
        device: Device to run on; defaults to the model's own device
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[[list[str], list[str]], Any] | None = None,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        batch_size: int = 8,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> list[str]:
        """Generate one completion per prompt, returning only the generated text."""
        was_training = self.model.training
        self.model.eval()

        completions: list[str] = []
        try:
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start : start + batch_size])
                encoded = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens, **generate_kwargs)

                # Strip the prompt so scoring sees the completion alone.
                prompt_len = encoded["input_ids"].shape[1]
                completions.extend(
                    self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
                )
        finally:
            if was_training:
                self.model.train()

        return completions

    def evaluate(
        self,
        prompts: Sequence[str],
        targets: Sequence[str] | None = None,
        batch_size: int = 8,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> EvalResult:
        """Generate completions for ``prompts`` and score them.

        Args:
            prompts: Prompts to evaluate
            targets: Optional reference answers; enables exact and contains match
            batch_size: Prompts per generation batch
            max_new_tokens: Generation budget per prompt
            **generate_kwargs: Forwarded to ``model.generate``

        Returns:
            An :class:`EvalResult`. Metrics are whatever could be computed: ``reward_mean``
            and ``reward_std`` when a reward function was supplied, ``exact_match`` and
            ``contains_match`` when targets were.
        """
        if targets is not None and len(targets) != len(prompts):
            raise ValueError(f"prompts ({len(prompts)}) and targets ({len(targets)}) differ in length")
        if not prompts:
            return EvalResult(num_samples=0)

        completions = self.generate(
            prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, **generate_kwargs
        )

        metrics: dict[str, float] = {}

        if self.reward_fn is not None:
            rewards = self.reward_fn(list(prompts), completions)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.detach().float().cpu().tolist()
            rewards = [float(r) for r in rewards]
            if len(rewards) != len(prompts):
                raise ValueError(
                    f"reward_fn returned {len(rewards)} rewards for {len(prompts)} prompts"
                )
            metrics["reward_mean"] = mean(rewards)
            metrics["reward_std"] = float(torch.tensor(rewards).std(correction=0)) if len(rewards) > 1 else 0.0

        if targets is not None:
            metrics["exact_match"] = exact_match(completions, targets)
            metrics["contains_match"] = contains_match(completions, targets)

        if not metrics:
            logger.warning(
                "Evaluator ran with neither a reward_fn nor targets, so only completions were produced."
            )

        result = EvalResult(num_samples=len(prompts), metrics=metrics, completions=completions)
        logger.info(str(result))
        return result


__all__ = ["EvalResult", "Evaluator"]
