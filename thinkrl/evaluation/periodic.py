"""Held-out evaluation during an RL run.

Training reward is the quantity the policy is directly optimizing, so it rises whether or
not the model improves, which makes reward hacking invisible by construction. A held-out
number that diverges from the training number is exactly what catches a policy that has
learned the verifier instead of the task. See #135.

This is the trainer-side hook; the loop itself is :class:`thinkrl.evaluation.Evaluator`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from thinkrl.evaluation.evaluators import Evaluator
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)

# Keys RLHFDataset emits, most specific first: prompt_text carries the chat-template
# rendering when one was applied, so it is what the policy actually saw.
_PROMPT_KEYS = ("prompt_text", "prompt")
_TARGET_KEYS = ("target", "answer")


def _first_present(sample: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        try:
            value = sample[key]
        except (KeyError, IndexError, TypeError):
            continue
        if value is not None:
            return value
    return None


def extract_prompts_and_targets(dataset: Any) -> tuple[list[str], list[str] | None]:
    """Pull prompt and reference strings out of a dataset of mappings.

    Accepts anything indexable yielding mappings, which covers ``RLHFDataset`` and a plain
    list of dicts. Targets come back as ``None`` unless every sample has one, since a
    partially populated list would silently score some samples against nothing.
    """
    prompts: list[str] = []
    targets: list[str] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = _first_present(sample, _PROMPT_KEYS)
        if prompt is None:
            raise ValueError(f"eval sample {i} has none of {_PROMPT_KEYS}; cannot evaluate it")
        prompts.append(prompt)

        target = _first_present(sample, _TARGET_KEYS)
        if target is not None:
            targets.append(target)

    if targets and len(targets) != len(prompts):
        logger.warning(
            "%d of %d eval samples carry a reference answer, so match metrics are skipped; "
            "reward metrics are unaffected.",
            len(targets),
            len(prompts),
        )
        return prompts, None

    return prompts, (targets or None)


class PeriodicEvaluator:
    """Evaluate a held-out set every N steps during training.

    Args:
        evaluator: The :class:`Evaluator` to run
        prompts: Held-out prompts
        targets: Optional reference answers, enabling match metrics
        every: Evaluate every N steps. 0 disables it entirely, which is the default so
            existing callers are unaffected.
        batch_size: Prompts per generation batch
        max_new_tokens: Generation budget per prompt
    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        prompts: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
        every: int = 0,
        batch_size: int = 8,
        max_new_tokens: int = 128,
    ):
        self.evaluator = evaluator
        self.prompts = list(prompts) if prompts else []
        self.targets = list(targets) if targets else None
        self.every = every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    @property
    def enabled(self) -> bool:
        return bool(self.every and self.evaluator is not None and self.prompts)

    def evaluate(self) -> dict[str, float]:
        """Run one pass, returning metrics namespaced under ``eval/``."""
        if not self.enabled:
            return {}

        result = self.evaluator.evaluate(
            self.prompts,
            targets=self.targets,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
        )
        metrics = {f"eval/{name}": value for name, value in result.metrics.items()}
        if metrics:
            logger.info("eval @ %d samples: %s", result.num_samples, result)
        return metrics

    def maybe_evaluate(self, step: int) -> dict[str, float]:
        """Evaluate when ``step`` lands on the interval, otherwise return nothing."""
        if not self.enabled or step <= 0 or step % self.every:
            return {}
        return self.evaluate()


def build_periodic_evaluator(
    *,
    model: Any,
    tokenizer: Any,
    reward_fn: Any = None,
    dataset: Any = None,
    every: int = 0,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> PeriodicEvaluator:
    """Construct the hook, or a disabled one when evaluation was not asked for.

    Returning a disabled instance rather than None keeps the call sites free of a null
    check, which is what the three RL trainers wanted.
    """
    if not every or dataset is None:
        return PeriodicEvaluator()

    prompts, targets = extract_prompts_and_targets(dataset)
    return PeriodicEvaluator(
        evaluator=Evaluator(model=model, tokenizer=tokenizer, reward_fn=reward_fn),
        prompts=prompts,
        targets=targets,
        every=every,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
