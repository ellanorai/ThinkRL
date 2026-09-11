"""Scoring helpers for evaluation.

Deliberately small: these are the aggregations an evaluation loop needs before anything
task-specific, and they are kept separate so a caller can use them without an Evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence


def exact_match(predictions: Sequence[str], targets: Sequence[str], strip: bool = True) -> float:
    """Fraction of predictions equal to their target.

    Args:
        predictions: Model outputs
        targets: Expected outputs
        strip: Compare with surrounding whitespace removed

    Returns:
        Accuracy in [0, 1]; 0.0 for an empty input
    """
    if len(predictions) != len(targets):
        raise ValueError(f"predictions ({len(predictions)}) and targets ({len(targets)}) differ in length")
    if not predictions:
        return 0.0

    def normalise(text: str) -> str:
        return text.strip() if strip else text

    hits = sum(normalise(p) == normalise(t) for p, t in zip(predictions, targets))
    return hits / len(predictions)


def contains_match(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Fraction of predictions containing their target.

    Looser than :func:`exact_match`, and the usual choice for reasoning traces where the
    answer is embedded in a longer completion.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"predictions ({len(predictions)}) and targets ({len(targets)}) differ in length")
    if not predictions:
        return 0.0

    return sum(t.strip() in p for p, t in zip(predictions, targets)) / len(predictions)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty input rather than ZeroDivisionError."""
    return sum(values) / len(values) if values else 0.0


__all__ = ["exact_match", "contains_match", "mean"]
