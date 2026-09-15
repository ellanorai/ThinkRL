"""Evaluation utilities: run a trained policy over prompts and score the result."""

from thinkrl.evaluation.evaluators import EvalResult, Evaluator
from thinkrl.evaluation.metrics import contains_match, exact_match, mean
from thinkrl.evaluation.periodic import PeriodicEvaluator, extract_prompts_and_targets


__all__ = [
    "EvalResult",
    "Evaluator",
    "PeriodicEvaluator",
    "contains_match",
    "exact_match",
    "extract_prompts_and_targets",
    "mean",
]
