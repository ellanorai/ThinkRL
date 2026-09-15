"""Held-out evaluation during an RL run.

Training reward is the number being optimized, so it rises whether or not the policy
improves. Without a held-out number there is nothing that distinguishes a better policy
from one that has learned the verifier, and nothing for CheckpointManager's mode="max"
selection to rank by. See #135.
"""

import pytest
import torch

from thinkrl.evaluation.periodic import (
    PeriodicEvaluator,
    build_periodic_evaluator,
    extract_prompts_and_targets,
)


class _StubEvaluator:
    """Stands in for Evaluator so these tests do not generate text."""

    def __init__(self, reward: float = 0.5):
        self.reward = reward
        self.calls = 0

    def evaluate(self, prompts, targets=None, batch_size=8, max_new_tokens=128):
        from thinkrl.evaluation.evaluators import EvalResult

        self.calls += 1
        return EvalResult(num_samples=len(prompts), metrics={"reward_mean": self.reward})


def _hook(every=2, reward=0.5):
    return PeriodicEvaluator(
        evaluator=_StubEvaluator(reward), prompts=["a", "b"], targets=None, every=every
    )


def test_disabled_by_default_so_existing_callers_are_unaffected():
    assert PeriodicEvaluator().enabled is False
    assert PeriodicEvaluator().maybe_evaluate(10) == {}


def test_zero_every_never_evaluates():
    hook = _hook(every=0)
    assert hook.enabled is False
    for step in range(1, 10):
        assert hook.maybe_evaluate(step) == {}


def test_runs_only_on_the_interval():
    hook = _hook(every=3)

    fired = [step for step in range(1, 10) if hook.maybe_evaluate(step)]

    assert fired == [3, 6, 9]
    assert hook.evaluator.calls == 3


def test_metrics_are_namespaced_so_they_never_collide_with_training_reward():
    """train/reward and eval/reward diverging is the whole signal; they must not be
    written to the same key."""
    hook = _hook(every=1)

    metrics = hook.maybe_evaluate(1)

    assert metrics == {"eval/reward_mean": 0.5}
    assert "reward_mean" not in metrics


def test_step_zero_does_not_evaluate_an_untrained_policy():
    assert _hook(every=1).maybe_evaluate(0) == {}


def test_extract_reads_prompt_text_in_preference_to_prompt():
    """prompt_text carries the chat-template rendering, which is what the policy saw."""
    dataset = [{"prompt": "raw", "prompt_text": "<|user|>rendered", "target": "4"}]

    prompts, targets = extract_prompts_and_targets(dataset)

    assert prompts == ["<|user|>rendered"]
    assert targets == ["4"]


def test_extract_raises_when_a_sample_has_no_prompt():
    with pytest.raises(ValueError, match="cannot evaluate"):
        extract_prompts_and_targets([{"answer": "4"}])


def test_partial_targets_are_dropped_rather_than_misaligned(caplog):
    """A half-filled target list would score some samples against nothing."""
    dataset = [{"prompt": "a", "target": "1"}, {"prompt": "b"}]

    prompts, targets = extract_prompts_and_targets(dataset)

    assert prompts == ["a", "b"]
    assert targets is None


def test_builder_returns_a_disabled_hook_when_evaluation_was_not_requested():
    hook = build_periodic_evaluator(model=None, tokenizer=None, dataset=[{"prompt": "a"}], every=0)

    assert hook.enabled is False


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_builder_wires_the_reward_function_through():
    def reward_fn(prompts, completions):
        return torch.ones(len(prompts))

    hook = build_periodic_evaluator(
        model=_TinyLM(),
        tokenizer=object(),
        reward_fn=reward_fn,
        dataset=[{"prompt": "a"}],
        every=5,
    )

    assert hook.enabled is True
    assert hook.evaluator.reward_fn is reward_fn
    assert hook.every == 5
