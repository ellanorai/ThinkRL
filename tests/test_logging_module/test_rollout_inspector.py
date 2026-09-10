"""Periodic rollout sampling: a scalar reward cannot tell a bad policy from a broken
reward function or empty completions, so the run should be able to show the text."""

import inspect
import io

import pytest
import torch

import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer
from thinkrl.logging import RolloutInspector
from thinkrl.logging.rollout import truncate


def _inspector(**kwargs):
    kwargs.setdefault("every", 2)
    kwargs.setdefault("use_rich", False)
    kwargs.setdefault("file", io.StringIO())
    return RolloutInspector(**kwargs)


def test_disabled_by_default():
    assert RolloutInspector().enabled is False
    assert RolloutInspector().maybe_show(10, ["p"], ["c"], [1.0]) is False


def test_fires_only_on_multiples_and_never_on_step_zero():
    inspector = _inspector(every=3)

    assert [inspector.should_show(s) for s in range(7)] == [False, False, False, True, False, False, True]


def test_shows_prompt_completion_and_reward():
    inspector = _inspector()
    inspector.show(4, ["What is 2+2?"], ["4"], [1.25])

    output = inspector.file.getvalue()
    assert "step 4" in output
    assert "What is 2+2?" in output
    assert "+1.2500" in output


def test_tensor_rewards_are_accepted():
    """Trainers hand rewards over as a tensor."""
    inspector = _inspector()
    inspector.show(2, ["p"], ["c"], torch.tensor([0.5]))

    assert "+0.5000" in inspector.file.getvalue()


def test_empty_completion_is_called_out_rather_than_printed_blank():
    inspector = _inspector()
    inspector.show(2, ["p"], [""], [0.0])

    assert "(empty)" in inspector.file.getvalue()


def test_long_text_is_truncated():
    assert truncate("a" * 100, 10) == "a" * 9 + "…"
    assert truncate("keep\nnewlines  collapsed", 100) == "keep newlines collapsed"


def test_sample_count_is_respected():
    inspector = _inspector(num_samples=2)
    inspector.show(2, ["p1", "p2", "p3"], ["c1", "c2", "c3"], [1.0, 2.0, 3.0])

    output = inspector.file.getvalue()
    assert "c1" in output and "c2" in output and "c3" not in output


def test_mismatched_lengths_do_not_raise_mid_run():
    inspector = _inspector()
    inspector.show(2, ["p1", "p2"], ["c1"], [1.0])

    assert "c1" in inspector.file.getvalue()


def test_missing_rewards_are_shown_as_unknown():
    inspector = _inspector()
    inspector.show(2, ["p"], ["c"], None)

    assert "-" in inspector.file.getvalue()


def test_num_samples_must_be_positive():
    with pytest.raises(ValueError, match="num_samples"):
        RolloutInspector(every=1, num_samples=0)


@pytest.mark.parametrize("module", [grpo_trainer, reinforce_pp_trainer])
def test_both_rl_trainers_expose_and_call_the_inspector(module):
    trainer = next(
        obj for name, obj in vars(module).items() if name.endswith("Trainer") and hasattr(obj, "train")
    )
    params = inspect.signature(trainer.train).parameters

    assert "inspect_every" in params, f"{module.__name__} has no inspect_every option"
    assert "maybe_show" in inspect.getsource(module), f"{module.__name__} never calls the inspector"
