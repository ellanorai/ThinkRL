"""Maths answers must be compared as maths, not as trailing numbers.

_check_math_correctness compared only the last number found in each string,
so 1/3 never matched 0.333, \\boxed{42} never matched 42, and any answer that
was not a bare number fell through to string equality.
"""

import pytest

from thinkrl.rewards.universal import UniversalReward


@pytest.fixture
def reward():
    return UniversalReward(math_tolerance=1e-4)


@pytest.mark.parametrize(
    ("pred", "target"),
    [
        ("4", "4"),
        ("4.0", "4"),
        ("1,000", "1000"),
        (r"\boxed{42}", "42"),
        (r"The answer is \boxed{42}.", "42"),
        (r"\frac{1}{2}", "0.5"),
        (r"\dfrac{3}{4}", "0.75"),
        ("1/3", "0.3333333"),
        ("#### 18", "18"),
        ("$25$", "25"),
        ("50%", "50"),
        ("x^{2}", "x^2"),
        ("(x+1)^2", "x^2+2*x+1"),
        ("x+1", "1+x"),
        ("The answer is 4", "4"),
    ],
)
def test_equivalent_answers_are_accepted(reward, pred, target):
    assert reward._check_math_correctness(pred, target) is True


@pytest.mark.parametrize(
    ("pred", "target"),
    [
        ("5", "6"),
        ("1/3", "0.5"),
        (r"\boxed{41}", "42"),
        ("x+2", "1+x"),
    ],
)
def test_wrong_answers_are_rejected(reward, pred, target):
    assert reward._check_math_correctness(pred, target) is False


def test_sympy_only_sees_expressions_made_of_maths_characters(reward):
    """sympy parsing has eval semantics and completions are model output."""
    payload = "__import__('os').system('touch /tmp/thinkrl_sympy_pwned')"
    assert reward._SAFE_EXPR.match(payload) is None
    assert reward._sympy_equal(payload, "1") is False


def test_unparseable_input_is_not_a_match(reward):
    assert reward._sympy_equal("((((", "1") is False


def test_end_to_end_scores_a_latex_answer(reward):
    """The whole reward, not just the helper."""
    completions = [r"<think>half of one</think><answer>\frac{1}{2}</answer>"]
    rewards = reward(["What is 1/2?"], completions, targets=["0.5"])
    assert rewards[0].item() == pytest.approx(0.2 + 1.0)
