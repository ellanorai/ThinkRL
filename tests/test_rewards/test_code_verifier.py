"""The code half of #126.

`_check_code_correctness` stripped whitespace from both strings and returned
`normalized_target in normalized_pred`. No execution, no test cases. Under a policy
gradient that is worse than exact match: the highest-reward strategy it teaches is to
reproduce reference-looking text.
"""

import pytest

from thinkrl.rewards.universal import UniversalReward


REFERENCE = "def add(a, b):\n    return a + b"


def test_scoring_code_without_a_verifier_refuses():
    reward = UniversalReward()

    with pytest.raises(NotImplementedError, match="cannot score code"):
        reward._check_code_correctness("def add(a, b):\n    return a + b", REFERENCE)


def test_the_refusal_says_what_to_pass():
    reward = UniversalReward()

    with pytest.raises(NotImplementedError, match="code_verifier"):
        reward._check_code_correctness("anything", REFERENCE)


def test_a_supplied_verifier_is_used():
    reward = UniversalReward(code_verifier=lambda pred, target: pred.strip() == target.strip())

    assert reward._check_code_correctness(REFERENCE, REFERENCE) is True
    assert reward._check_code_correctness("def sub(a, b): return a - b", REFERENCE) is False


def test_the_verifier_result_is_coerced_to_bool():
    """A verifier that returns a truthy non-bool, such as a count of passing tests,
    should not leak that type into the score arithmetic."""
    reward = UniversalReward(code_verifier=lambda pred, target: 3)

    result = reward._check_code_correctness("x", REFERENCE)

    assert result is True


def test_the_old_containment_hole_is_closed():
    """The regression that motivated this: pasting the reference into a comment used to
    score as a correct answer."""
    gaming = f"# I do not know the answer\n# {REFERENCE}\ndef add(a, b):\n    return 0"
    reward = UniversalReward(code_verifier=lambda pred, target: pred.strip() == target.strip())

    assert reward._check_code_correctness(gaming, REFERENCE) is False


def test_maths_scoring_still_works_without_a_code_verifier():
    """The two halves are independent; requiring a code verifier must not break maths,
    which #141 already fixed."""
    reward = UniversalReward()

    assert reward._check_math_correctness("The answer is 42", "42") is True
