import pytest
import torch

from thinkrl.rewards import (
    FunctionScorer,
    KeywordScorer,
    LengthScorer,
    RegexScorer,
    RewardPipeline,
)


def test_length_scorer():
    prompts = ["p1", "p2"]
    completions = ["short", "longer_completion"]

    # Test: Longer is better
    scorer = LengthScorer(longer_is_better=True, scale=1.0)
    scores = scorer(prompts, completions)
    assert torch.allclose(scores, torch.tensor([5.0, 17.0]))

    # Test: Target length
    scorer = LengthScorer(target_length=10, scale=1.0)
    scores = scorer(prompts, completions)
    # "short" (5) -> -|5-10| = -5
    # "longer_completion" (17) -> -|17-10| = -7
    assert torch.allclose(scores, torch.tensor([-5.0, -7.0]))
    assert scores.shape == (2,)


def test_keyword_scorer():
    prompts = ["p1", "p2"]
    completions = ["This is good", "This is bad"]

    scorer = KeywordScorer(keywords=["good"], score=1.0)
    scores = scorer(prompts, completions)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]))

    # Case insensitive
    scorer = KeywordScorer(keywords=["GOOD"], score=1.0, case_sensitive=False)
    scores = scorer(prompts, completions)
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]))


def test_regex_scorer():
    prompts = ["p"]
    completions = ["abc 123", "no digits"]

    # Match digit
    scorer = RegexScorer(pattern=r"\d+", score=2.0)
    scores = scorer(prompts, completions)
    assert torch.allclose(scores, torch.tensor([2.0, 0.0]))

    # Count matches
    completions = ["1 2 3", "none"]
    scorer = RegexScorer(pattern=r"\d", score=1.0, match_count=True)
    scores = scorer(prompts, completions)
    assert torch.allclose(scores, torch.tensor([3.0, 0.0]))


def test_function_scorer():
    def my_func(prompts, completions):
        return [1.0 if "a" in c else 0.0 for c in completions]

    scorer = FunctionScorer(func=my_func)
    scores = scorer([], ["apple", "b"])
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]))


def test_pipeline():
    prompts = ["p1"]
    completions = ["good length"]  # len=11, has "good"

    # Pipeline: Length (scale 0.1) + Keyword ("good", score 2.0)
    scorer1 = LengthScorer(scale=0.1)  # 11 * 0.1 = 1.1
    scorer2 = KeywordScorer(keywords=["good"], score=2.0)  # 2.0

    pipeline = RewardPipeline(
        [
            (scorer1, 1.0),  # weight 1.0
            (scorer2, 0.5),  # weight 0.5 -> 2.0 * 0.5 = 1.0
        ]
    )

    scores = pipeline(prompts, completions)
    # Expected: 1.1 + 1.0 = 2.1
    assert torch.allclose(scores, torch.tensor([2.1]))


if __name__ == "__main__":
    test_length_scorer()
    test_keyword_scorer()
    test_regex_scorer()
    test_function_scorer()
    test_pipeline()
    print("All tests passed!")
