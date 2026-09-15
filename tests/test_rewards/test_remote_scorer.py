"""#129: 410 lines of remote reward-model client that no trainer or CLI could reach.

These pin the adapter's contract rather than the HTTP behaviour, which
tests/test_utils/test_remote_rm_utils.py already covers.
"""

import pytest
import torch

from thinkrl.rewards import RemoteRewardScorer


class _StubClient:
    def __init__(self, scores):
        self.scores = scores
        self.seen = {}

    def get_rewards(self, queries, prompts=None, labels=None):
        self.seen = {"queries": queries, "prompts": prompts, "labels": labels}
        return self.scores


def _scorer(scores):
    scorer = RemoteRewardScorer(remote_urls="http://localhost:8000")
    scorer.client = _StubClient(scores)
    return scorer


def test_construction_requires_a_source_of_rewards():
    with pytest.raises(ValueError, match="remote_urls or reward_fn_path"):
        RemoteRewardScorer()


def test_it_returns_a_tensor_of_one_score_per_completion():
    """The trainers call .to(device) on the result, so a list would not survive."""
    scorer = _scorer([1.0, 2.0])

    rewards = scorer(["p1", "p2"], ["c1", "c2"])

    assert isinstance(rewards, torch.Tensor)
    assert rewards.dtype == torch.float32
    assert rewards.tolist() == [1.0, 2.0]


def test_completions_are_scored_not_prompts():
    """get_rewards takes the texts to score as `queries`. Passing prompts there would
    score the questions, return the right number of floats, and look fine."""
    scorer = _scorer([0.5, 0.5])

    scorer(["what is 2+2?", "what is 3+3?"], ["4", "6"])

    assert scorer.client.seen["queries"] == ["4", "6"]
    assert scorer.client.seen["prompts"] == ["what is 2+2?", "what is 3+3?"]


def test_targets_are_forwarded_as_labels():
    scorer = _scorer([1.0])

    scorer(["p"], ["c"], targets=["4"])

    assert scorer.client.seen["labels"] == ["4"]


@pytest.mark.parametrize("scores", [[1.0], [1.0, 2.0, 3.0]])
def test_a_wrong_score_count_raises_rather_than_broadcasting(scores):
    """The #78 failure in a new place: a short list broadcasts to give every completion
    the same reward, which is zero variance and no gradient while metrics look normal."""
    scorer = _scorer(scores)

    with pytest.raises(ValueError, match="refusing to broadcast"):
        scorer(["p1", "p2"], ["c1", "c2"])


def test_it_satisfies_the_same_shape_as_a_local_reward():
    """A trainer should not care which one it was handed."""
    import inspect

    from thinkrl.rewards import UniversalReward

    remote = inspect.signature(RemoteRewardScorer.__call__).parameters
    local = inspect.signature(UniversalReward.__call__).parameters

    assert list(remote)[:3] == list(local)[:3]
