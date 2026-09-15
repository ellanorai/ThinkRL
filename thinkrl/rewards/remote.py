"""Score completions with a reward model running in another process.

:mod:`thinkrl.utils.remote_rm_utils` has implemented remote reward calls, with request
batching, retries and response parsing, since before this package existed, and nothing in
``thinkrl/training``, ``thinkrl/cli`` or ``thinkrl/rewards`` referenced it. A user who
wanted a remote RM had to write the glue themselves, and there was no documented signature
telling them what a trainer expects a reward callable to look like. See #129.

This is that glue: one adapter that satisfies the same contract as
:class:`~thinkrl.rewards.universal.UniversalReward`, so a remote RM slots into the place a
local one occupies rather than becoming a second parallel path.
"""

from __future__ import annotations

from typing import Any

import torch

from thinkrl.utils.logging import get_logger
from thinkrl.utils.remote_rm_utils import RemoteRewardModel


logger = get_logger(__name__)


class RemoteRewardScorer:
    """Adapt :class:`RemoteRewardModel` to the trainers' reward-callable contract.

    The trainers call ``reward_fn(prompts, completions, **kwargs)`` and expect a tensor of
    one score per completion. ``RemoteRewardModel.get_rewards`` takes ``queries`` and
    returns a list of floats, where the *queries* are the texts being scored, so the
    completions map to ``queries`` and the prompts to ``prompts``. Getting that mapping
    backwards would score the questions instead of the answers and still return the right
    number of floats, which is why it is spelled out here rather than left to the reader.

    Args:
        remote_urls: URL(s) of the reward server(s)
        reward_fn_path: Alternative to ``remote_urls``: a Python file holding the function
        reward_fn_name: Name of that function
        micro_batch_size: Batch size for the remote calls
        use_ray: Whether to fan out through Ray

    Example:
        >>> scorer = RemoteRewardScorer(remote_urls="http://localhost:8000")
        >>> trainer = GRPOTrainer(..., reward_fn=scorer)
    """

    def __init__(
        self,
        remote_urls: list[str] | str | None = None,
        reward_fn_path: str | None = None,
        reward_fn_name: str = "compute_reward",
        micro_batch_size: int = 64,
        use_ray: bool = True,
    ):
        if not remote_urls and not reward_fn_path:
            raise ValueError("RemoteRewardScorer needs either remote_urls or reward_fn_path")

        self.client = RemoteRewardModel(
            remote_urls=remote_urls,
            reward_fn_path=reward_fn_path,
            reward_fn_name=reward_fn_name,
            micro_batch_size=micro_batch_size,
            use_ray=use_ray,
        )

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        targets: list[Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score ``completions``, returning one reward per completion."""
        scores = self.client.get_rewards(queries=completions, prompts=prompts, labels=targets)

        # Same guard as RewardPipeline grew in #78: a short or long list used to broadcast
        # silently, giving every completion in a group the same reward, which produces
        # zero variance and no gradient while the run reports normal metrics.
        if len(scores) != len(completions):
            raise ValueError(
                f"remote reward model returned {len(scores)} scores for {len(completions)} "
                "completions; refusing to broadcast or truncate"
            )

        return torch.tensor(scores, dtype=torch.float32)
