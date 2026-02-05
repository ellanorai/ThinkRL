"""
Reward Pipeline
===============

Tools for combining multiple scorers into a single reward signal.
"""

import torch

from thinkrl.rewards.scorer import BaseScorer


class RewardPipeline:
    """
    Combines multiple scorers with weights.
    """

    def __init__(self, scorers: list[tuple[BaseScorer, float]]):
        """
        Args:
            scorers: List of (Scorer, weight) tuples.
        """
        self.scorers = scorers

    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        """
        Compute weighted sum of rewards.
        """
        if not self.scorers:
            # Return zeros if no scorers
            return torch.zeros(len(completions), dtype=torch.float)

        total_rewards = None

        for scorer, weight in self.scorers:
            scores = scorer(prompts, completions)

            # Ensure tensor
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, dtype=torch.float)

            weighted_scores = scores * weight

            if total_rewards is None:
                total_rewards = weighted_scores
            else:
                total_rewards += weighted_scores

        # Ensure consistent device/type if needed, but for now return CPU tensor
        return total_rewards
