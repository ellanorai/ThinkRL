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

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
        """
        Compute weighted sum of rewards.

        Extra keyword arguments are accepted and forwarded to any scorer that
        declares them, so this satisfies the trainer contract, which passes
        ``targets=`` on every step.
        """
        if not self.scorers:
            raise ValueError(
                "RewardPipeline was constructed with no scorers. Returning zeros would "
                "give every completion an identical reward, which yields no gradient."
            )

        total_rewards = None

        for scorer, weight in self.scorers:
            try:
                scores = scorer(prompts, completions, **kwargs)
            except TypeError:
                # Scorer does not accept the extra keywords; call it plainly.
                scores = scorer(prompts, completions)

            # Ensure tensor
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, dtype=torch.float)

            if scores.shape[0] != len(completions):
                raise ValueError(
                    f"{type(scorer).__name__} returned {scores.shape[0]} scores for "
                    f"{len(completions)} completions. Broadcasting a mismatched count "
                    f"would silently give completions the wrong reward."
                )

            weighted_scores = scores * weight

            if total_rewards is None:
                total_rewards = weighted_scores
            else:
                total_rewards += weighted_scores

        # Ensure consistent device/type if needed, but for now return CPU tensor
        return total_rewards
