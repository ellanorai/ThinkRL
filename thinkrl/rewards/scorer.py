"""
Reward Scorers
==============

Atomic scoring modules for constructing reward functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import re
from re import Pattern

import torch


class BaseScorer(ABC):
    """
    Abstract base class for reward scorers.
    """

    @abstractmethod
    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        """
        Compute rewards for a batch of completions.

        Args:
            prompts: List of prompt texts.
            completions: List of completion texts.

        Returns:
            torch.Tensor: Tensor of shape [batch_size] containing rewards.
        """
        pass


class LengthScorer(BaseScorer):
    """
    Rewards completions based on their length.
    """

    def __init__(self, target_length: int | None = None, longer_is_better: bool = True, scale: float = 0.01):
        """
        Args:
            target_length: If set, reward is negative distance to target.
            longer_is_better: If True (and target_length is None), reward is length * scale.
            scale: Scaling factor for the score.
        """
        self.target_length = target_length
        self.longer_is_better = longer_is_better
        self.scale = scale

    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        rewards = []
        for comp in completions:
            length = len(comp)
            if self.target_length is not None:
                # Negative distance to target
                score = -abs(length - self.target_length) * self.scale
            elif self.longer_is_better:
                score = length * self.scale
            else:
                score = -length * self.scale
            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float)


class KeywordScorer(BaseScorer):
    """
    Rewards presence of specific keywords.
    """

    def __init__(self, keywords: list[str], score: float = 1.0, case_sensitive: bool = False):
        """
        Args:
            keywords: List of keywords to look for.
            score: Score to add each time a keyword is found (or once per keyword?).
                   Current impl: +score if ANY keyword is present.
            case_sensitive: Whether matching is case sensitive.
        """
        self.keywords = keywords
        self.score = score
        self.case_sensitive = case_sensitive

    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        rewards = []
        for comp in completions:
            text = comp if self.case_sensitive else comp.lower()
            found = False
            for kw in self.keywords:
                kw_text = kw if self.case_sensitive else kw.lower()
                if kw_text in text:
                    found = True
                    break

            rewards.append(self.score if found else 0.0)

        return torch.tensor(rewards, dtype=torch.float)


class RegexScorer(BaseScorer):
    """
    Rewards matches of a regex pattern.
    """

    def __init__(self, pattern: str | Pattern, score: float = 1.0, match_count: bool = False):
        """
        Args:
            pattern: Regex pattern string or compiled pattern.
            score: Score to award.
            match_count: If True, score = count(matches) * score. If False, score = score if matched else 0.
        """
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.score = score
        self.match_count = match_count

    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        rewards = []
        for comp in completions:
            matches = self.pattern.findall(comp)
            if self.match_count:
                rewards.append(len(matches) * self.score)
            else:
                rewards.append(self.score if matches else 0.0)

        return torch.tensor(rewards, dtype=torch.float)


class FunctionScorer(BaseScorer):
    """
    Wrapper for a user-defined function.
    """

    def __init__(self, func: Callable[[list[str], list[str]], list[float] | torch.Tensor]):
        """
        Args:
            func: Function taking (prompts, completions) and returning list of floats or tensor.
        """
        self.func = func

    def __call__(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        output = self.func(prompts, completions)
        if isinstance(output, list):
            return torch.tensor(output, dtype=torch.float)
        return output.float()
