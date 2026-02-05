"""
Universal Reward Function
=========================

A single, comprehensive reward function for Reinforce++ that supports:
- Reasoning structure validation (<think>...</think><answer>...</answer>)
- Multi-domain answer correctness (Math, Code, Text)
- Multilingual support (basic normalization)
"""

import re
import math
import torch
from .scorer import BaseScorer


class UniversalReward(BaseScorer):
    """
    Universal Reward Function for Reinforce++.
    
    Rewards:
    1. Structure: <think> and <answer> tags presence.
    2. Correctness: Matches extracted answer against ground truth.
       - Math: Numerical equivalence (e.g., 1.0 == 1).
       - Code: Checks for markdown code blocks.
       - Text: Normalization (case, whitespace).
    """

    def __init__(
        self,
        format_reward: float = 0.1,
        answer_reward: float = 1.0,
        structure_penalty: float = -0.5,
        math_tolerance: float = 1e-6,
        length_penalty: float = 0.0,
    ):
        """
        Args:
            format_reward: Reward for correct XML structure.
            answer_reward: Reward for correct answer.
            structure_penalty: Penalty for broken structure.
            math_tolerance: Tolerance for float comparisons.
            length_penalty: Coefficient for length penalty (score -= coeff * num_words).
        """
        self.format_reward = format_reward
        self.answer_reward = answer_reward
        self.structure_penalty = structure_penalty
        self.math_tolerance = math_tolerance
        self.length_penalty = length_penalty

    def _extract_content(self, text: str, tag: str) -> str | None:
        """Extract content between <tag> and </tag>."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _is_number(self, s: str) -> float | None:
        """Check if string is a number and return it."""
        try:
            return float(s.replace(",", "")) # Handle 1,000
        except ValueError:
            return None

    def _extract_last_number(self, text: str) -> float | None:
        """Extract the last number found in text."""
        # Regex for integers and floats
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            return self._is_number(nums[-1])
        return None

    def _check_math_correctness(self, pred: str, target: str) -> bool:
        """Check mathematical equivalence."""
        pred_val = self._extract_last_number(pred)
        target_val = self._extract_last_number(target)
        
        if pred_val is not None and target_val is not None:
            return math.isclose(pred_val, target_val, abs_tol=self.math_tolerance)
            
        # Fallback to exact string match for non-numeric math answers (e.g. "x + y")
        return self._check_text_correctness(pred, target)

    def _check_code_correctness(self, pred: str, target: str) -> bool:
        """
        Check code answer correctness.
        Uses containment instead of exact match for stability.
        """
        # Normalize: remove whitespace, markdown ticks
        def normalize_code(s):
            # Remove markdown code blocks
            # Handle ```python, ```, etc.
            s = re.sub(r"```\w*\n", "", s) # Remove opening ```python
            s = s.replace("```", "")       # Remove closing ```
            return "".join(s.split())

        norm_pred = normalize_code(pred)
        norm_target = normalize_code(target)
        
        # Containment is safer for RL than exact match
        return norm_target in norm_pred

    def _check_text_correctness(self, pred: str, target: str) -> bool:
        """Check text correctness with normalization and gaming prevention."""
        # Simple normalization
        norm_pred = " ".join(pred.lower().split())
        norm_target = " ".join(target.lower().split())
        
        if norm_pred == norm_target:
            return True
            
        # Inclusion checking with ratio check to prevent gaming
        if len(norm_target) > 5 and norm_target in norm_pred:
            # Prevent pasting target + lot of junk
            # If pred is more than 3x target length, punish it.
            return len(norm_pred) < 3 * len(norm_target)
            
        return False

    def _is_correct(self, pred: str, target: str) -> bool:
        """Dispatch to appropriate correctness check."""
        # Heuristic to detect domain
        
        # 1. Math: Target looks like a number or contains one at the end (GSM8K style)
        if self._is_number(target) is not None or self._extract_last_number(target) is not None:
             return self._check_math_correctness(pred, target)
             
        # 2. Code: Target contains code blocks or common code keywords (simple heuristic)
        if "```" in target or "def " in target or "import " in target:
            return self._check_code_correctness(pred, target)
            
        # 3. Default: Text
        return self._check_text_correctness(pred, target)

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> torch.Tensor:
        targets = kwargs.get("targets", None)
        rewards = []

        for i, comp in enumerate(completions):
            score = 0.0

            # 1. Structure Check (Mutually Exclusive)
            has_think = "<think>" in comp and "</think>" in comp
            has_answer = "<answer>" in comp and "</answer>" in comp

            if has_think and has_answer:
                score += 2 * self.format_reward
            else:
                 score += self.structure_penalty

            # 2. Answer Correctness Check
            pred_answer = self._extract_content(comp, "answer")
            
            # If we have a target
            if targets is not None and i < len(targets):
                target = targets[i]
                if pred_answer:
                    if self._is_correct(pred_answer, target):
                        score += self.answer_reward
                else:
                    # Missing answer block when target exists -> Penalty
                    # Prevent collapse to "skipping answer is safer"
                    score -= self.answer_reward
            
            # 3. Length Regularization
            if self.length_penalty > 0:
                # Approximate word count
                num_words = len(comp.split())
                score -= self.length_penalty * num_words

            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float)
