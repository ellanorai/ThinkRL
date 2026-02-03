"""
ThinkRL Data Processors
========================

Data processing utilities for RLHF training.
Aligned with OpenRLHF patterns for data preprocessing.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from tqdm import tqdm


logger = logging.getLogger(__name__)


def reward_normalization(
    objs: list[dict[str, Any]],
    reward_key: str = "reward",
) -> list[dict[str, Any]]:
    """
    Normalize reward values using z-score standardization.

    Args:
        objs: List of objects containing reward values
        reward_key: Key for reward values in objects

    Returns:
        Objects with normalized rewards

    Example:
        ```python
        data = [
            {"input": "text1", "reward": 0.8},
            {"input": "text2", "reward": 0.2},
            {"input": "text3", "reward": 0.5},
        ]
        normalized = reward_normalization(data)
        # Rewards are now z-score normalized
        ```
    """
    rewards = [obj[reward_key] for obj in objs if reward_key in obj]

    if not rewards:
        return objs

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance**0.5 + 1e-8

    for obj in objs:
        if reward_key in obj:
            obj[reward_key] = (obj[reward_key] - mean) / std

    logger.info(f"Normalized {len(rewards)} rewards: mean={mean:.4f}, std={std:.4f}")
    return objs


def conditional_sft_processor(
    args: Any,
    objs: list[dict[str, Any]],
    input_key: str = "input",
    output_key: str = "output",
    reward_key: str = "reward",
    input_template: str = "{input} <rm_score>: {reward} ",
    normalize_rewards: bool = False,
) -> list[dict[str, Any]]:
    """
    Process data for conditional supervised fine-tuning.

    Formats inputs with reward scores, allowing the model to learn
    reward-conditional generation.

    Args:
        args: Configuration arguments
        objs: List of data objects
        input_key: Key for input text
        output_key: Key for output text
        reward_key: Key for reward values
        input_template: Template for formatting (must contain {input} and {reward})
        normalize_rewards: Whether to normalize rewards first

    Returns:
        Processed data with formatted inputs

    Example:
        ```python
        data = [
            {"input": "Question?", "output": "Answer", "reward": 0.9},
        ]
        processed = conditional_sft_processor(args, data)
        # processed[0]["input"] = "Question? <rm_score>: 0.9 "
        ```
    """
    if "{input}" not in input_template or "{reward}" not in input_template:
        raise ValueError("input_template must contain both {input} and {reward} placeholders")

    if normalize_rewards:
        objs = reward_normalization(objs, reward_key)

    for obj in tqdm(objs, desc="Processing conditional SFT data"):
        if input_key in obj and reward_key in obj:
            reward = obj[reward_key]
            if isinstance(reward, float):
                reward = f"{reward:.4f}"
            obj[input_key] = input_template.format(
                input=obj[input_key],
                reward=reward,
            )

    logger.info(f"Processed {len(objs)} samples for conditional SFT")
    return objs


def rejection_sampling_processor(
    args: Any,
    objs: list[dict[str, Any]],
    input_key: str = "input",
    output_key: str = "output",
    reward_key: str = "reward",
) -> list[dict[str, Any]]:
    """
    Process data using rejection sampling (best-of-n).

    For each unique input, keeps only the output with the highest reward.

    Args:
        args: Configuration arguments
        objs: List of data objects
        input_key: Key for input text
        output_key: Key for output text
        reward_key: Key for reward values

    Returns:
        Filtered data with only best outputs

    Example:
        ```python
        data = [
            {"input": "Q1", "output": "A1", "reward": 0.5},
            {"input": "Q1", "output": "A2", "reward": 0.9},  # Best for Q1
            {"input": "Q2", "output": "A3", "reward": 0.7},
        ]
        filtered = rejection_sampling_processor(args, data)
        # Returns data for A2 and A3
        ```
    """
    # Group by input
    input_to_outputs: dict[str, list[dict[str, Any]]] = {}

    for obj in objs:
        input_text = obj.get(input_key, "")
        if input_text not in input_to_outputs:
            input_to_outputs[input_text] = []
        input_to_outputs[input_text].append(obj)

    # Keep best output for each input
    result = []
    for _input_text, outputs in input_to_outputs.items():
        best = max(outputs, key=lambda x: x.get(reward_key, float("-inf")))
        result.append(best)

    logger.info(
        f"Rejection sampling: {len(objs)} -> {len(result)} samples " f"({len(input_to_outputs)} unique inputs)"
    )
    return result


def iterative_dpo_processor(
    args: Any,
    objs: list[dict[str, Any]],
    input_key: str = "input",
    output_key: str = "output",
    reward_key: str = "reward",
) -> list[dict[str, Any]]:
    """
    Process data for iterative Direct Preference Optimization (DPO).

    Creates preference pairs by selecting highest and lowest reward outputs
    for each input.

    Args:
        args: Configuration arguments
        objs: List of data objects
        input_key: Key for input text
        output_key: Key for output text
        reward_key: Key for reward values

    Returns:
        Preference pairs with chosen and rejected outputs

    Example:
        ```python
        data = [
            {"input": "Q1", "output": "Good", "reward": 0.9},
            {"input": "Q1", "output": "Bad", "reward": 0.1},
        ]
        pairs = iterative_dpo_processor(args, data)
        # pairs[0] = {
        #     "input": "Q1",
        #     "chosen": "Good",
        #     "rejected": "Bad",
        #     "chosen_reward": 0.9,
        #     "rejected_reward": 0.1,
        # }
        ```
    """
    # Group by input
    input_to_outputs: dict[str, list[dict[str, Any]]] = {}

    for obj in objs:
        input_text = obj.get(input_key, "")
        if input_text not in input_to_outputs:
            input_to_outputs[input_text] = []
        input_to_outputs[input_text].append(obj)

    # Create preference pairs
    result = []
    for input_text, outputs in input_to_outputs.items():
        if len(outputs) < 2:
            continue

        # Sort by reward
        sorted_outputs = sorted(
            outputs,
            key=lambda x: x.get(reward_key, 0),
            reverse=True,
        )

        best = sorted_outputs[0]
        worst = sorted_outputs[-1]

        result.append(
            {
                input_key: input_text,
                "chosen": best.get(output_key, ""),
                "rejected": worst.get(output_key, ""),
                "chosen_reward": best.get(reward_key, 0),
                "rejected_reward": worst.get(reward_key, 0),
            }
        )

    logger.info(f"Iterative DPO: {len(objs)} samples -> {len(result)} preference pairs")
    return result


def best_of_n_processor(
    args: Any,
    objs: list[dict[str, Any]],
    n: int = 4,
    input_key: str = "input",
    output_key: str = "output",
    reward_key: str = "reward",
) -> list[dict[str, Any]]:
    """
    Process data using best-of-n sampling.

    For each unique input with at least n outputs, keeps only the best output.

    Args:
        args: Configuration arguments
        objs: List of data objects
        n: Minimum number of samples required per input
        input_key: Key for input text
        output_key: Key for output text
        reward_key: Key for reward values

    Returns:
        Filtered data with best outputs

    Example:
        ```python
        # Requires at least 4 samples per input
        filtered = best_of_n_processor(args, data, n=4)
        ```
    """
    # Group by input
    input_to_outputs: dict[str, list[dict[str, Any]]] = {}

    for obj in objs:
        input_text = obj.get(input_key, "")
        if input_text not in input_to_outputs:
            input_to_outputs[input_text] = []
        input_to_outputs[input_text].append(obj)

    # Keep best output for inputs with >= n samples
    result = []
    skipped = 0
    for _input_text, outputs in input_to_outputs.items():
        if len(outputs) < n:
            skipped += 1
            continue
        best = max(outputs, key=lambda x: x.get(reward_key, float("-inf")))
        result.append(best)

    logger.info(f"Best-of-{n}: {len(objs)} -> {len(result)} samples " f"(skipped {skipped} inputs with < {n} samples)")
    return result


def filter_by_reward_threshold(
    objs: list[dict[str, Any]],
    threshold: float,
    reward_key: str = "reward",
    keep_above: bool = True,
) -> list[dict[str, Any]]:
    """
    Filter data by reward threshold.

    Args:
        objs: List of data objects
        threshold: Reward threshold
        reward_key: Key for reward values
        keep_above: If True, keep samples above threshold; if False, keep below

    Returns:
        Filtered data

    Example:
        ```python
        # Keep only high-reward samples
        high_quality = filter_by_reward_threshold(data, threshold=0.8)
        ```
    """
    if keep_above:
        result = [obj for obj in objs if obj.get(reward_key, 0) >= threshold]
    else:
        result = [obj for obj in objs if obj.get(reward_key, 0) < threshold]

    logger.info(
        f"Filtered by reward {'above' if keep_above else 'below'} {threshold}: "
        f"{len(objs)} -> {len(result)} samples"
    )
    return result


def create_pairwise_data(
    objs: list[dict[str, Any]],
    input_key: str = "input",
    output_key: str = "output",
    reward_key: str = "reward",
    margin: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Create pairwise preference data from ranked outputs.

    Creates all valid pairs where the chosen output has higher reward
    than the rejected output by at least the specified margin.

    Args:
        objs: List of data objects
        input_key: Key for input text
        output_key: Key for output text
        reward_key: Key for reward values
        margin: Minimum reward difference for a valid pair

    Returns:
        List of preference pairs

    Example:
        ```python
        pairs = create_pairwise_data(data, margin=0.1)
        # Creates pairs where reward difference >= 0.1
        ```
    """
    # Group by input
    input_to_outputs: dict[str, list[dict[str, Any]]] = {}

    for obj in objs:
        input_text = obj.get(input_key, "")
        if input_text not in input_to_outputs:
            input_to_outputs[input_text] = []
        input_to_outputs[input_text].append(obj)

    # Create all valid pairs
    result = []
    for input_text, outputs in input_to_outputs.items():
        for i, chosen in enumerate(outputs):
            for j, rejected in enumerate(outputs):
                if i == j:
                    continue

                chosen_reward = chosen.get(reward_key, 0)
                rejected_reward = rejected.get(reward_key, 0)

                if chosen_reward - rejected_reward >= margin:
                    result.append(
                        {
                            input_key: input_text,
                            "chosen": chosen.get(output_key, ""),
                            "rejected": rejected.get(output_key, ""),
                            "chosen_reward": chosen_reward,
                            "rejected_reward": rejected_reward,
                        }
                    )

    logger.info(f"Created {len(result)} preference pairs from {len(objs)} samples")
    return result


# Processor registry
PROCESSORS: dict[str, Callable] = {
    "rs": rejection_sampling_processor,
    "rejection_sampling": rejection_sampling_processor,
    "csft": conditional_sft_processor,
    "conditional_sft": conditional_sft_processor,
    "iter_dpo": iterative_dpo_processor,
    "iterative_dpo": iterative_dpo_processor,
    "best_of_n": best_of_n_processor,
    "bon": best_of_n_processor,
}


def get_processor(name: str) -> Callable:
    """
    Get a processor function by name.

    Args:
        name: Processor name

    Returns:
        Processor function

    Raises:
        ValueError: If processor name is not found

    Example:
        ```python
        processor = get_processor("rs")
        processed_data = processor(args, raw_data)
        ```
    """
    if name not in PROCESSORS:
        available = ", ".join(PROCESSORS.keys())
        raise ValueError(f"Unknown processor: {name}. Available: {available}")

    return PROCESSORS[name]


def register_processor(name: str, processor: Callable) -> None:
    """
    Register a custom processor function.

    Args:
        name: Processor name
        processor: Processor function

    Example:
        ```python
        def my_processor(args, objs):
            # Custom processing logic
            return objs

        register_processor("my_processor", my_processor)
        ```
    """
    PROCESSORS[name] = processor
    logger.info(f"Registered processor: {name}")


# Public API
__all__ = [
    # Core processors
    "reward_normalization",
    "conditional_sft_processor",
    "rejection_sampling_processor",
    "iterative_dpo_processor",
    "best_of_n_processor",
    "filter_by_reward_threshold",
    "create_pairwise_data",
    # Registry
    "PROCESSORS",
    "get_processor",
    "register_processor",
]
