"""
ThinkRL Algorithms
==================

RLHF algorithms for training language models with human feedback.

Available algorithms:
- PPO: Proximal Policy Optimization (with value function)
- GRPO: Group Relative Policy Optimization (critic-free)
- DPO: Direct Preference Optimization (preference-based)
- DAPO: Decoupled clip and dynamic sampling Policy Optimization
- VAPO: Value-model-based Augmented PPO (for reasoning)
- REINFORCE: Classic Monte Carlo policy gradient

Author: EllanorAI
"""

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig, create_dapo
from thinkrl.algorithms.dpo import DPOAlgorithm, DPOConfig, create_dpo
from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig, create_ppo
from thinkrl.algorithms.reinforce import REINFORCEAlgorithm, REINFORCEConfig, create_reinforce
from thinkrl.algorithms.vapo import VAPOAlgorithm, VAPOConfig


# Algorithm registry for factory pattern
ALGORITHMS = {
    "ppo": PPOAlgorithm,
    "grpo": GRPOAlgorithm,
    "dpo": DPOAlgorithm,
    "dapo": DAPOAlgorithm,
    "vapo": VAPOAlgorithm,
    "reinforce": REINFORCEAlgorithm,
}

CONFIGS = {
    "ppo": PPOConfig,
    "grpo": GRPOConfig,
    "dpo": DPOConfig,
    "dapo": DAPOConfig,
    "vapo": VAPOConfig,
    "reinforce": REINFORCEConfig,
}


def get_algorithm(name: str) -> type[BaseRLHFAlgorithm]:
    """Get algorithm class by name."""
    name = name.lower()
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name]


def get_config(name: str):
    """Get config class by algorithm name."""
    name = name.lower()
    if name not in CONFIGS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]


__all__ = [
    # Base
    "BaseRLHFAlgorithm",
    # PPO
    "PPOAlgorithm",
    "PPOConfig",
    "create_ppo",
    # GRPO
    "GRPOAlgorithm",
    "GRPOConfig",
    # DPO
    "DPOAlgorithm",
    "DPOConfig",
    "create_dpo",
    # DAPO
    "DAPOAlgorithm",
    "DAPOConfig",
    "create_dapo",
    # VAPO
    "VAPOAlgorithm",
    "VAPOConfig",
    # REINFORCE
    "REINFORCEAlgorithm",
    "REINFORCEConfig",
    "create_reinforce",
    # Registry
    "ALGORITHMS",
    "CONFIGS",
    "get_algorithm",
    "get_config",
]
