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
- REINFORCE++: REINFORCE with PPO tricks (no critic)
- RLOO: REINFORCE Leave-One-Out
- PRIME: Process Reinforcement through Implicit Rewards
- DrGRPO: Doctor GRPO (global normalization)
- IPO: Identity Preference Optimization
- KTO: Kahneman-Tversky Optimization
- OnlineDPO: Iterative/Online DPO
- STaR: Self-Taught Reasoner
- ORPO: Odds Ratio Preference Optimization

Author: EllanorAI
"""

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig, create_dapo
from thinkrl.algorithms.dpo import DPOAlgorithm, DPOConfig, create_dpo
from thinkrl.algorithms.dr_grpo import DrGRPOAlgorithm, DrGRPOConfig, create_dr_grpo
from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.algorithms.ipo import IPOAlgorithm, IPOConfig, create_ipo
from thinkrl.algorithms.kto import KTOAlgorithm, KTOConfig, create_kto
from thinkrl.algorithms.orpo import ORPOAlgorithm, ORPOConfig, create_orpo
from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig, create_ppo
from thinkrl.algorithms.prime import PRIMEAlgorithm, PRIMEConfig, create_prime
from thinkrl.algorithms.reinforce import REINFORCEAlgorithm, REINFORCEConfig, create_reinforce

# New algorithms (stubs)
from thinkrl.algorithms.reinforce_pp import REINFORCEPPAlgorithm, REINFORCEPPConfig, create_reinforce_pp
from thinkrl.algorithms.rloo import RLOOAlgorithm, RLOOConfig, create_rloo
from thinkrl.algorithms.star import STaRAlgorithm, STaRConfig, create_star
from thinkrl.algorithms.vapo import VAPOAlgorithm, VAPOConfig


# Algorithm registry for factory pattern
ALGORITHMS = {
    "ppo": PPOAlgorithm,
    "grpo": GRPOAlgorithm,
    "dpo": DPOAlgorithm,
    "dapo": DAPOAlgorithm,
    "vapo": VAPOAlgorithm,
    "reinforce": REINFORCEAlgorithm,
    # New algorithms
    "reinforce_pp": REINFORCEPPAlgorithm,
    "reinforce++": REINFORCEPPAlgorithm,
    "rloo": RLOOAlgorithm,
    "prime": PRIMEAlgorithm,
    "dr_grpo": DrGRPOAlgorithm,
    "drgrpo": DrGRPOAlgorithm,
    "ipo": IPOAlgorithm,
    "kto": KTOAlgorithm,
    "star": STaRAlgorithm,
    "orpo": ORPOAlgorithm,
}

CONFIGS = {
    "ppo": PPOConfig,
    "grpo": GRPOConfig,
    "dpo": DPOConfig,
    "dapo": DAPOConfig,
    "vapo": VAPOConfig,
    "reinforce": REINFORCEConfig,
    # New configs
    "reinforce_pp": REINFORCEPPConfig,
    "reinforce++": REINFORCEPPConfig,
    "rloo": RLOOConfig,
    "prime": PRIMEConfig,
    "dr_grpo": DrGRPOConfig,
    "drgrpo": DrGRPOConfig,
    "ipo": IPOConfig,
    "kto": KTOConfig,
    "star": STaRConfig,
    "orpo": ORPOConfig,
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
    # REINFORCE++
    "REINFORCEPPAlgorithm",
    "REINFORCEPPConfig",
    "create_reinforce_pp",
    # RLOO
    "RLOOAlgorithm",
    "RLOOConfig",
    "create_rloo",
    # PRIME
    "PRIMEAlgorithm",
    "PRIMEConfig",
    "create_prime",
    # DrGRPO
    "DrGRPOAlgorithm",
    "DrGRPOConfig",
    "create_dr_grpo",
    # IPO
    "IPOAlgorithm",
    "IPOConfig",
    "create_ipo",
    # KTO
    "KTOAlgorithm",
    "KTOConfig",
    "create_kto",
    # STaR
    "STaRAlgorithm",
    "STaRConfig",
    "create_star",
    # ORPO
    "ORPOAlgorithm",
    "ORPOConfig",
    "create_orpo",
    # Registry
    "ALGORITHMS",
    "CONFIGS",
    "get_algorithm",
    "get_config",
]
