"""
ThinkRL Algorithms Module

This module provides implementations of state-of-the-art reinforcement learning algorithms
for human feedback (RLHF) training, including VAPO, DAPO, GRPO, PPO, and REINFORCE.

Available Algorithms:
    - VAPO: Value-model-based Augmented PPO with Length-adaptive GAE
    - DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization
    - GRPO: Group Relative Policy Optimization
    - PPO: Enhanced Proximal Policy Optimization
    - REINFORCE: Policy gradient with variance reduction

Example:
    >>> from thinkrl.algorithms import VAPO, PPO
    >>>
    >>> # Initialize VAPO algorithm
    >>> vapo = VAPO(config=vapo_config)
    >>>
    >>> # Initialize PPO algorithm
    >>> ppo = PPO(config=ppo_config)
"""

from typing import Dict, Type, Union

# Import base classes
from .base import BaseAlgorithm, AlgorithmConfig

# Import algorithm implementations
from .vapo import VAPO, VAPOConfig
from .dapo import DAPO, DAPOConfig
from .grpo import GRPO, GRPOConfig
from .ppo import PPO, PPOConfig
from .reinforce import REINFORCE, REINFORCEConfig

# Version information
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Base classes
    "BaseAlgorithm",
    "AlgorithmConfig",
    # Algorithm classes
    "VAPO",
    "DAPO",
    "GRPO",
    "PPO",
    "REINFORCE",
    # Configuration classes
    "VAPOConfig",
    "DAPOConfig",
    "GRPOConfig",
    "PPOConfig",
    "REINFORCEConfig",
    # Utility functions
    "get_algorithm",
    "list_algorithms",
    "get_algorithm_config",
]

# Algorithm registry for dynamic loading
ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "vapo": VAPO,
    "dapo": DAPO,
    "grpo": GRPO,
    "ppo": PPO,
    "reinforce": REINFORCE,
}

# Configuration registry
CONFIG_REGISTRY: Dict[str, Type[AlgorithmConfig]] = {
    "vapo": VAPOConfig,
    "dapo": DAPOConfig,
    "grpo": GRPOConfig,
    "ppo": PPOConfig,
    "reinforce": REINFORCEConfig,
}


def get_algorithm(name: str) -> Type[BaseAlgorithm]:
    """
    Get algorithm class by name.

    Args:
        name: Algorithm name (case-insensitive)

    Returns:
        Algorithm class

    Raises:
        ValueError: If algorithm name is not recognized

    Example:
        >>> PPOClass = get_algorithm("ppo")
        >>> ppo = PPOClass(config)
    """
    name = name.lower().strip()
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm '{name}'. Available algorithms: {available}"
        )
    return ALGORITHM_REGISTRY[name]


def get_algorithm_config(name: str) -> Type[AlgorithmConfig]:
    """
    Get algorithm configuration class by name.

    Args:
        name: Algorithm name (case-insensitive)

    Returns:
        Algorithm configuration class

    Raises:
        ValueError: If algorithm name is not recognized

    Example:
        >>> PPOConfigClass = get_algorithm_config("ppo")
        >>> config = PPOConfigClass(learning_rate=3e-4)
    """
    name = name.lower().strip()
    if name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm '{name}'. Available algorithms: {available}"
        )
    return CONFIG_REGISTRY[name]


def list_algorithms() -> Dict[str, str]:
    """
    List all available algorithms with descriptions.

    Returns:
        Dictionary mapping algorithm names to descriptions

    Example:
        >>> algorithms = list_algorithms()
        >>> print(algorithms["vapo"])
    """
    descriptions = {
        "vapo": "Value-model-based Augmented PPO with Length-adaptive GAE",
        "dapo": "Decoupled Clip and Dynamic Sampling Policy Optimization",
        "grpo": "Group Relative Policy Optimization",
        "ppo": "Enhanced Proximal Policy Optimization",
        "reinforce": "Policy gradient with variance reduction",
    }
    return descriptions


def create_algorithm(
    name: str, config: Union[AlgorithmConfig, Dict, None] = None, **kwargs
) -> BaseAlgorithm:
    """
    Create algorithm instance by name with configuration.

    Args:
        name: Algorithm name (case-insensitive)
        config: Algorithm configuration (dict, config object, or None)
        **kwargs: Additional configuration parameters

    Returns:
        Initialized algorithm instance

    Raises:
        ValueError: If algorithm name is not recognized

    Example:
        >>> # Create with config object
        >>> config = PPOConfig(learning_rate=3e-4)
        >>> ppo = create_algorithm("ppo", config)
        >>>
        >>> # Create with dict
        >>> ppo = create_algorithm("ppo", {"learning_rate": 3e-4})
        >>>
        >>> # Create with kwargs
        >>> ppo = create_algorithm("ppo", learning_rate=3e-4)
    """
    algorithm_class = get_algorithm(name)
    config_class = get_algorithm_config(name)

    # Handle different config input types
    if config is None:
        config = config_class(**kwargs)
    elif isinstance(config, dict):
        # Merge dict config with kwargs
        merged_config = {**config, **kwargs}
        config = config_class(**merged_config)
    elif isinstance(config, AlgorithmConfig):
        # Update config with kwargs if provided
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"Unknown config parameter: {key}")
    else:
        raise TypeError(
            f"Config must be AlgorithmConfig, dict, or None. Got {type(config)}"
        )

    return algorithm_class(config)


# Convenience aliases for common algorithms
PPOAlgorithm = PPO
VAPOAlgorithm = VAPO
DAPOAlgorithm = DAPO
GRPOAlgorithm = GRPO
REINFORCEAlgorithm = REINFORCE
