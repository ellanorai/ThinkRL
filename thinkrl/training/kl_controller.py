"""
KL Controller for Adaptive KL Penalty
======================================

Dynamically adjusts KL coefficient during RLHF training to maintain
the policy within a target KL divergence from the reference model.

Inspired by OpenRLHF and TRL implementations.

Author: EllanorAI
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any


logger = logging.getLogger(__name__)


class KLControllerType(str, Enum):
    """KL controller types."""

    FIXED = "fixed"  # Constant KL coefficient
    ADAPTIVE = "adaptive"  # PID-like adjustment
    LINEAR = "linear"  # Linear schedule
    COSINE = "cosine"  # Cosine schedule


@dataclass
class KLControllerConfig:
    """Configuration for KL Controller."""

    # Controller type
    controller_type: KLControllerType = KLControllerType.ADAPTIVE

    # Initial KL coefficient
    init_kl_coef: float = 0.1

    # Target KL divergence
    target_kl: float = 0.01

    # Adaptive controller parameters
    kl_horizon: int = 10000  # Steps over which to adjust
    kl_lr: float = 0.1  # Learning rate for KL adjustment

    # Bounds for adaptive control
    min_kl_coef: float = 0.001
    max_kl_coef: float = 10.0

    # For linear/cosine schedules
    final_kl_coef: float = 0.01
    total_steps: int = 100000


class KLController:
    """
    Adaptive KL penalty controller for RLHF training.

    Adjusts the KL coefficient to maintain the policy within a target
    KL divergence from the reference model. This prevents the policy
    from diverging too far while still allowing meaningful updates.

    Example:
        >>> controller = KLController(
        ...     init_kl_coef=0.1,
        ...     target_kl=0.01,
        ... )
        >>> for step in range(num_steps):
        ...     # Compute KL divergence
        ...     kl = compute_kl(policy_logprobs, ref_logprobs)
        ...     # Update controller and get current coefficient
        ...     kl_coef = controller.update(kl)
        ...     # Use in loss computation
        ...     loss = policy_loss - kl_coef * kl
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        controller_type: KLControllerType | str = KLControllerType.ADAPTIVE,
        kl_horizon: int = 10000,
        kl_lr: float = 0.1,
        min_kl_coef: float = 0.001,
        max_kl_coef: float = 10.0,
        final_kl_coef: float = 0.01,
        total_steps: int = 100000,
    ):
        """
        Initialize KL controller.

        Args:
            init_kl_coef: Initial KL penalty coefficient
            target_kl: Target KL divergence to maintain
            controller_type: Type of controller ("fixed", "adaptive", "linear", "cosine")
            kl_horizon: Number of steps for adjustment horizon
            kl_lr: Learning rate for adaptive adjustment
            min_kl_coef: Minimum KL coefficient
            max_kl_coef: Maximum KL coefficient
            final_kl_coef: Final KL coefficient for scheduled controllers
            total_steps: Total training steps for scheduled controllers
        """
        if isinstance(controller_type, str):
            controller_type = KLControllerType(controller_type.lower())

        self.controller_type = controller_type
        self.init_kl_coef = init_kl_coef
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.kl_horizon = kl_horizon
        self.kl_lr = kl_lr
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef
        self.final_kl_coef = final_kl_coef
        self.total_steps = total_steps

        # State tracking
        self.step = 0
        self.kl_history: list[float] = []

    @classmethod
    def from_config(cls, config: KLControllerConfig) -> KLController:
        """Create controller from config."""
        return cls(
            init_kl_coef=config.init_kl_coef,
            target_kl=config.target_kl,
            controller_type=config.controller_type,
            kl_horizon=config.kl_horizon,
            kl_lr=config.kl_lr,
            min_kl_coef=config.min_kl_coef,
            max_kl_coef=config.max_kl_coef,
            final_kl_coef=config.final_kl_coef,
            total_steps=config.total_steps,
        )

    def update(self, current_kl: float) -> float:
        """
        Update KL coefficient based on current KL divergence.

        Args:
            current_kl: Current KL divergence value

        Returns:
            Updated KL coefficient
        """
        self.step += 1
        self.kl_history.append(current_kl)

        if self.controller_type == KLControllerType.FIXED:
            return self.kl_coef

        elif self.controller_type == KLControllerType.ADAPTIVE:
            return self._adaptive_update(current_kl)

        elif self.controller_type == KLControllerType.LINEAR:
            return self._linear_update()

        elif self.controller_type == KLControllerType.COSINE:
            return self._cosine_update()

        return self.kl_coef

    def _adaptive_update(self, current_kl: float) -> float:
        """
        Adaptive PID-like update of KL coefficient.

        Increases coefficient if KL is too high (policy diverging),
        decreases if KL is too low (policy not learning).
        """
        # Compute error
        error = current_kl - self.target_kl

        # Proportional adjustment
        # If error > 0 (KL too high), increase coefficient
        # If error < 0 (KL too low), decrease coefficient
        adjustment = 1.0 + self.kl_lr * error / self.target_kl

        # Apply adjustment with bounds
        self.kl_coef = self.kl_coef * adjustment
        self.kl_coef = max(self.min_kl_coef, min(self.max_kl_coef, self.kl_coef))

        # Log significant changes
        if self.step % 100 == 0:
            logger.debug(
                f"KL Controller step {self.step}: "
                f"kl={current_kl:.6f}, target={self.target_kl:.6f}, "
                f"kl_coef={self.kl_coef:.6f}"
            )

        return self.kl_coef

    def _linear_update(self) -> float:
        """Linear schedule from init to final coefficient."""
        progress = min(self.step / self.total_steps, 1.0)
        self.kl_coef = self.init_kl_coef + progress * (self.final_kl_coef - self.init_kl_coef)
        return self.kl_coef

    def _cosine_update(self) -> float:
        """Cosine annealing schedule."""
        import math

        progress = min(self.step / self.total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.kl_coef = self.final_kl_coef + cosine_decay * (self.init_kl_coef - self.final_kl_coef)
        return self.kl_coef

    def get_kl_coef(self) -> float:
        """Get current KL coefficient without updating."""
        return self.kl_coef

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.kl_coef = self.init_kl_coef
        self.step = 0
        self.kl_history = []

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics."""
        import statistics

        stats = {
            "kl_coef": self.kl_coef,
            "step": self.step,
            "controller_type": self.controller_type.value,
            "target_kl": self.target_kl,
        }

        if self.kl_history:
            stats["mean_kl"] = statistics.mean(self.kl_history[-100:])
            stats["max_kl"] = max(self.kl_history[-100:])
            stats["min_kl"] = min(self.kl_history[-100:])

        return stats

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "kl_coef": self.kl_coef,
            "step": self.step,
            "kl_history": self.kl_history[-1000:],  # Keep last 1000
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.kl_coef = state.get("kl_coef", self.init_kl_coef)
        self.step = state.get("step", 0)
        self.kl_history = state.get("kl_history", [])


class AdaptiveKLController(KLController):
    """
    Convenience class for adaptive KL control.

    This is the recommended controller for most RLHF applications.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        kl_lr: float = 0.1,
        min_kl_coef: float = 0.001,
        max_kl_coef: float = 10.0,
    ):
        super().__init__(
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
            controller_type=KLControllerType.ADAPTIVE,
            kl_lr=kl_lr,
            min_kl_coef=min_kl_coef,
            max_kl_coef=max_kl_coef,
        )


class FixedKLController(KLController):
    """Fixed KL coefficient (no adaptation)."""

    def __init__(self, kl_coef: float = 0.1):
        super().__init__(
            init_kl_coef=kl_coef,
            controller_type=KLControllerType.FIXED,
        )


def create_kl_controller(
    controller_type: str = "adaptive",
    **kwargs,
) -> KLController:
    """
    Factory function to create KL controller.

    Args:
        controller_type: Type of controller
        **kwargs: Controller-specific arguments

    Returns:
        KLController instance
    """
    if controller_type == "adaptive":
        return AdaptiveKLController(**kwargs)
    elif controller_type == "fixed":
        return FixedKLController(**kwargs)
    else:
        return KLController(controller_type=controller_type, **kwargs)


__all__ = [
    "KLControllerType",
    "KLControllerConfig",
    "KLController",
    "AdaptiveKLController",
    "FixedKLController",
    "create_kl_controller",
]
