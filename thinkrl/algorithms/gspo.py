"""
GSPO Algorithm Implementation
==============================

Group Sequence Policy Optimization (GSPO), the algorithm behind Qwen3's RL
training.

GSPO keeps GRPO's group-relative advantage but replaces its token-level
importance ratio with a length-normalized, sequence-level one. The paper's
own ablations show this removes the instability GRPO's token-level ratio
causes when training Mixture-of-Experts models, and lets GSPO drop the
explicit KL penalty GRPO needs (beta=0 by default here, matching the paper).

References:
    GSPO: https://arxiv.org/abs/2507.18071
    GRPO: https://arxiv.org/abs/2402.03300

Author: EllanorAI
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.models.loss import GSPOLoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class GSPOConfig(GRPOConfig):
    """
    Configuration for GSPO. Inherits group_size/n_epochs/clip_grad_norm/
    advantage_eps/use_vllm/learning_rate from GRPOConfig.

    clip_epsilon (inherited) is not used -- GSPO clips its sequence-level
    ratio with the asymmetric epsilon_low/epsilon_high below instead, the
    same shape DAPO already uses for its own asymmetric clip.
    """

    epsilon_low: float = 3e-4
    epsilon_high: float = 4e-4
    # The paper runs with no explicit KL penalty: the sequence ratio's length
    # normalization is already far less volatile than GRPO's token ratio.
    beta: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        assert self.epsilon_low >= 0, "epsilon_low must be non-negative"
        assert self.epsilon_high >= 0, "epsilon_high must be non-negative"


class GSPOAlgorithm(GRPOAlgorithm):
    """
    Group Sequence Policy Optimization (GSPO).

    Reuses GRPOAlgorithm's group-relative advantage (compute_advantages) and
    its generic train_on_rollout/training_step loop unchanged; only
    compute_loss differs, swapping GRPOLoss's token-level ratio for
    GSPOLoss's sequence-level one.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: GSPOConfig | None = None,
        **kwargs,
    ):
        config = config or GSPOConfig()

        if config.beta > 0:
            logger.warning(
                "GSPOConfig.beta=%s but GSPOAlgorithm.compute_loss does not apply a KL "
                "penalty (the paper runs with beta=0; its sequence-level ratio is already "
                "far less volatile than GRPO's token-level one).",
                config.beta,
            )

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            config=config,
            **kwargs,
        )

        self.config: GSPOConfig = config
        # Not self.loss_fn: GRPOAlgorithm's own __init__ already set that to a
        # GRPOLoss, and this module (unlike grpo.py/papo.py) isn't in pyproject.toml's
        # mypy debt list, so overriding it with an incompatible type is held to strict
        # checking. A distinct name sidesteps it cleanly, matching VAPOAlgorithm's
        # policy_loss_fn/value_loss_fn/entropy_loss_fn rather than reusing loss_fn.
        self.gspo_loss_fn = GSPOLoss(epsilon_low=config.epsilon_low, epsilon_high=config.epsilon_high)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        J_GSPO = E[ 1/G sum_i min(s_i*A_i, clip(s_i)*A_i) ], s_i the sequence-level,
        length-normalized importance ratio (see GSPOLoss).

        Args:
            batch: Dict containing:
                - input_ids: [B, S]
                - attention_mask: [B, S]
                - labels: [B, S] (with -100 for prompt)
                - rewards: [B]
                - old_log_probs: [B, S] (fixed from sampling phase)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        rewards = batch["rewards"]
        old_log_probs = batch["old_log_probs"]

        # One advantage per sequence (GRPO's group-relative formula, unchanged).
        # Unlike GRPO, this is never expanded to [B, S]: GSPO's surrogate and
        # clipping operate on the sequence-level ratio, not a per-token one.
        advantages = self.compute_advantages(rewards)

        self.policy_model.train()
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = self.get_log_probs(outputs, labels)

        token_mask = (labels != -100).float()

        total_loss, metrics_loss = self.gspo_loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            action_mask=token_mask,
        )

        with torch.no_grad():
            metrics = {
                "advantage_mean": advantages.mean(),
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "clip_fraction": metrics_loss["clip_frac"],
                "sequence_ratio_mean": metrics_loss["sequence_ratio_mean"],
            }

        return {
            "loss": total_loss,
            **metrics,
        }


def create_gspo(
    policy_model: nn.Module,
    ref_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-6,
    group_size: int = 4,
    epsilon_low: float = 3e-4,
    epsilon_high: float = 4e-4,
    config: GSPOConfig | None = None,
    **kwargs,
) -> GSPOAlgorithm:
    """
    Factory function to create a GSPOAlgorithm instance.

    config: Pre-built GSPOConfig. If given, learning_rate/group_size/epsilon_low/
        epsilon_high/kwargs are ignored.
    """
    if config is None:
        config = GSPOConfig(
            learning_rate=learning_rate,
            group_size=group_size,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            **kwargs,
        )
    return GSPOAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
    )


__all__ = ["GSPOAlgorithm", "GSPOConfig", "create_gspo"]
