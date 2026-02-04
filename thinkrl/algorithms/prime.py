"""
PRIME Algorithm Implementation
==============================

Process Reinforcement through Implicit Rewards (PRIME).

PRIME addresses the sparsity of outcome rewards by learning an "Implicit PRM"
online. It updates a reward model (initialized from the policy/SFT model)
using outcome labels, then uses this evolving PRM to provide dense, token-level
feedback to the policy.

Key Components:
1. Implicit PRM Update: The PRM is trained to maximize the likelihood of correct
   outcomes vs incorrect ones (via PRIMELoss).
2. Advantage Estimation: Uses RLOO (Reinforce Leave-One-Out) baseline with
   mixed rewards (Outcome Reward + Implicit Process Reward).
3. Policy Update: Standard PPO-Clip objective.

Reference:
    PRIME: Process Reinforcement through Implicit Rewards (https://arxiv.org/abs/2502.01456)
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.models.loss import PolicyLoss, PRIMELoss
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PRIMEConfig:
    """
    Configuration for PRIME algorithm.
    """

    # PRIME Hyperparameters
    beta: float = 0.05  # Coefficient for implicit reward scaling (Eq. 3)
    prm_learning_rate: float = 1e-6  # Learning rate for the Implicit PRM

    # Advantage Estimation
    gamma: float = 1.0  # Discount factor (Paper uses 1.0 for math/code)
    advantage_estimator: str = "rloo"  # 'rloo' (default/best) or 'reinforce'

    # Policy Optimization (PPO)
    clip_epsilon: float = 0.2
    n_epochs: int = 1
    batch_size: int = 64

    # Generation
    num_generations_per_prompt: int = 8  # K samples (Algorithm 1, Line 3)
    max_new_tokens: int = 512
    temperature: float = 1.0


class PRIMEAlgorithm(BaseRLHFAlgorithm):
    """
    PRIME Algorithm.

    Manages two trainable models:
    1. Policy Model (pi_theta): The actor being optimized.
    2. Implicit PRM (pi_phi): The reward model being learned online.

    And one frozen model:
    3. Reference Model (pi_ref): Frozen SFT baseline (used for both KL and Implicit Reward calc).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None = None,
        prm_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: PRIMEConfig | None = None,
        tokenizer: Any | None = None,
        **kwargs,
    ):
        # Default config
        config = config or PRIMEConfig()

        # Initialize Base Algorithm (handles policy_model setup)
        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.config: PRIMEConfig = config

        # 1. Setup Reference Model (Frozen)
        # If not provided, clone from policy (SFT start)
        if self.ref_model is None:
            logger.info("PRIME: No reference model provided. Cloning initial policy as reference.")
            self.ref_model = copy.deepcopy(self.policy_model)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # 2. Setup Implicit PRM (Trainable)
        # Paper: "Initialize... Implicit PRM... with the reference model" (SFT)
        if prm_model is None:
            logger.info("PRIME: No PRM model provided. Cloning reference model as Implicit PRM.")
            self.prm_model = copy.deepcopy(self.ref_model)
        else:
            self.prm_model = prm_model

        # PRM needs gradients
        self.prm_model.train()
        self.prm_model.requires_grad_(True)

        # PRM Optimizer (Separate from Policy Optimizer)
        self.prm_optimizer = torch.optim.AdamW(self.prm_model.parameters(), lr=config.prm_learning_rate)

        # 3. Loss Functions
        self.prime_loss_fn = PRIMELoss(beta=config.beta)
        self.policy_loss_fn = PolicyLoss(clip_eps=config.clip_epsilon)

    def compute_implicit_rewards(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute token-level implicit rewards: r_phi(y_t)

        Formula: r_phi(y_t) = beta * ( log_prm(y_t) - log_ref(y_t) )
        """
        with torch.no_grad():
            # Get PRM log probs
            # Note: We use the *current* state of the PRM for reward calculation
            # But we detach to ensure we don't backprop through the reward signal into PRM here
            prm_outputs = self.prm_model(input_ids=input_ids, attention_mask=attention_mask)
            prm_log_probs = self.get_log_probs(prm_outputs.logits, input_ids)

            # Get Ref log probs
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = self.get_log_probs(ref_outputs.logits, input_ids)

        # Calculate Reward (Eq. 3)
        # r_phi = beta * log( pi_phi / pi_ref )
        implicit_rewards = self.config.beta * (prm_log_probs - ref_log_probs)

        return implicit_rewards.detach()

    def update_prm(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Update the Implicit PRM using outcome labels (Algorithm 1, Lines 6-8).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outcome_labels = batch["rewards"]  # Assuming binary rewards 1.0/0.0

        # Forward pass on PRM
        self.prm_model.train()
        prm_outputs = self.prm_model(input_ids=input_ids, attention_mask=attention_mask)
        prm_log_probs = self.get_log_probs(prm_outputs.logits, input_ids)  # [B, L]

        # Forward pass on Ref (Frozen)
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = self.get_log_probs(ref_outputs.logits, input_ids)  # [B, L]

        # Completion Mask (ignore prompt)
        # Assuming labels have -100 for prompt, or we construct mask dynamically
        if "labels" in batch:
            action_mask = (batch["labels"] != -100).float()
        else:
            # Fallback: Treat all non-padding as action (risky if prompt not masked)
            action_mask = attention_mask.float()

        # Compute PRIMELoss
        loss, metrics = self.prime_loss_fn(
            log_probs=prm_log_probs,
            ref_log_probs=ref_log_probs,
            outcome_labels=outcome_labels,
            action_mask=action_mask,
        )

        # Optimization Step
        self.prm_optimizer.zero_grad()
        loss.backward()
        # Optional: Clip grad norm for PRM
        nn.utils.clip_grad_norm_(self.prm_model.parameters(), 1.0)
        self.prm_optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    def compute_advantages(
        self, implicit_rewards: torch.Tensor, outcome_rewards: torch.Tensor, action_mask: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """
        Compute mixed advantages using RLOO baseline (Algorithm 1, Eq. 5).

        Advantage = A_process + A_outcome
        A_process = Sum[ r_phi - LOO_Mean(r_phi) ]
        A_outcome = r_o - LOO_Mean(r_o)
        """
        # Reshape to [Group, Sample, Seq] for broadcasting
        # Assuming batch is perfectly divisible by group_size (K)
        batch_size, seq_len = implicit_rewards.shape
        num_groups = batch_size // group_size

        # 1. Process Reward Advantage (Token-level RLOO)
        # Shape: [G, K, S]
        proc_r = implicit_rewards.view(num_groups, group_size, seq_len)

        # Calculate Leave-One-Out Mean for Process Rewards
        # Sum over K dim, subtract self, divide by K-1
        sum_proc_r = proc_r.sum(dim=1, keepdim=True)  # [G, 1, S]
        loo_mean_proc = (sum_proc_r - proc_r) / (group_size - 1)

        # A_process_t = r_phi_t - baseline_t
        adv_process = proc_r - loo_mean_proc

        # Apply discount (gamma) - usually 1.0 for this paper
        # If gamma < 1.0, we would need cumulative sum logic here.
        # For simple sum (gamma=1), we can just sum the token advantages later for the return,
        # but standard RLOO usually keeps token-level signals for token-level updates.
        # Paper Eq 5 implies summation from t to T.
        if self.config.gamma == 1.0:
            # Simple accumulation from t to T
            # We can use a simplified approach: just the token advantage
            # Or strictly implement the sum: A_t = Sum_{s=t}^T (r_s - b_s)
            # Efficient implementation: Flip, Cumsum, Flip
            adv_process = torch.flip(torch.cumsum(torch.flip(adv_process, dims=[2]), dim=2), dims=[2])

        # 2. Outcome Reward Advantage (Sequence-level RLOO)
        # Shape: [G, K]
        out_r = outcome_rewards.view(num_groups, group_size)
        sum_out_r = out_r.sum(dim=1, keepdim=True)
        loo_mean_out = (sum_out_r - out_r) / (group_size - 1)

        adv_outcome = out_r - loo_mean_out

        # Broadcast outcome advantage to sequence length [G, K, S]
        adv_outcome = adv_outcome.unsqueeze(-1).expand_as(adv_process)

        # 3. Combine
        total_advantage = adv_process + adv_outcome

        # Flatten back to [B, S]
        total_advantage = total_advantage.view(batch_size, seq_len)

        # Mask out padding/prompt
        total_advantage = total_advantage * action_mask

        return total_advantage

    def train_on_rollout(self, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
        """
        Execute one iteration of PRIME (Algorithm 1 Loop).

        Steps:
        1. Update Implicit PRM (using current batch outcomes).
        2. Compute Implicit Rewards (using updated PRM).
        3. Compute Advantages (RLOO on combined rewards).
        4. Update Policy (PPO).
        """
        metrics = {}

        # --- Step 1: Update Implicit PRM ---
        # Note: Paper says update PRM *before* calculating advantages for the policy step
        prm_metrics = self.update_prm(batch)
        metrics.update(prm_metrics)

        # --- Step 2: Compute Rewards & Advantages ---
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outcome_rewards = batch["rewards"]

        # Get action mask (completion only)
        if "labels" in batch:
            action_mask = (batch["labels"] != -100).float()
        else:
            action_mask = attention_mask.float()  # Fallback

        # Implicit Rewards r_phi
        implicit_rewards = self.compute_implicit_rewards(input_ids, attention_mask)
        implicit_rewards = implicit_rewards * action_mask

        # Compute Advantages
        advantages = self.compute_advantages(
            implicit_rewards=implicit_rewards,
            outcome_rewards=outcome_rewards,
            action_mask=action_mask,
            group_size=self.config.num_generations_per_prompt,
        )

        # --- Step 3: Update Policy (PPO) ---
        # We need old_log_probs for PPO ratio
        with torch.no_grad():
            outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            old_log_probs = self.get_log_probs(outputs.logits, input_ids)

        policy_metrics_list = []

        # PPO Epochs
        for _ in range(self.config.n_epochs):
            self.optimizer.zero_grad()

            # Forward Policy
            outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = self.get_log_probs(outputs.logits, input_ids)

            # PPO Loss
            loss, pol_metrics = self.policy_loss_fn(
                log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages, action_mask=action_mask
            )

            loss.backward()

            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.clip_grad_norm)

            self.optimizer.step()
            policy_metrics_list.append({k: v.item() for k, v in pol_metrics.items()})

        # Average policy metrics
        avg_pol_metrics = {
            k: sum(d[k] for d in policy_metrics_list) / len(policy_metrics_list) for k in policy_metrics_list[0]
        }
        metrics.update(avg_pol_metrics)

        # Add scalar metrics for inspection
        metrics["prime/outcome_reward_mean"] = outcome_rewards.mean().item()
        metrics["prime/implicit_reward_mean"] = implicit_rewards[action_mask.bool()].mean().item()

        return [metrics]


def create_prime(
    policy_model: nn.Module,
    ref_model: nn.Module | None = None,
    prm_model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    learning_rate: float = 1e-6,
    beta: float = 0.05,
    **kwargs,
) -> PRIMEAlgorithm:
    """
    Factory function to create PRIME algorithm.

    Args:
        policy_model: Policy model
        ref_model: Reference model (optional, cloned if None)
        prm_model: Implicit PRM model (optional, cloned if None)
        optimizer: Optimizer (optional)
        learning_rate: Learning rate for PRM (policy LR is separate or shared?)
                       Note: PRIMEAlgorithm takes config which has prm_learning_rate.
                       Ideally we pass separate LRs.
                       For now, let's map this to prm_learning_rate or policy?
                       PRIMEConfig has prm_learning_rate. Base has learning_rate.
        beta: Reward coefficient
        **kwargs: Additional args
    """
    # Extract config args
    config_args = {k: v for k, v in kwargs.items() if hasattr(PRIMEConfig, k)}
    if "prm_learning_rate" not in config_args:
        # Default policy LR if not specified? Or specifically for factory:
        config_args["prm_learning_rate"] = learning_rate

    config = PRIMEConfig(beta=beta, **config_args)

    # Remaining args
    algo_kwargs = {k: v for k, v in kwargs.items() if k not in config_args}

    return PRIMEAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model,
        prm_model=prm_model,
        optimizer=optimizer,
        config=config,
        learning_rate=learning_rate,  # Base algo LR
        **algo_kwargs,
    )


__all__ = ["PRIMEConfig", "PRIMEAlgorithm", "create_prime"]
