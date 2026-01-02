"""
ThinkRL Loss Functions
=======================

Loss functions for RLHF training.
Aligned with OpenRLHF patterns.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTLMLoss(nn.Module):
    """
    Language modeling loss for causal LM training.

    Standard cross-entropy loss with proper handling of
    padding tokens and distributed training.
    """

    def __init__(self, ignore_index: int = -100):
        """
        Initialize the LM loss.

        Args:
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute language modeling loss.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Check if all labels are ignored
        valid_mask = shift_labels != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self.loss_fn(shift_logits, shift_labels)


class SFTLoss(nn.Module):
    """
    Supervised fine-tuning loss.

    Computes masked mean of negative log probabilities.
    """

    def __init__(self, ignore_index: int = -100):
        """
        Initialize SFT loss.

        Args:
            ignore_index: Index to ignore
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SFT loss.

        Args:
            log_probs: Log probabilities [batch, seq_len]
            labels: Target labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Create mask for valid positions
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = labels != self.ignore_index

        # Compute masked loss
        masked_log_probs = log_probs * mask
        loss = -masked_log_probs.sum() / mask.sum().clamp(min=1)

        return loss


class PolicyLoss(nn.Module):
    """
    Policy loss for PPO and variants.

    Supports clipping, dual-clip, and importance sampling.
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        dual_clip: float | None = None,
    ):
        """
        Initialize policy loss.

        Args:
            clip_eps: PPO clipping epsilon
            dual_clip: Dual clip threshold (None to disable)
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.dual_clip = dual_clip

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute policy loss.

        Args:
            log_probs: Current policy log probs [batch, seq_len]
            old_log_probs: Old policy log probs [batch, seq_len]
            advantages: Advantages [batch, seq_len]
            action_mask: Mask for action tokens [batch, seq_len]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute importance ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        # Surrogate losses
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # PPO loss (take minimum)
        loss = -torch.min(surr1, surr2)

        # Dual clip for very negative advantages
        if self.dual_clip is not None:
            dual_clip_loss = -self.dual_clip * advantages
            loss = torch.where(
                advantages < 0,
                torch.max(loss, dual_clip_loss),
                loss,
            )

        # Apply action mask
        if action_mask is not None:
            loss = loss * action_mask
            num_actions = action_mask.sum().clamp(min=1)
        else:
            num_actions = loss.numel()

        # Mean loss
        policy_loss = loss.sum() / num_actions

        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)).float()
            if action_mask is not None:
                clip_fraction = (clip_fraction * action_mask).sum() / num_actions
            else:
                clip_fraction = clip_fraction.mean()

            approx_kl = 0.5 * ((log_probs - old_log_probs) ** 2).mean()

        metrics = {
            "policy_loss": policy_loss.detach(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "ratio_mean": ratio.mean().detach(),
        }

        return policy_loss, metrics


class ValueLoss(nn.Module):
    """
    Value function loss for PPO.

    Supports value clipping for stability.
    """

    def __init__(
        self,
        clip_eps: float | None = None,
    ):
        """
        Initialize value loss.

        Args:
            clip_eps: Value clipping epsilon (None to disable)
        """
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute value loss.

        Args:
            values: Current value estimates [batch, seq_len]
            old_values: Old value estimates [batch, seq_len]
            returns: Target returns [batch, seq_len]
            action_mask: Mask for action tokens [batch, seq_len]

        Returns:
            Scalar value loss
        """
        if self.clip_eps is not None:
            # Clipped value loss
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_eps,
                self.clip_eps,
            )
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (values_clipped - returns) ** 2
            loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        else:
            loss = 0.5 * (values - returns) ** 2

        # Apply action mask
        if action_mask is not None:
            loss = loss * action_mask
            num_actions = action_mask.sum().clamp(min=1)
        else:
            num_actions = loss.numel()

        return loss.sum() / num_actions


class PairWiseLoss(nn.Module):
    """
    Pairwise ranking loss for reward model training.

    Uses log-sigmoid of reward differences.
    """

    def __init__(self, margin: float = 0.0):
        """
        Initialize pairwise loss.

        Args:
            margin: Margin for preference pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute pairwise ranking loss.

        Args:
            chosen_rewards: Rewards for chosen responses [batch]
            rejected_rewards: Rewards for rejected responses [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards - self.margin)
        loss = loss.mean()

        # Compute accuracy
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_diff = (chosen_rewards - rejected_rewards).mean()

        metrics = {
            "rm_loss": loss.detach(),
            "rm_accuracy": accuracy,
            "reward_diff": reward_diff,
            "chosen_reward_mean": chosen_rewards.mean().detach(),
            "rejected_reward_mean": rejected_rewards.mean().detach(),
        }

        return loss, metrics


class LogExpLoss(nn.Module):
    """
    Log-exp loss for reward model training.

    Alternative to pairwise loss: log(1 + exp(r_reject - r_chosen))
    """

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute log-exp loss.

        Args:
            chosen_rewards: Rewards for chosen responses [batch]
            rejected_rewards: Rewards for rejected responses [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        loss = torch.log(1 + torch.exp(rejected_rewards - chosen_rewards))
        loss = loss.mean()

        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "rm_loss": loss.detach(),
            "rm_accuracy": accuracy,
        }

        return loss, metrics


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss.

    Supports standard DPO and IPO variants.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        ipo: bool = False,
    ):
        """
        Initialize DPO loss.

        Args:
            beta: Temperature parameter
            label_smoothing: Label smoothing factor
            ipo: Use IPO variant instead of standard DPO
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Policy log probs for chosen [batch]
            policy_rejected_logps: Policy log probs for rejected [batch]
            reference_chosen_logps: Reference log probs for chosen [batch]
            reference_rejected_logps: Reference log probs for rejected [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Compute implicit rewards
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        if self.ipo:
            # IPO loss
            loss = (chosen_logratios - rejected_logratios - 1 / (2 * self.beta)) ** 2
        else:
            # Standard DPO loss
            logits = chosen_rewards - rejected_rewards

            if self.label_smoothing > 0:
                # Soft labels
                loss = (
                    -F.logsigmoid(logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-logits) * self.label_smoothing
                )
            else:
                loss = -F.logsigmoid(logits)

        loss = loss.mean()

        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

        metrics = {
            "dpo_loss": loss.detach(),
            "dpo_accuracy": accuracy,
            "chosen_reward": chosen_rewards.mean().detach(),
            "rejected_reward": rejected_rewards.mean().detach(),
            "reward_margin": reward_margin,
        }

        return loss, metrics


class KTOLoss(nn.Module):
    """
    Kahneman-Tversky Optimization loss.

    For preference learning with unpaired data.
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ):
        """
        Initialize KTO loss.

        Args:
            beta: Temperature parameter
            desirable_weight: Weight for desirable examples
            undesirable_weight: Weight for undesirable examples
        """
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        labels: torch.Tensor,
        kl_div: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute KTO loss.

        Args:
            policy_logps: Policy log probs [batch]
            reference_logps: Reference log probs [batch]
            labels: Binary labels (1 for desirable, 0 for undesirable) [batch]
            kl_div: Optional pre-computed KL divergence

        Returns:
            Tuple of (loss, metrics_dict)
        """
        logratios = policy_logps - reference_logps

        if kl_div is None:
            kl_div = logratios.mean()

        # Separate desirable and undesirable
        desirable_mask = labels == 1
        undesirable_mask = labels == 0

        # Compute losses
        desirable_loss = torch.zeros(1, device=policy_logps.device)
        undesirable_loss = torch.zeros(1, device=policy_logps.device)

        if desirable_mask.any():
            desirable_logratios = logratios[desirable_mask]
            desirable_loss = -F.logsigmoid(self.beta * (desirable_logratios - kl_div))
            desirable_loss = desirable_loss.mean() * self.desirable_weight

        if undesirable_mask.any():
            undesirable_logratios = logratios[undesirable_mask]
            undesirable_loss = -F.logsigmoid(self.beta * (kl_div - undesirable_logratios))
            undesirable_loss = undesirable_loss.mean() * self.undesirable_weight

        loss = desirable_loss + undesirable_loss

        metrics = {
            "kto_loss": loss.detach(),
            "desirable_loss": desirable_loss.detach(),
            "undesirable_loss": undesirable_loss.detach(),
            "kl_div": kl_div.detach(),
        }

        return loss, metrics


class EntropyLoss(nn.Module):
    """
    Entropy bonus for policy exploration.
    """

    def __init__(self, coef: float = 0.01):
        """
        Initialize entropy loss.

        Args:
            coef: Entropy coefficient (higher = more exploration)
        """
        super().__init__()
        self.coef = coef

    def forward(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute entropy bonus (negative loss to maximize entropy).

        Args:
            logits: Policy logits [batch, seq_len, vocab_size]
            action_mask: Mask for action tokens [batch, seq_len]

        Returns:
            Entropy loss (negative, to be added to total loss)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        if action_mask is not None:
            entropy = entropy * action_mask
            mean_entropy = entropy.sum() / action_mask.sum().clamp(min=1)
        else:
            mean_entropy = entropy.mean()

        # Return negative because we want to maximize entropy
        return -self.coef * mean_entropy
