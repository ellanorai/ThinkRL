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
                    -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing
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


class PAPOLoss(nn.Module):
    """
    Perception-Aware Policy Optimization (PAPO) Loss.

    Combines Group Relative Policy Optimization (GRPO) loss with:
    1. Implicit Perception Loss (Bounded KL divergence between original and DETACHED masked policies).
       Note: We MAXIMIZE this KL divergence to encourage visual grounding, but bounded to prevent hallucinations.
    2. Double Entropy Loss (Regularization to prevent policy collapse).
       Note: We MAXIMIZE entropy (minimize negative entropy) to counter the KL maximization pressure.

    Reference: https://arxiv.org/abs/2507.06448
    """

    def __init__(
        self,
        gamma: float = 0.01,
        eta: float = 0.03,
        clip_eps: float = 0.2,
        beta: float = 0.04,
        kl_prcp_cap: float = 5.0,
    ):
        """
        Initialize PAPO loss.

        Args:
            gamma: Coefficient for Implicit Perception Loss (KL_prcp)
            eta: Coefficient for Double Entropy Loss
            clip_eps: Clipping epsilon for GRPO surrogate
            beta: Coefficient for KL divergence penalty (relative to reference)
            kl_prcp_cap: Maximum value for the perception KL term to prevent exploding gradients.
        """
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.clip_eps = clip_eps
        self.beta = beta
        self.kl_prcp_cap = kl_prcp_cap

    def set_gamma(self, new_gamma: float):
        """Update gamma coefficient (useful for annealing)."""
        self.gamma = new_gamma

    def set_beta(self, new_beta: float):
        """Update beta coefficient (useful for annealing)."""
        self.beta = new_beta

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        normalize_advantages: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute PAPO loss.

        Args:
            log_probs: Log probs of policy on original inputs [batch, seq_len]
            log_probs_mask: Log probs of policy on corrupted/masked inputs [batch, seq_len]
            old_log_probs: Log probs of old policy (for GRPO clipping) [batch, seq_len]
            advantages: Group relative advantages [batch, seq_len] or [batch]
            ref_log_probs: Log probs of reference model (optional) [batch, seq_len]
            action_mask: Mask for action tokens [batch, seq_len]
            normalize_advantages: Whether to normalize advantages within this batch (default: False)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Ensure advantages are properly broadcasted
        if advantages.dim() == 1 and log_probs.dim() == 2:
            advantages = advantages.unsqueeze(1)

        # Optional: Advantage Normalization
        # Helps with mixed-modal reward scales if not handled externally
        if normalize_advantages and advantages.numel() > 1:
            if action_mask is not None:
                # Normalize only based on valid tokens to avoid padding skew
                valid_adv = advantages[action_mask.bool()]
                mean, std = valid_adv.mean(), valid_adv.std()
                advantages = (advantages - mean) / (std + 1e-8)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 1. GRPO Surrogate Loss
        # ----------------------
        # Ratio = pi / old_pi = exp(log_pi - log_old_pi)
        log_ratio = torch.clamp(log_probs - old_log_probs, -20, 20)
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # Maximize surrogate -> Minimize negative surrogate
        loss_surrogate = -torch.min(surr1, surr2)

        # 2. Reference KL (Corrected Estimator)
        # -------------------------------------
        # Using positive-definite estimator matching Schulman approximation
        kl_ref = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and self.beta > 0:
            # log_ratio_ref = log(ref / pi) = ref_log_probs - log_probs
            log_ratio_ref = ref_log_probs - log_probs
            # KL approx: exp(ratio) - ratio - 1
            kl_ref = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

        loss_kl_ref = self.beta * kl_ref

        # 3. Implicit Perception Loss (Corrected & Bounded)
        # -------------------------------------------------
        # Objective: MAXIMIZE KL(pi || pi_mask)
        # Fix 1: Stop-gradient on pi_mask (detach) to prevent mutual divergence escalation
        # Fix 2: Cap the KL to prevent adversarial behavior/hallucination

        # log_ratio_prcp = log(pi / pi_mask_detached)
        log_ratio_prcp = torch.clamp(log_probs - log_probs_mask.detach(), -20, 20)
        ratio_prcp = torch.exp(log_ratio_prcp)

        # Schulman approx for KL: ratio - log_ratio - 1
        kl_prcp_raw = ratio_prcp - log_ratio_prcp - 1.0

        # Clamp to prevent explosion
        kl_prcp = torch.clamp(kl_prcp_raw, min=0.0, max=self.kl_prcp_cap)

        # We want to MAXIMIZE this, so we subtract it from the loss
        loss_prcp = -self.gamma * kl_prcp

        # 4. Double Entropy Loss (Corrected Sign)
        # ---------------------------------------
        # Objective: MAXIMIZE Entropy (Prevent collapse)
        # Fix 3: Maximize entropy H = -log_pi
        # Loss term: -eta * (H_pi + H_mask)

        entropy = -log_probs
        entropy_mask = -log_probs_mask  # We also encourage the masked policy to stay entropic

        # Maximizing entropy means Minimizing negative entropy
        loss_entropy = -self.eta * (entropy + entropy_mask)

        # Combine terms
        # ----------------------
        total_loss_per_token = loss_surrogate + loss_kl_ref + loss_prcp + loss_entropy

        # Apply masking and average
        if action_mask is not None:
            valid_count = action_mask.sum().clamp(min=1)
            total_loss = (total_loss_per_token * action_mask).sum() / valid_count

            # Metrics (detached)
            with torch.no_grad():
                metrics = {
                    "papo_loss": total_loss.detach(),
                    "surrogate_loss": (loss_surrogate * action_mask).sum() / valid_count,
                    "kl_ref_val": (kl_ref * action_mask).sum() / valid_count,
                    "kl_prcp_val": (kl_prcp * action_mask).sum() / valid_count,
                    "entropy_val": (entropy * action_mask).sum() / valid_count,
                    "clip_fraction": (
                        ((ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)).float() * action_mask
                    ).sum()
                    / valid_count,
                }
        else:
            total_loss = total_loss_per_token.mean()
            with torch.no_grad():
                metrics = {
                    "papo_loss": total_loss.detach(),
                    "surrogate_loss": loss_surrogate.mean().detach(),
                    "kl_prcp_val": kl_prcp.mean().detach(),
                    "entropy_val": entropy.mean().detach(),
                }

        return total_loss, metrics


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


class COPOLoss(nn.Module):
    """
    Count-based Online Preference Optimization (COPO) loss.

    Combines DPO loss with a count-based exploration bonus:
    J = J_DPO + alpha * E[1/sqrt(N)]
    """

    def __init__(
        self,
        beta: float = 0.1,
        alpha: float = 0.1,
        cfn_output_dim: int = 20,
        loss_type: str = "sigmoid",
    ):
        """
        Initialize COPO loss.

        Args:
            beta: DPO temperature / KL coefficient
            alpha: Exploration bonus coefficient
            cfn_output_dim: Output dimension of CFN (for scaling bonus)
            loss_type: DPO loss type ('sigmoid', 'hinge', 'ipo')
        """
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.cfn_output_dim = cfn_output_dim
        self.loss_type = loss_type

    def forward(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        chosen_hidden_states: torch.Tensor,
        cfn_model: nn.Module,
        batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute COPO loss.

        Args:
            policy_log_probs: Policy log probs [2*batch] (concatenated chosen+rejected)
            ref_log_probs: Reference log probs [2*batch]
            chosen_hidden_states: Hidden states for chosen responses [batch, dim]
            cfn_model: Coin Flipping Network model (for bonus calculation)
            batch_size: Batch size (half of log_probs length)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # 1. DPO Component
        chosen_logratios = policy_log_probs[:batch_size] - ref_log_probs[:batch_size]
        rejected_logratios = policy_log_probs[batch_size:] - ref_log_probs[batch_size:]
        logits = self.beta * (chosen_logratios - rejected_logratios)

        if self.loss_type == "sigmoid":
            dpo_loss = -F.logsigmoid(logits).mean()
        elif self.loss_type == "hinge":
            dpo_loss = torch.relu(1 - logits).mean()
        else:  # ipo
            dpo_loss = ((logits - 1 / (2 * self.beta)) ** 2).mean()

        # 2. Exploration Bonus (Only for CHOSEN responses)
        # CFN Forward (Evaluation Mode)
        cfn_model.eval()
        with torch.no_grad():
            cfn_out = cfn_model(chosen_hidden_states)
            # Bonus = ||f(s)|| / sqrt(d)
            norm = torch.norm(cfn_out, p=2, dim=1)
            # Import math strictly inside method if needed, but it's usually top-level
            # Assuming math is imported or we can use N**0.5
            bonus = norm / (self.cfn_output_dim**0.5)

        # Exploration Loss term (maximize bonus => minimize negative bonus)
        # We weigh it by the policy probability of the chosen response to encourage
        # high probability on unvisited (high bonus) states.
        exploration_loss = -(self.alpha * bonus * policy_log_probs[:batch_size]).mean()

        total_loss = dpo_loss + exploration_loss

        # Metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "loss": total_loss.detach(),
            "loss/dpo": dpo_loss.detach(),
            "loss/exploration": exploration_loss.detach(),
            "rewards/chosen": chosen_rewards.mean().detach(),
            "rewards/rejected": rejected_rewards.mean().detach(),
            "rewards/accuracies": accuracy.detach(),
            "exploration/bonus": bonus.mean().detach(),
        }

        return total_loss, metrics


class ReinforceLoss(nn.Module):
    """
    REINFORCE (Vanilla Policy Gradient) loss.
    Loss = -log(pi(a|s)) * A_t
    """

    def forward(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy loss.
        """
        loss = -(log_probs * advantages)

        if action_mask is not None:
            loss = loss * action_mask
            return loss.sum() / action_mask.sum().clamp(min=1.0)

        return loss.mean()


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) loss.
    J = E[ min(r*A, clip(r)*A) - beta * D_KL ]
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        beta: float = 0.04,
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta = beta

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        kl_div: torch.Tensor,  # Expected to be computed per-token or compatible shape
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute GRPO loss.
        We want to maximize J, so minimize Loss = -J.
        """
        # Ratio = pi / pi_old
        ratio = torch.exp(log_probs - old_log_probs)

        # Surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        surrogate = torch.min(surr1, surr2)

        # Loss per token = -surrogate + beta * KL
        # Note: KL passed in should be positive D_KL(pi_ref || pi) or similar penalty
        loss_per_token = -surrogate + self.beta * kl_div

        if action_mask is not None:
            total_loss = (loss_per_token * action_mask).sum() / action_mask.sum().clamp(min=1.0)

            # Metrics
            with torch.no_grad():
                clip_frac = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
                clip_frac = (clip_frac.float() * action_mask).sum() / action_mask.sum().clamp(min=1.0)
        else:
            total_loss = loss_per_token.mean()
            with torch.no_grad():
                clip_frac = ((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)).float().mean()

        metrics = {
            "grpo_loss": total_loss.detach(),
            "clip_frac": clip_frac,
        }

        return total_loss, metrics


class VAPOLoss(nn.Module):
    """
    VAPO Policy Loss with Asymmetric Clipping.
    """

    def __init__(
        self,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        lm_loss_coeff: float = 0.1,
    ):
        super().__init__()
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.lm_loss_coeff = lm_loss_coeff

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        lm_loss: torch.Tensor | float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute VAPO policy loss.
        """
        ratio = torch.exp(log_probs - old_log_probs)

        # Asymmetric Clipping
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)

        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages

        # PPO Objective
        policy_loss = -torch.min(surr1, surr2)

        # Mask and aggregate
        num_valid = action_mask.sum().clamp(min=1.0)
        policy_loss_mean = (policy_loss * action_mask).sum() / num_valid

        # Add LM Loss
        total_loss = policy_loss_mean + self.lm_loss_coeff * lm_loss

        with torch.no_grad():
            clip_frac = (ratio < 1.0 - self.epsilon_low) | (ratio > 1.0 + self.epsilon_high)
            clip_frac = (clip_frac.float() * action_mask).sum() / num_valid

        metrics = {
            "policy_loss": policy_loss_mean.detach(),
            "total_loss": total_loss.detach(),
            "clip_frac": clip_frac,
        }

        return total_loss, metrics


class IPOLoss(nn.Module):
    """
    Identity Preference Optimization (IPO) loss.

    optimizes the policy to maximize the log-odds gap towards a target margin 1/(2*tau).
    """

    def __init__(
        self,
        tau: float = 0.1,
        loss_type: str = "ipo",
    ):
        """
        Initialize IPO loss.

        Args:
            tau: Regularization parameter
            loss_type: 'ipo' or 'ipo_hinge'
        """
        super().__init__()
        self.tau = tau
        self.loss_type = loss_type

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute IPO loss.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Logits = log(pi/ref) gap
        logits = chosen_logratios - rejected_logratios

        # Compute Loss
        if self.loss_type == "ipo":
            # (logits - 1/(2*tau))^2
            losses = (logits - 1 / (2 * self.tau)) ** 2
        elif self.loss_type == "ipo_hinge":
            # ReLU(1/(2*tau) - logits)^2
            losses = torch.relu(1 / (2 * self.tau) - logits) ** 2
        else:
            raise ValueError(f"Unknown IPO loss type: {self.loss_type}")

        loss = losses.mean()

        # Metrics
        with torch.no_grad():
            # Use tau as proxy for beta-scaled rewards if helpful or raw
            # IPO paper doesn't explicitly scale rewards by beta/tau for reporting usually,
            # but usually we want to see margin.
            accuracy = (chosen_logratios > rejected_logratios).float().mean()
            kl_dist = 0.5 * (chosen_logratios + rejected_logratios).mean()

        metrics = {
            "ipo_loss": loss.detach(),
            "ipo_accuracy": accuracy.detach(),
            "kl_approx": kl_dist.detach(),
            "log_odds_margin": logits.mean().detach(),
        }

        return loss, metrics


class DAPOLoss(nn.Module):
    """
    DAPO Policy Loss with Asymmetric Clipping.
    """

    def __init__(
        self,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
    ):
        super().__init__()
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute DAPO policy loss.
        """
        ratio = torch.exp(log_probs - old_log_probs)

        # Asymmetric Clipping
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)

        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages

        # PPO Objective
        policy_loss = -torch.min(surr1, surr2)

        # Mask and aggregate by total tokens
        num_valid = action_mask.sum().clamp(min=1.0)
        total_loss = (policy_loss * action_mask).sum() / num_valid

        with torch.no_grad():
            clip_frac = (ratio < 1.0 - self.epsilon_low) | (ratio > 1.0 + self.epsilon_high)
            clip_frac = (clip_frac.float() * action_mask).sum() / num_valid

        metrics = {
            "dapo_loss": total_loss.detach(),
            "clip_frac": clip_frac,
        }

        return total_loss, metrics
