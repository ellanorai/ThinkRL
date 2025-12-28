from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.distributions.categorical import Categorical
from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Multi epoch training
    n_epochs: int = 10
    batch_size: int = 64

    # Training stability
    clip_grad_norm: float = 1
    normalize_advantages: bool = True

    # value function
    use_separate_value_network: bool = False

    def __post_init__(self):
        assert self.policy_clip > 0
        assert self.gamma > 0 and self.gamma <= 1
        assert self.gae_lambda >= 0 and self.gae_lambda <= 1
        assert self.n_epochs >= 1
        assert self.batch_size > 0


class PPOMemory:
    """
    Memory buffer for PPO rollouts.

    Stores states, actions, rewards, values, and log probabilities
    for computing advantage and training
    """

    def __init__(self, batch_size: int):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        """
        Generate random mini-batches for training
        Returns:
            Tuple of (states, actions, probs, vals, rewards, dones, batches)
        """

        n_states = len(self.states)
        if n_states == 0:
            raise ValueError("Cannot generate batches from empty memory buffer")

        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        indices = indices[torch.randperm(n_states)]

        batches = [indices[i : i + self.batch_size] for i in batch_start]

        # Convert to numpy first to handle mixed types efficiently
        states_array = np.array(self.states, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.int64)
        probs_array = np.array(self.probs, dtype=np.float32)
        vals_array = np.array(self.vals, dtype=np.float32)
        rewards_array = np.array(self.rewards, dtype=np.float32)
        dones_array = np.array(self.dones, dtype=np.float32)

        return (
            torch.from_numpy(states_array),
            torch.from_numpy(actions_array),
            torch.from_numpy(probs_array),
            torch.from_numpy(vals_array),
            torch.from_numpy(rewards_array),
            torch.from_numpy(dones_array),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        """Store a single transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear all stored memory"""
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    """
    Actor network for policy learning
    Outputs action probabilities given state
    """

    def __init__(
        self,
        n_actions: int,
        input_dims: tuple,
        alpha: float,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
    ):

        super(ActorNetwork, self).__init__()

        self.input_layer = nn.Linear(*input_dims, fc1_dims)
        self.hidden_layer = nn.Linear(fc1_dims, fc2_dims)
        self.output_layer = nn.Linear(fc2_dims, n_actions)
        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=-1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through actor network

        Args:
            State: Current state tensor

        Returns:
            Categorical distribution over actions
        """

        x = self.activation(self.input_layer(state))
        x = self.activation(self.hidden_layer(x))
        action_probs = self.output_activation(self.output_layer(x))

        return Categorical(action_probs)


class CriticNetwork(nn.Module):
    """
    Value network for state evaluation

    Estimates the expected cumulative reward from a given state using
    a multi layer perceptron architecture.
    """

    def __init__(
        self, input_dims: tuple, alpha: float, fc1_dims: int = 256, fc2_dims: int = 256
    ):
        super(CriticNetwork, self).__init__()

        self.input_layer = nn.Linear(*input_dims, fc1_dims)
        self.hidden_layer = nn.Linear(fc1_dims, fc2_dims)
        self.value_head = nn.Linear(fc2_dims, 1)

        self.activation = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through critic network

        Args:
            State: Current state tensor

        Returns:
            Value estimate for the state
        """

        x = self.activation(self.input_layer(state))
        x = self.activation(self.hidden_layer(x))
        value = self.value_head(x)
        return value


class PPOAlgorithm(BaseRLHFAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    Uses clipped surrogate objective with actor-critic architecture.
    GAE for advantage estimates, and multi-epoch mini-batch training
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        config: Optional[PPOConfig] = None,
        n_actions: Optional[int] = None,
        input_dims: Optional[tuple] = None,
        **kwargs,
    ):
        config = config or PPOConfig()

        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=0.0,
            gamma=config.gamma,
            lambda_=config.gae_lambda,
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )

        self.config = config
        self.memory = PPOMemory(config.batch_size)

        # Initialize actor and critic networks if using separate networks
        if config.use_separate_value_network and n_actions and input_dims:
            self.actor = ActorNetwork(
                n_actions=n_actions, input_dims=input_dims, alpha=config.learning_rate
            )
            self.critic = CriticNetwork(
                input_dims=input_dims, alpha=config.learning_rate
            )
            self.use_separate_value_network = True
        else:
            self.actor = None
            self.critic = None
            self.use_separate_value_network = False

        logger.info(
            f"Initialized PPO (clip={config.policy_clip}, entropy={config.entropy_coeff}, "
            f"value_coeff={config.value_coeff}, n_epochs={config.n_epochs})"
        )

    def remember(self, state, action, probs, vals, reward, done):
        """Store transition in memory."""
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        """
        Sample action from policy

        Args:
            observation: Current state observation
        Returns:
            Tuple of (action, log_prob, value)
        """

        if self.use_separate_value_network:
            state = (
                torch.tensor(observation, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.actor.device)
            )
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()
            return action, probs, value
        else:

            return self._choose_action_from_policy_model(observation)

    def _choose_action_from_policy_model(self, observation):
        """Choose action using the main policy model (for LLM case)."""
        raise NotImplementedError(
            "Implement action selection for your specific policy model"
        )

    def compute_gae_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: Rewards tensor[T]
            values: value estimates [T]
            dones: Done flags [T]

        Returns:
            advantages: Computed advantages [T]
        """

        cfg = self.config
        device = rewards.device

        advantage = torch.zeros(len(rewards), dtype=torch.float).to(device)
        gae = 0

        # Backward iteration for O(n) complexity
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
            advantage[t] = gae
        return advantage

    def get_log_probs(
        self,
        outputs: Union[dict[str, torch.Tensor], torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per token log probabilities.

        Args:
            outputs: Model outputs (dict with 'logits' or tensor of logits)
            labels: Target token IDs [B,S], -100 for masked positions

        Returns:
            log_probs: [B,S] with 0.0at masked positions
        """

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)

        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0

        token_log_probs = log_probs.gather(
            dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs[shift_labels == -100] = 0.0

        padding = torch.zeros(
            token_log_probs.size(0),
            1,
            device=token_log_probs.device,
            dtype=token_log_probs.dtype,
        )
        return torch.cat([token_log_probs, padding], dim=1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute PPO loss with clipped surrogate objective.

        Args:
            batch: Training batch containing states, actions, old_probs, advantages, returns

        Returns:
            Dictionary with loss components
        """

        if self.use_separate_value_network:
            return self._compute_loss_separate_networks(batch)
        else:
            return self._compute_loss_unified_model(batch)

    def _compute_loss_separate_networks(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss using separate actor-critic networks"""

        cfg = self.config
        device = self.actor.device

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        old_probs = batch["old_probs"].to(device)
        advantages = batch["advantages"].to(device)
        returns = batch["returns"].to(device)

        dist = self.actor(states)
        critic_value = self.critic(states)
        critic_value = torch.squeeze(critic_value)

        new_probs = dist.log_prob(actions)
        prob_ratio = torch.exp(new_probs - old_probs)

        weighted_probs = advantages * prob_ratio
        weighted_clipped_probs = (
            torch.clamp(prob_ratio, 1 - cfg.policy_clip, 1 + cfg.policy_clip)
            * advantages
        )

        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()

        entropy = dist.entropy().mean()
        entropy_loss = -cfg.entropy_coeff * entropy

        total_loss = actor_loss + cfg.value_coeff * critic_loss + entropy_loss
        return {
            "loss": total_loss,
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "entropy": entropy.detach(),
        }

    def _compute_loss_unified_model(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss using unified policy model (for LLM or simple state cases)."""

        cfg = self.config

        # Handle simple state-action case (non-LLM)
        if "states" in batch and "input_ids" not in batch:
            states = batch["states"]
            device = states.device
            actions = batch["actions"]
            old_probs = batch["old_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # Forward pass through policy model
            # Check if model expects integer input (has embedding layer)
            if hasattr(self.policy_model, "embedding") and states.dtype == torch.float:
                # Convert float states to long for embedding models
                # Ensure values are within vocab range
                vocab_size = self.policy_model.embedding.num_embeddings
                states_input = states.long() % vocab_size  # Wrap to valid range
                outputs = self.policy_model(input_ids=states_input)
            else:
                outputs = self.policy_model(states)

            # Compute action probabilities
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs

            # Handle sequence dimension if present: take last token
            if logits.dim() == 3:  # [batch, seq, vocab]
                logits_for_action = logits[:, -1, :]  # Take last token
            else:  # [batch, vocab]
                logits_for_action = logits

            dist = Categorical(logits=logits_for_action)
            new_log_probs = dist.log_prob(actions)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - cfg.policy_clip, 1 + cfg.policy_clip)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (estimate value from logits)
            values = self._extract_values(logits).squeeze(-1)
            if values.dim() > 1:
                values = values.mean(dim=1)  # Average over sequence dimension if needed
            value_loss = F.smooth_l1_loss(values, returns)

            # Entropy bonus
            entropy = dist.entropy().mean()
            entropy_loss = -cfg.entropy_coeff * entropy

            total_loss = policy_loss + cfg.value_coeff * value_loss + entropy_loss

            return {
                "loss": total_loss,
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "entropy_loss": entropy_loss.detach(),
                "entropy": entropy.detach(),
            }

        # Handle LLM case with input_ids
        device = batch["input_ids"].device

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        log_probs = self.get_log_probs(outputs, labels)

        ratio = torch.exp(log_probs - old_log_probs)

        token_mask = labels != -100

        adv_expanded = advantages.unsqueeze(1).expand_as(ratio)
        surr_unclipped = ratio * adv_expanded
        surr_clipped = (
            torch.clamp(ratio, 1 - cfg.policy_clip, 1 + cfg.policy_clip) * adv_expanded
        )

        surr_min = torch.min(surr_unclipped, surr_clipped)
        num_tokens = token_mask.sum().float().clamp(min=1.0)
        policy_loss = -surr_min[token_mask].sum() / num_tokens

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        values = self._extract_values(logits).squeeze(-1)  # [B, S]
        # Average values across sequence to match returns shape [B]
        values_mean = (values * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(
            min=1.0
        )
        value_loss = F.smooth_l1_loss(values_mean, returns)

        entropy_loss = torch.tensor(0.0, device=device)
        if cfg.entropy_coeff > 0:
            entropy = self._compute_entropy(logits, token_mask)
            entropy_loss = -cfg.entropy_coeff * entropy

        total_loss = policy_loss + cfg.value_coeff * value_loss + entropy_loss

        with torch.no_grad():
            metrics = self._compute_diagnostics(
                ratio=ratio, advantages=advantages, token_mask=token_mask
            )

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            **metrics,
        }

    def _extract_values(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract value estimates from logits using proper value head projection"""
        # Use mean pooling over vocabulary dimension as value estimate
        # This is more stable than max and represents expected value better
        probs = F.softmax(logits, dim=-1)
        values = (probs * torch.arange(logits.size(-1), device=logits.device)).sum(
            dim=-1
        )
        # Normalize by vocabulary size for stability
        values = values / logits.size(-1)
        return values.unsqueeze(-1)

    def _compute_entropy(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean entropy over valid tokens."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return (entropy * mask).sum() / mask.sum().clamp(min=1.0)

    def _compute_diagnostics(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training diagnostics"""

        cfg = self.config

        clipped = (ratio < (1 - cfg.policy_clip)) | (ratio > (1 + cfg.policy_clip))
        valid_tokens = token_mask.sum().float().clamp(min=1.0)
        clip_frac = clipped[token_mask].float().sum() / valid_tokens

        ratio_masked = ratio[token_mask]

        return {
            "clip_frac": clip_frac,
            "ratio_mean": ratio_masked.mean(),
            "ratio_std": (
                ratio_masked.std() if ratio_masked.numel() > 1 else torch.tensor(0.0)
            ),
            "ratio_max": ratio_masked.max(),
            "ratio_min": ratio_masked.min(),
        }

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Execute single training step

        Args:
            batch: training batch

        Returns:
            Dictionary of metrics

        """

        if self.use_separate_value_network:
            self.actor.train()
            self.critic.train()
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
        else:
            self.policy_model.train()
            self.optimizer.zero_grad()

        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]

        loss.backward()

        if self.use_separate_value_network:
            # Apply gradient clipping to both networks
            _ = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.clip_grad_norm
            )
            _ = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.clip_grad_norm
            )
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.clip_grad_norm
            )
            self.optimizer.step()

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        if not self.use_separate_value_network:
            metrics["grad_norm"] = grad_norm.item()

        return metrics

    def learn(self):
        """
        Multi-epoch training using stored rollouts,

        This implements the core PPO learning loop with mini-batch updates.
        """

        all_metrics = []

        for epoch in range(self.config.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                val_arr,
                reward_arr,
                done_arr,
                batches,
            ) = self.memory.generate_batches()
            if self.use_separate_value_network:
                device = self.actor.device
            else:
                device = next(self.policy_model.parameters()).device

            values = val_arr.to(device)
            reward_arr = reward_arr.to(device)
            dones = done_arr.to(device)

            advantages = self.compute_gae_advantages(reward_arr, values, dones)
            returns = advantages + values

            for batch_indices in batches:
                batch_indices = batch_indices.to(device)

                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Normalize advantages per batch if enabled
                if self.config.normalize_advantages:
                    adv_mean = batch_advantages.mean()
                    adv_std = batch_advantages.std()
                    # Ensure std is not too small to avoid division issues
                    if adv_std < 1e-8:
                        adv_std = torch.tensor(1.0, device=batch_advantages.device)
                    batch_advantages = (batch_advantages - adv_mean) / (adv_std + 1e-8)

                batch = {
                    "states": state_arr[batch_indices].to(device),
                    "actions": action_arr[batch_indices].to(device),
                    "old_probs": old_prob_arr[batch_indices].to(device),
                    "advantages": batch_advantages,
                    "returns": batch_returns,
                }

                metrics = self.training_step(batch)
                metrics["epoch"] = epoch
                all_metrics.append(metrics)

        self.memory.clear_memory()
        return all_metrics

    def state_dict(self) -> dict:
        """Override base state_dict to handle PPOConfig dataclass."""
        from dataclasses import asdict

        state = {
            "policy_model": self.policy_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "learning_rate": self.learning_rate,
                "kl_coeff": self.kl_coeff,
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "normalize_rewards": self.normalize_rewards,
                "normalize_advantages": self.normalize_advantages,
                "clip_grad_norm": self.clip_grad_norm,
                **asdict(self.config),
            },
        }

        if self.ref_model is not None:
            state["ref_model"] = self.ref_model.state_dict()

        if self.use_separate_value_network:
            state["actor"] = self.actor.state_dict()
            state["critic"] = self.critic.state_dict()

        return state


def create_ppo(
    policy_model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 3e-4,
    policy_clip: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    **kwargs,
) -> PPOAlgorithm:
    """
    Factory function to create PPOAlgorithm instance.

    Args:
        policy_model: The policy model to be optimized
        optimizer: Optional optimizer, defaults to Adam
        learning_rate: Learning rate for optimizer
        policy_clip: Clipping parameter for PPO
        n_epochs: Number of training epochs per rollout
        batch_size: Mini-batch size for training
        **kwargs: Additional arguments for PPOAlgorithm

    Returns:
        Configured PPOAlgorithm instance
    """

    config = PPOConfig(
        learning_rate=learning_rate,
        policy_clip=policy_clip,
        n_epochs=n_epochs,
        batch_size=batch_size,
        **kwargs,
    )

    if optimizer is None:
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

    return PPOAlgorithm(
        policy_model=policy_model,
        optimizer=optimizer,
        config=config,
    )


__all__ = [
    "PPOAlgorithm",
    "PPOConfig",
    "PPOMemory",
    "ActorNetwork",
    "CriticNetwork",
    "create_ppo",
]
