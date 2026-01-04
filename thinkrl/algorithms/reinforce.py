from dataclasses import asdict, dataclass
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Optimizer
from thinkrl.algorithms.base import BaseRLHFAlgorithm
from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)

@dataclass
class REINFORCEConfig:
    """
    Configuration for REINFORCE Algorithm.
    
    Attributes:
        learning_rate: Learning rate for optimizer
        gamma: Discount factor for rewards (0 < gamma <= 1)
        entropy_coeff: Coefficient for entropy regularization
        use_baseline: Whether to use a value baseline to reduce variance
        normalize_returns: Whether to normalize returns
        clip_grad_norm: Maximum gradient norm for clipping
    """
    
    learning_rate: float = 1e-3
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    use_baseline: bool = True
    normalize_returns: bool = True
    clip_grad_norm: float = 1.0
    
    use_replay: bool= False  # REINFORCE does not use replay by default
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    epochs_per_episode: int = 1
    importance_sampling: bool = False
    max_importance_weight: float = 10.0
    use_gae: bool = False
    gae_lambda: float = 0.95
    
    
    
    def __post_init__(self):
        """Validate configuration parameters."""
        
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert self.entropy_coeff >= 0, "entropy_coeff must be non-negative"
        assert self.clip_grad_norm > 0, "clip_grad_norm must be positive"
        assert self.epochs_per_episode > 0, "epochs_per_episode must be positive"
        
        if self.use_replay:
            assert self.replay_buffer_size > 0, "replay_buffer_size must be positive"
            assert self.replay_batch_size > 0, "replay_batch_size must be positive"
        
        if self.importance_sampling:
            assert self.max_importance_weight > 0, "max_importance_weight must be positive"
        if self.use_gae:
            assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"
            

class ReplayBuffer:
    """Experience replay buffer for sample-efficient REINFORCE"""
    
    def __init__(self,capacity:int):
        """Initialize Replay Buffer"""
        self.capacity = capacity
        self.buffer: list[dict] = []
        self.position = 0
        
    def push(
        self,
        state,
        action: int,
        reward: float,
        log_prob: float,
        done: bool,
        return_: float | None = None,
        advantage: float | None = None,
        value: float | None = None,
    ):
        
        """Store transition in the buffer"""
        
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "done": done,
            "return": return_,
            "advantage": advantage,
            "value": value,
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
        
    def sample(self, batch_size: int) -> dict:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "dones": [],
            "returns": [],
            "advantages": [],
            "values": [],
        }
        
        for idx in indices:
            transition = self.buffer[idx]
            batch["states"].append(transition["state"])
            batch["actions"].append(transition["action"])
            batch["rewards"].append(transition["reward"])
            batch["log_probs"].append(transition["log_prob"])
            batch["dones"].append(transition["done"])
            batch["returns"].append(transition["return"])
            batch["advantages"].append(transition["advantage"])
            batch["values"].append(transition["value"])
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
    
class REINFORCEMemory:
    """
    Memory buffer for REINFORCE episodes.

    Stores complete episodes of (state, action, reward, log_prob) tuples.
    Unlike PPO's mini-batch approach, REINFORCE processes full episodes.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def store_step(
        self,
        state: Any,
        action: int,
        reward: float,
        log_prob: float,
    ) -> None:
        """
        Store a single step in the current episode.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
        """
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        
    def get_episode(self) -> tuple[list,list,list,list]:
        """
        Get the stored episode data.

        Returns:
            Tuple of (states, actions, rewards, log_probs)
        """
        
        return self.states, self.actions, self.rewards, self.log_probs
    
    def clear(self) -> None:
        """clear all stored episode data"""
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def __len__(self) -> int:
        """Return the number of steps in the current episode"""
        return len(self.states)
        

class PolicyNetwork(nn.Module):
    """
    Policy network for REINFORCE
    
    A simple multi-layer perceptron that outputs action probabilities
    """
    
    
    def __init__(
        self,
        n_actions: int,
        input_dims: tuple,
        learning_rate: float = 1e-3,
        hidden_dims: int = 128,
    ):
    
        """
        Initialize policy network.
        
        Args:
            n_action: Number of possible actions
            input_dims: Dimensions of input state
            learning_rate: Learning rate for optimizer
            hidden_dims: Number of hidden units in the network
        """
        
        super().__init__()
        
        self.fc1 = nn.Linear(*input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, n_actions)
        self.activation = nn.ReLU()
        
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Forward pass through policy network.

        Args:
            state: Current state tensor of shape [batch_size, *input_dims]
            
        Returns:
            Categorical distribution over actions
        """
        
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        action_logits = self.fc3(x)
        
        return Categorical(logits=action_logits)
        

class ValueNetwork(nn.Module):
    """
    Value network (baseline) for REINFORCE
    
    Estimates state values to reduce gradient variance
    """
    
    
    def __init__(
        self,
        input_dims: tuple,
        learning_rate: float = 1e-3,
        hidden_dims: int = 128,
    ):
        """
        Initialize value network
        
        Args:
            input_dims: Tuple of input dimensions
            learning_rate: Learning rate for optimizer
            hidden_dims: Size of hidden layer
            
        
        """
        
        super().__init__()
        
        self.fc1 = nn.Linear(*input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.value_head = nn.Linear(hidden_dims, 1)
        self.activation = nn.ReLU()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network
        
        Args:
            state: Current state tensor of shape [batch_size, *input_dims]
        
        Returns:
            Value estimates of shape [batch_size, 1]
        """
        
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.value_head(x)
        return value 
    
class REINFORCEAlgorithm(BaseRLHFAlgorithm):
    """
    REINFORCE (Monte Carlo Policy Gradient) algorithm.

    REINFORCE is a foundational policy gradient method that:
    1. Collects complete episodes using the current policy
    2. Computes discounted returns for each timestep
    3. Updates policy to increase probability of actions with high returns
    4. Optionally uses a baseline (value function) to reduce variance
    """
    
    
    def __init__(
        self,
        policy_model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        config: REINFORCEConfig | None = None,
        n_actions: int | None = None,
        input_dims: tuple | None = None,
        **kwargs,
    ):
        
        """
        Initialize REINFORCE algorithm.

        Args:
            policy_model: The policy model (for LLM-based RL)
            ref_model: Reference model (not used in REINFORCE)
            optimizer: Optional optimizer
            config: REINFORCE configuration
            n_actions: Number of actions (for simple environments)
            input_dims: Input dimensions (for simple environments)
            **kwargs: Additional arguments for BaseRLHFAlgorithm
        """
        
        
        config = config or REINFORCEConfig()
        
        # Create a dummy policy model if none provided and using simple networks
        if policy_model is None and n_actions and input_dims:
            policy_model = PolicyNetwork(
                n_actions=n_actions,
                input_dims=input_dims,
                learning_rate=config.learning_rate,
            )
        
        super().__init__(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
            kl_coeff=0.0,
            gamma=config.gamma,
            clip_grad_norm=config.clip_grad_norm,
            **kwargs,
        )
        
        self.config = config
        self.memory = REINFORCEMemory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        if config.use_replay:
            self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
        else:
            self.replay_buffer = None
            
        # Initialize simple policy/value networks if using basic RL
        if n_actions and input_dims:
            self.policy_net = PolicyNetwork(
                n_actions=n_actions,
                input_dims=input_dims,
                learning_rate=config.learning_rate,
            )
            if config.use_baseline:
                self.value_net = ValueNetwork(
                    input_dims=input_dims,
                    learning_rate=config.learning_rate,
                )
            else:
                self.value_net = None
            self.use_simple_networks = True
        else:
            self.policy_net = None
            self.value_net = None
            self.use_simple_networks = False
        
        logger.info(
            f"Initialized REINFORCE (lr={config.learning_rate}, gamma={config.gamma}, "
            f"baseline={config.use_baseline}, entropy_coeff={config.entropy_coeff})"
        )
    
    
    def compute_gae(
    self,
    rewards: list[float],
    values: torch.Tensor,
    dones: list[bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: Value estimates
            dones: Done flags

        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to tensors if needed
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float32)
        
        # Ensure values is 1D
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if values.dim() > 1:
            values = values.squeeze()
            
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1].item() if values[t + 1].dim() == 0 else values[t + 1]
            
            reward_t = rewards[t].item() if rewards[t].dim() == 0 else rewards[t]
            value_t = values[t].item() if values[t].dim() == 0 else values[t]
            done_t = dones[t].item() if dones[t].dim() == 0 else dones[t]
            
            delta = reward_t + self.config.gamma * next_value * (1 - done_t) - value_t
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - done_t) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values.to(self.device)
        
        return advantages, returns
    
    
    def compute_importance_weights(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance sampling weights.

        Args:
            old_log_probs: Log probabilities from behavior policy
            new_log_probs: Log probabilities from current policy

        Returns:
            Clipped importance weights
    """
        # ρ = π(a|s) / μ(a|s) = exp(log π(a|s) - log μ(a|s))
        importance_weights = torch.exp(new_log_probs - old_log_probs)
        
        # clip to prevent high variance
        importance_weights = torch.clamp(
            importance_weights,
            1.0/self.config.max_importance_weight,
            self.config.max_importance_weight,
            
        )
        
        return importance_weights
    
    def choose_action(self, observation: np.ndarray | list) -> tuple[int, float]:
        """
        Sample action from policy.

        Args:
            observation: Current state observation

        Returns:
            Tuple of (action, log_probability)
        """
        if self.use_simple_networks:
            assert self.policy_net is not None, "Policy network not initialized"
            
            # Convert to numpy array first if it's a list to avoid slow tensor creation
            if isinstance(observation, list):
                observation = np.array(observation)
            
            # Convert to tensor, then add batch dimension
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(
                self.policy_net.fc1.weight.device
            )
            
            dist = self.policy_net(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            action = action.item()
            return action, log_prob
        else:
            return self._choose_action_from_policy_model(observation)
    
    def _choose_action_from_policy_model(
        self, observation: Any
    ) -> tuple[int, float]:
        """
        Choose action using the main policy model (for LLM case)
        
        Args:
            observation: Current observation
        Returns:
            Tuple of (action, log_probability)
        Raises:
            NotImplementedError: Must be implemented for specific policy models
        """
        raise NotImplementedError(
            "Implement action selection for your specific policy model"
        )
    
    def store_step(
        self,
        state: Any,
        action: int,
        reward: float,
        log_prob: float,
    ) -> None:
        """
        Store a step in the current episode.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
        """
        self.memory.store_step(state, action, reward, log_prob)
    
    def compute_returns(self, rewards: list, normalize: bool = True) -> torch.Tensor:
        """
        Compute discounted returns for an episode.
        
        Use the formula G_t = Σ(γ^k * r_{t+k}) for k=0 to T-t
        
        Args:
            rewards: List of rewards for the episode
            normalize: Whether to normalize returns
        
        Returns:
            Tensor of discounted returns [T]
        
        Time Complexity: O(T) where T is episode length
        """
        returns = []
        G = 0.0
        
        # Compute returns backwards (more efficient)
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float)
        
        # Normalize returns to reduce variance
        if normalize and len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )
        
        return returns_tensor
    
    def compute_loss(
        self,
        states: list,
        actions: list,
        returns: torch.Tensor,
        old_log_probs: list[float],
        advantages: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute REINFORCE loss for an episode.

        Loss = -Σ(log π(a_t|s_t) * G_t)
        Where G_t is the return (optionally with baseline subtracted)

        Args:
            states: List of states
            actions: List of actions
            returns: Discounted returns
            old_log_probs: Log probabilities of actions
            advantages: Optional precomputed advantages

        Returns:
            Dictionary containing loss and metrics
        """
        # Convert to tensors
        device = self.policy_net.device if self.use_simple_networks else self.policy_model.device
        
        # Convert list of numpy arrays to single numpy array before tensor creation
        if isinstance(states, list):
            states = np.array(states)
        
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        returns = returns.to(device)

        # Get current policy distribution
        if self.use_simple_networks:
            dist = self.policy_net(states_tensor)
        else:
            dist = self.policy_model(states_tensor)

        # Get current log probabilities
        new_log_probs = dist.log_prob(actions_tensor)

        # Compute advantages if not provided
        if advantages is None:
            if self.config.use_baseline:
                if self.use_simple_networks:
                    values = self.value_net(states_tensor).squeeze()
                else:
                    values = self.value_model(states_tensor).squeeze()
                advantages = returns - values.detach()
            else:
                advantages = returns

        # Compute importance weights if using importance sampling
        if self.config.importance_sampling:
            importance_weights = self.compute_importance_weights(
                old_log_probs_tensor, new_log_probs
            )
            policy_loss = -(importance_weights * new_log_probs * advantages).mean()
        else:
            policy_loss = -(new_log_probs * advantages.detach()).mean()

        # Compute value loss if using baseline
        if self.config.use_baseline:
            if self.use_simple_networks:
                values = self.value_net(states_tensor).squeeze()
            else:
                values = self.value_model(states_tensor).squeeze()
            # Ensure both values and returns have the same shape
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            value_loss = F.mse_loss(values, returns)
        else:
            value_loss = torch.tensor(0.0, device=device)

        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.entropy_coeff * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss
        if self.config.use_baseline:
            total_loss = total_loss + value_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach() if self.config.use_baseline else value_loss,
            "entropy": entropy.detach(),
            "mean_return": returns.mean().detach(),
        }
    
    def _compute_loss_simple_networks(
        self,
        states: list,
        actions: list,
        returns: torch.Tensor,
        log_probs: list,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss using simple policy/value networks.

        Args:
            states: List of states
            actions: List of actions
            returns: Discounted returns
            log_probs: Log probabilities

        Returns:
            Dictionary with loss and metrics
        """
        assert self.policy_net is not None, "Policy network not initialized"
        device = self.policy_net.device
        
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        returns = returns.to(device)
        
        # Compute new log probabilities
        dist = self.policy_net(states_tensor)
        new_log_probs = dist.log_prob(actions_tensor)
        
        # Compute advantages (returns or returns - baseline)
        if self.config.use_baseline and self.value_net is not None:
            values = self.value_net(states_tensor).squeeze()
            advantages = returns - values.detach()
            value_loss = F.mse_loss(values, returns)
        else:
            advantages = returns
            value_loss = torch.tensor(0.0, device=device)
        
        # Policy gradient loss -E[log π * A]
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Entropy bonus to encourage exploration
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.entropy_coeff * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        if self.config.use_baseline:
            total_loss = total_loss + value_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": (
                value_loss.detach() if self.config.use_baseline else value_loss
            ),
            "entropy": entropy.detach(),
            "mean_return": returns.mean().detach(),
        }
            
    def _compute_loss_policy_model(
        self,
        states: list,
        actions: list,
        returns: torch.Tensor,
        log_probs: list,
        
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss using unified policy model (for LLM).

        Args:
            states: List of input sequences
            actions: List of action tokens
            returns: Discounted returns
            log_probs: Old log probabilities

        Returns:
            Dictionary with loss and metrics
        """
        
        raise NotImplementedError(
            "Implement loss computation for your specific policy model"
        )
        
        
    def training_step(
        self,
        states: list,
        actions: list,
        returns: torch.Tensor,
        old_log_probs: list[float],
        advantages: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Execute single training step
        
        Args:
            states: List of states
            actions: List of actions
            returns: Discounted returns
            old_log_probs: Log probabilities of actions
            advantages: Optional precomputed advantages
        
        Returns:
            Dictionary of training metrics
        """
        
        # Zero Gradient
        if self.use_simple_networks:
            assert self.policy_net is not None, "Policy network not initialized"
            self.policy_net.optimizer.zero_grad()
            if self.value_net is not None:
                self.value_net.optimizer.zero_grad()
        
        
        else:
            assert self.optimizer is not None, "Optimizer not initialized"
            self.optimizer.zero_grad()
            
        # Compute loss
        loss_dict = self.compute_loss(states, actions, returns, old_log_probs, advantages)
        loss = loss_dict["loss"]
        
        # Backward loss
        loss.backward()
        
        # Gradient clipping
        grad_norm: float = 0.0
        if self.use_simple_networks:
            assert self.policy_net is not None, "Policy network not initialized"
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.clip_grad_norm,
            )
            
            grad_norm = float(grad_norm_tensor.item())
            if self.value_net is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.config.clip_grad_norm,
                )
        else:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.clip_grad_norm,
            )
            grad_norm = float(grad_norm_tensor.item())
            
        if self.use_simple_networks:
            assert self.policy_net is not None, "Policy network not initialized"
            self.policy_net.optimizer.step()
            if self.value_net is not None:
                self.value_net.optimizer.step()
        else:
            assert self.optimizer is not None, "Optimizer not initialized"
            self.optimizer.step()
            
        metrics: dict[str,float] = {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_dict.items()
        }
        metrics["grad_norm"] = grad_norm
        
        return metrics
    
    
    def learn(self) -> dict[str, float]:
        """
        Learn from collected episode with sample efficiency improvements.
        
        Process:
        1. Retrieve episode data from memory
        2. Compute discounted returns or GAE advantages
        3. Optionally store in replay buffer
        4. Perform multiple epochs of training
        5. Clear episode memory
        
        Returns:
            Dictionary of training metrics
            
        Raises:
            ValueError: If memory is empty
        """
        
        if len(self.memory) == 0:
            raise ValueError("Cannot learn from empty episode")
        
        # Get episode data
        states, actions, rewards, log_probs = self.memory.get_episode()
        
        # Compute returns or advantages
        if self.config.use_gae and self.config.use_baseline:
            # Compute GAE
            device = self.policy_net.device if self.use_simple_networks else self.policy_model.device
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            if self.use_simple_networks:
                values = self.value_net(states_tensor).squeeze().detach()
            else:
                values = self.value_model(states_tensor).squeeze().detach()
            
            dones = [False] * (len(rewards) - 1) + [True]
            advantages, returns = self.compute_gae(rewards, values, dones)
        else:
            # Standard return computation
            returns = self.compute_returns(rewards, self.config.normalize_returns)
            advantages = None
        
        # Store in replay buffer if enabled
        if self.config.use_replay:
            for i in range(len(states)):
                self.replay_buffer.push(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    log_prob=log_probs[i],
                    done=(i == len(states) - 1),
                    return_=returns[i].item() if advantages is None else None,
                    advantage=advantages[i].item() if advantages is not None else None,
                )
        
        # Multiple epochs of training
        all_metrics = []
        for epoch in range(self.config.epochs_per_episode):
            if self.config.use_replay and len(self.replay_buffer) >= self.config.replay_batch_size:
                # Sample from replay buffer
                batch = self.replay_buffer.sample(self.config.replay_batch_size)
                device = self.policy_net.device if self.use_simple_networks else self.policy_model.device
                batch_returns = torch.tensor(
                    batch["returns"] if batch["returns"][0] is not None else batch["advantages"],
                    dtype=torch.float32
                ).to(device)
                batch_advantages = torch.tensor(
                    batch["advantages"], dtype=torch.float32
                ).to(device) if batch["advantages"][0] is not None else None
                
                metrics = self.training_step(
                    batch["states"],
                    batch["actions"],
                    batch_returns,
                    batch["log_probs"],
                    batch_advantages,
                )
            else:
                # Use current episode
                metrics = self.training_step(states, actions, returns, log_probs, advantages)
            
            all_metrics.append(metrics)
        
        # Average metrics across epochs
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Add episode statistics
        avg_metrics["episode_length"] = len(states)
        avg_metrics["episode_return"] = sum(rewards)
        
        # Clear episode memory
        self.memory.clear()
        
        return avg_metrics
    
    def state_dict(self) -> dict[str, Any]:
        """
        Get algorithm state for checkpointing.

        Returns:
            State dictionary
        """
        
        state: dict[str, Any] = {
            "config": asdict(self.config),
        }
        
        if self.use_simple_networks:
            assert self.policy_net is not None, "policy network not initialized"
            state["policy_net"] = self.policy_net.state_dict()
            if self.value_net is not None:
                state["value_net"] = self.value_net.state_dict()
        else:
            state["policy_model"] = self.policy_model.state_dict()
            if self.optimizer is not None:
                state["optimizer"] = self.optimizer.state_dict()
                
        return state
    

def create_reinforce(
    n_actions: int | None = None,
    input_dims: tuple[int, ...] | None = None,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    use_baseline: bool = True,
    entropy_coeff: float = 0.01,
    policy_model: nn.Module | None = None,
    use_replay: bool = False,
    replay_buffer_size: int = 10000,
    replay_batch_size: int = 32,
    epochs_per_episode: int = 1,
    importance_sampling: bool = False,
    max_importance_weight: float = 10.0,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
) -> REINFORCEAlgorithm:
    """
    Factory function to create REINFORCEAlgorithm instance.

    Args:
        policy_model: The policy model (optional, for LLM)
        optimizer: Optional optimizer
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        use_baseline: Whether to use value baseline
        n_actions: Number of actions (for simple RL)
        input_dims: Input dimensions (for simple RL)
        **kwargs: Additional arguments for REINFORCEConfig

    Returns:
        Configured REINFORCEAlgorithm instance
    """
    
    config = REINFORCEConfig(
        learning_rate=learning_rate,
        gamma=gamma,
        use_baseline=use_baseline,
        entropy_coeff=entropy_coeff,
        use_replay=use_replay,
        replay_buffer_size=replay_buffer_size,
        replay_batch_size=replay_batch_size,
        epochs_per_episode=epochs_per_episode,
        importance_sampling=importance_sampling,
        max_importance_weight=max_importance_weight,
        use_gae=use_gae,
        gae_lambda=gae_lambda,
    )
    
    # Create the algorithm instance first to get the policy_model
    algorithm = REINFORCEAlgorithm(
        policy_model=policy_model,
        optimizer=None,  # Will be set after policy_model is created
        config=config,
        n_actions=n_actions,
        input_dims=input_dims,
    )
    
    # Now create the optimizer with the policy_model from the algorithm
    optimizer = torch.optim.Adam(
        algorithm.policy_model.parameters(), lr=learning_rate
    )
    algorithm.optimizer = optimizer
    
    return algorithm
        
        
__all__ = [
    "REINFORCEAlgorithm",
    "REINFORCEConfig",
    "REINFORCEMemory",
    "PolicyNetwork",
    "ValueNetwork",
    "create_reinforce",
]