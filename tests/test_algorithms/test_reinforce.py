import pytest
import torch
import torch.nn as nn
import numpy as np
from thinkrl.algorithms.reinforce import (
    REINFORCEAlgorithm,
    REINFORCEConfig,
    REINFORCEMemory,
    PolicyNetwork,
    ValueNetwork,
    create_reinforce,
    ReplayBuffer
)



# Configuration Tests

class TestREINFORCEConfig:
    """Test REINFORCE configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = REINFORCEConfig()
        
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.99
        assert config.entropy_coeff == 0.01
        assert config.use_baseline is True
        assert config.normalize_returns is True
        assert config.clip_grad_norm == 1.0
        assert config.use_replay is False
        assert config.replay_buffer_size == 10000
        assert config.epochs_per_episode == 1
        assert config.importance_sampling is False
        assert config.use_gae is False
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = REINFORCEConfig(
            learning_rate=5e-4,
            gamma=0.95,
            entropy_coeff=0.05,
            use_baseline=False,
            clip_grad_norm=0.5,
            use_replay=True,
            replay_buffer_size=5000,
            epochs_per_episode=4,
            importance_sampling=True,
            use_gae=True,
        )
        
        
        assert config.learning_rate == 5e-4
        assert config.gamma == 0.95
        assert config.entropy_coeff == 0.05
        assert config.use_baseline is False
        assert config.normalize_returns is True  # Default value
        assert config.clip_grad_norm == 0.5
        assert config.use_replay is True
        assert config.replay_buffer_size == 5000
        assert config.epochs_per_episode == 4
        assert config.importance_sampling is True
        assert config.use_gae is True
    

    
    @pytest.mark.parametrize("learning_rate", [-100.0, -10.0, -1.0, -0.5, -0.1, -0.01, -0.001, 0.0])
    def test_learning_rate_invalid_non_positive(self, learning_rate):
        """Test reinforce config learning rate validation"""
        
        assert learning_rate <= 0.0
        
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            REINFORCEConfig(learning_rate=learning_rate)
        
    @pytest.mark.parametrize("learning_rate", [0.0001,0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    
    def test_learning_rate_valid_positive(self, learning_rate):
        """Test that learning rate > 0 is valid"""

        assert learning_rate > 0.0
        config = REINFORCEConfig(learning_rate=learning_rate)
        assert config.learning_rate == learning_rate
        assert config.learning_rate > 0
    
    # Gamma validation
    
    @pytest.mark.parametrize("gamma", [-10.0, -1.0, -0.5,0.0])
    def test_gamma_invalid_too_low(self, gamma):
        """Test that gamma <= 0 is invalid"""
        
        assert gamma <= 0.0
        
        with pytest.raises(AssertionError, match="gamma must be in"):
            REINFORCEConfig(gamma=gamma)
        
    @pytest.mark.parametrize("gamma", [1.01, 1.1, 1.5, 2.0, 10.0])
    def test_gamma_invalid_too_high(self, gamma):
        """Test that gamma > 1 is invalid"""
        assert gamma > 1.0
        
        with pytest.raises(AssertionError, match="gamma must be in"):
            REINFORCEConfig(gamma=gamma)
    
    @pytest.mark.parametrize("gamma", [0.001, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    def test_gamma_valid_in_range(self, gamma):
        """Test that 0 < gamma <= 1 is valid"""
        
        assert 0.0 < gamma <= 1.0
        
        config = REINFORCEConfig(gamma=gamma)
        assert config.gamma == gamma
        assert 0.0 < config.gamma <= 1.0

    # Entropy coefficient validation
    @pytest.mark.parametrize("entropy_coeff", [-100.0, -10.0, -1.0, -0.5, -0.1, -0.01])
    def test_entropy_coeff_invalid_negative(self, entropy_coeff):
        """Test that entropy coefficient < 0 is invalid"""
        
        assert entropy_coeff < 0.0
        
        with pytest.raises(AssertionError, match="entropy_coeff must be non-negative"):
            REINFORCEConfig(entropy_coeff=entropy_coeff)
            
    @pytest.mark.parametrize("entropy_coeff", [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    def test_entropy_coeff_valid_non_negative(self, entropy_coeff):
        """Test that entropy_coeff >= 0 is valid"""
        assert entropy_coeff >= 0
        
        config = REINFORCEConfig(entropy_coeff=entropy_coeff)
        assert config.entropy_coeff == entropy_coeff
        assert config.entropy_coeff >= 0
    
    # Gradient clipping validation
    @pytest.mark.parametrize("clip_grad_norm", [-100.0, -10.0, -1.0, -0.5, 0.0])
    def test_clip_grad_norm_invalid_non_positive(self, clip_grad_norm):
        """Test that clip_grad_norm <= 0 is invalid"""
        assert clip_grad_norm <= 0
        
        with pytest.raises(AssertionError, match="clip_grad_norm must be positive"):
            REINFORCEConfig(clip_grad_norm=clip_grad_norm)
    
    @pytest.mark.parametrize("clip_grad_norm", [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
    def test_clip_grad_norm_valid_positive(self, clip_grad_norm):
        """Test that clip_grad_norm > 0 is valid"""
        assert clip_grad_norm > 0
        
        config = REINFORCEConfig(clip_grad_norm=clip_grad_norm)
        assert config.clip_grad_norm == clip_grad_norm
        assert config.clip_grad_norm > 0
        
    @pytest.mark.parametrize("buffer_size", [10, 100, 1000, 10000, 50000])
    def test_replay_buffer_size_valid(self, buffer_size):
        """Test valid replay buffer sizes"""
        config = REINFORCEConfig(use_replay=True, replay_buffer_size=buffer_size)
        assert config.replay_buffer_size == buffer_size
        assert config.replay_buffer_size > 0
    
    @pytest.mark.parametrize("epochs", [1, 2, 4, 8, 16, 32])
    def test_epochs_per_episode_valid(self, epochs):
        """Test valid epochs per episode"""
        config = REINFORCEConfig(epochs_per_episode=epochs)
        assert config.epochs_per_episode == epochs
        assert config.epochs_per_episode > 0
    
    @pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    def test_gae_lambda_valid(self, gae_lambda):
        """Test valid GAE lambda values"""
        config = REINFORCEConfig(use_gae=True, gae_lambda=gae_lambda)
        assert config.gae_lambda == gae_lambda
        assert 0 <= config.gae_lambda <= 1
    
    @pytest.mark.parametrize("max_weight", [1.0, 2.0, 5.0, 10.0, 20.0])
    def test_max_importance_weight_valid(self, max_weight):
        """Test valid max importance weights"""
        config = REINFORCEConfig(importance_sampling=True, max_importance_weight=max_weight)
        assert config.max_importance_weight == max_weight
        assert config.max_importance_weight > 0
        
        
        
# Memory Tests
class TestREINFORCEMemory:
    """Test REINFORCE memory buffer"""
    
    def test_initialization(self):
        """Test memory initialization"""
        memory = REINFORCEMemory()
        
        assert len(memory) == 0
        assert memory.states == []
        assert memory.actions == []
        assert memory.rewards == []
        assert memory.log_probs == []
    
    def test_store_step(self):
        """Test storing a single step"""
        memory = REINFORCEMemory()
        
        state = [0.1, 0.2, 0.3]
        action = 2
        reward = 1.0
        log_prob = -0.5
        
        memory.store_step(state, action, reward, log_prob)
        
        assert len(memory) == 1
        assert memory.states[0] == state
        assert memory.actions[0] == action
        assert memory.rewards[0] == reward
        assert memory.log_probs[0] == log_prob
    
    @pytest.mark.parametrize("num_steps", [1, 5, 10, 25, 50, 100])
    def test_store_multiple_steps(self, num_steps):
        """Test storing various numbers of steps"""
        memory = REINFORCEMemory()
        
        for i in range(num_steps):
            memory.store_step(
                state=[i, i + 1],
                action=i % 3,
                reward=float(i),
                log_prob=-float(i) * 0.1,
            )
        
        assert len(memory) == num_steps
    
    def test_get_episode(self):
        """Test retrieving episode data"""
        memory = REINFORCEMemory()
        
        for i in range(3):
            memory.store_step([i], i, float(i), -float(i))
        
        states, actions, rewards, log_probs = memory.get_episode()
        
        assert states == [[0], [1], [2]]
        assert actions == [0, 1, 2]
        assert rewards == [0.0, 1.0, 2.0]
        assert log_probs == [0.0, -1.0, -2.0]
    
    def test_clear(self):
        """Test clearing memory"""
        memory = REINFORCEMemory()
        
        memory.store_step([1, 2], 0, 1.0, -0.5)
        memory.store_step([3, 4], 1, 2.0, -0.3)
        assert len(memory) == 2
        
        memory.clear()
        assert len(memory) == 0
        assert memory.states == []
        assert memory.actions == []
        assert memory.rewards == []
        assert memory.log_probs == []
        
        
# Replay Buffer Tests
class TestReplayBuffer:
    """Test experience replay buffer"""
    
    def test_initialization(self):
        """Test buffer initialization"""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100
    
    def test_push_single_transition(self):
        """Test storing single transition"""
        buffer = ReplayBuffer(capacity=100)
        
        buffer.push(
            state=[0.1, 0.2],
            action=0,
            reward=1.0,
            log_prob=-0.5,
            done=False,
            return_=2.5,
            advantage=0.5,
        )
        
        assert len(buffer) == 1
    
    @pytest.mark.parametrize("num_transitions", [10, 50, 100, 200])
    def test_push_multiple_transitions(self, num_transitions):
        """Test storing multiple transitions with buffer wrap-around"""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(num_transitions):
            buffer.push(
                state=[i * 0.1, i * 0.2],
                action=i % 4,
                reward=float(i),
                log_prob=-float(i) * 0.1,
                done=(i % 10 == 9),
                return_=float(i),
            )
        
        # Buffer should wrap around if num_transitions > capacity
        expected_len = min(num_transitions, buffer.capacity)
        assert len(buffer) == expected_len
    
    def test_sample_batch(self):
        """Test sampling batches"""
        buffer = ReplayBuffer(capacity=100)
        
        # Add transitions
        for i in range(50):
            buffer.push(
                state=[i],
                action=i % 4,
                reward=1.0,
                log_prob=-0.5,
                done=False,
                return_=1.0,
            )
        
        # Sample batch
        batch = buffer.sample(batch_size=10)
        
        assert len(batch["states"]) == 10
        assert len(batch["actions"]) == 10
        assert len(batch["rewards"]) == 10
        assert len(batch["log_probs"]) == 10
    
    def test_clear(self):
        """Test clearing buffer"""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(10):
            buffer.push([i], i, 1.0, -0.5, False)
        
        assert len(buffer) == 10
        
        buffer.clear()
        assert len(buffer) == 0
        

# Network Tests
class TestPolicyNetwork:
    """Test policy network architecture"""
    
    @pytest.mark.parametrize(
        "n_actions,input_dims,hidden_dims",
        [
            (2, (4,), 32),
            (4, (8,), 64),
            (6, (12,), 128),
            (10, (16,), 256),
        ],
    )
    def test_initialization(self, n_actions, input_dims, hidden_dims):
        """Test policy network initialization with various dimensions"""
        policy = PolicyNetwork(
            n_actions=n_actions,
            input_dims=input_dims,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
        )
        
        assert isinstance(policy.fc1, nn.Linear)
        assert isinstance(policy.fc2, nn.Linear)
        assert isinstance(policy.fc3, nn.Linear)
        assert policy.fc1.in_features == input_dims[0]
        assert policy.fc1.out_features == hidden_dims
        assert policy.fc3.out_features == n_actions
        assert policy.optimizer is not None
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_forward_pass(self, batch_size):
        """Test forward pass with various batch sizes"""
        policy = PolicyNetwork(n_actions=4, input_dims=(8,))
        states = torch.randn(batch_size, 8)
        
        dist = policy(states)
        
        assert hasattr(dist, "sample")
        assert hasattr(dist, "log_prob")
        assert hasattr(dist, "entropy")
        
        actions = dist.sample()
        assert actions.shape == (batch_size,)
        assert torch.all(actions >= 0) and torch.all(actions < 4)


class TestValueNetwork:
    """Test value network architecture"""
    
    @pytest.mark.parametrize(
        "input_dims,hidden_dims",
        [
            ((4,), 32),
            ((8,), 64),
            ((12,), 128),
            ((16,), 256),
        ],
    )
    def test_initialization(self, input_dims, hidden_dims):
        """Test value network initialization with various dimensions"""
        value_net = ValueNetwork(
            input_dims=input_dims,
            learning_rate=1e-3,
            hidden_dims=hidden_dims,
        )
        
        assert isinstance(value_net.fc1, nn.Linear)
        assert isinstance(value_net.fc2, nn.Linear)
        assert isinstance(value_net.value_head, nn.Linear)
        assert value_net.fc1.in_features == input_dims[0]
        assert value_net.value_head.out_features == 1
        assert value_net.optimizer is not None
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_forward_pass(self, batch_size):
        """Test forward pass with various batch sizes"""
        value_net = ValueNetwork(input_dims=(8,))
        states = torch.randn(batch_size, 8)
        
        values = value_net(states)
        assert values.shape == (batch_size, 1)
        
# Sample Efficient REINFORCE Tests

class TestSampleEfficientREINFORCE:
    """Test sample-efficient REINFORCE features"""
    
    def test_create_with_replay(self):
        """Test creating agent with replay buffer"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=True,
            replay_buffer_size=1000,
        )
        
        assert agent.config.use_replay is True
        assert agent.replay_buffer is not None
        assert len(agent.replay_buffer) == 0
    
    def test_create_without_replay(self):
        """Test creating agent without replay buffer"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=False,
        )
        
        assert agent.config.use_replay is False
        assert agent.replay_buffer is None
    
    @pytest.mark.parametrize("epochs", [1, 2, 4, 8])
    def test_multiple_epochs_training(self, epochs):
        """Test training with multiple epochs per episode"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            epochs_per_episode=epochs,
        )
        
        # Store episode
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 1.0, -0.5)
        
        # Learn
        metrics = agent.learn()
        
        assert "policy_loss" in metrics
        assert "episode_length" in metrics
        assert metrics["episode_length"] == 5
    
    def test_importance_sampling(self):
        """Test importance sampling weights computation"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            importance_sampling=True,
            epochs_per_episode=2,
        )
        
        # Test importance weight computation
        old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
        new_log_probs = torch.tensor([-0.5, -0.5, -0.5])
        
        weights = agent.compute_importance_weights(old_log_probs, new_log_probs)
        
        assert weights.shape == old_log_probs.shape
        assert torch.all(weights > 0)
        assert torch.all(weights <= agent.config.max_importance_weight)
    
    def test_gae_computation(self):
        """Test Generalized Advantage Estimation"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
            gae_lambda=0.95,
        )
        
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([0.5, 0.6, 0.7])
        dones = [False, False, True]
        
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        assert torch.isfinite(advantages).all()
        assert torch.isfinite(returns).all()
    
    def test_replay_buffer_training(self):
        """Test training with replay buffer"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=True,
            replay_buffer_size=100,
            replay_batch_size=10,
            epochs_per_episode=2,
        )
        
        # Store multiple episodes
        for episode in range(3):
            for i in range(10):
                agent.store_step([i * 0.1] * 8, i % 4, 1.0, -0.5)
            
            metrics = agent.learn()
            assert "policy_loss" in metrics
        
        # Buffer should contain transitions
        assert len(agent.replay_buffer) > 0
        
# Return Computation Tests 
class TestReturnComputation:
    """Test discounted return computation"""
    
    @pytest.mark.parametrize(
        "gamma,rewards,expected",
        [
            (0.9, [1.0, 1.0, 1.0], [2.71, 1.9, 1.0]),
            (0.5, [2.0, 2.0, 2.0], [3.5, 3.0, 2.0]),
            (1.0, [1.0, 2.0, 3.0], [6.0, 5.0, 3.0]),
        ],
    )
    def test_compute_returns_various_gammas(self, gamma, rewards, expected):
        """Test return computation with different discount factors"""
        agent = create_reinforce(n_actions=4, input_dims=(8,), gamma=gamma)
        returns = agent.compute_returns(rewards, normalize=False)
        expected_tensor = torch.tensor(expected)
        
        assert torch.allclose(returns, expected_tensor, atol=1e-2)
    
    @pytest.mark.parametrize("num_steps", [2, 3, 5, 10, 20])
    def test_normalized_returns(self, num_steps):
        """Test normalized returns have mean≈0 and std≈1"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        rewards = [float(i) for i in range(1, num_steps + 1)]
        returns = agent.compute_returns(rewards, normalize=True)
        
        assert torch.allclose(returns.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(returns.std(), torch.tensor(1.0), atol=1e-5)


# Training Tests
class TestTraining:
    """Test REINFORCE training"""
    
    @pytest.mark.parametrize("episode_length", [1, 2, 5, 10, 20])
    def test_training_step(self, episode_length):
        """Test training step with various episode lengths"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        states = [[0.1] * 8] * episode_length
        actions = [i % 4 for i in range(episode_length)]
        returns = torch.tensor([1.0] * episode_length)
        log_probs = [-0.5] * episode_length
        
        metrics = agent.training_step(states, actions, returns, log_probs)
        
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "grad_norm" in metrics
        
        for value in metrics.values():
            assert np.isfinite(value)
    
    def test_learn_from_episode(self):
        """Test learning from complete episode"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        episode_length = 5
        for i in range(episode_length):
            agent.store_step([i * 0.1] * 8, i % 4, 1.0, -0.5)
        
        metrics = agent.learn()
        
        assert "policy_loss" in metrics
        assert "episode_length" in metrics
        assert metrics["episode_length"] == episode_length
        assert len(agent.memory) == 0
    
    def test_learn_empty_episode_raises_error(self):
        """Test that learning from empty episode raises error"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        with pytest.raises(ValueError, match="Cannot learn from empty episode"):
            agent.learn()
            
            
# Integration Tests


class TestIntegration:
    """Integration tests with sample efficiency"""
    
    @pytest.mark.parametrize("use_replay,epochs", [
        (False, 1),
        (True, 1),
        (True, 4),
        (False, 4),
    ])
    def test_training_variants(self, use_replay, epochs):
        """Test different training configurations"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=use_replay,
            epochs_per_episode=epochs,
            replay_buffer_size=100 if use_replay else 10000,
        )
        
        # Collect and learn from episode
        for i in range(10):
            obs = np.random.randn(8)
            action, log_prob = agent.choose_action(obs)
            reward = np.random.randn()
            agent.store_step(obs, action, reward, log_prob)
        
        metrics = agent.learn()
        
        assert "policy_loss" in metrics
        assert "episode_length" in metrics
        assert metrics["episode_length"] == 10
    
    def test_gae_with_baseline(self):
        """Test GAE with value baseline"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
            gae_lambda=0.95,
        )
        
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 1.0, -0.5)
        
        metrics = agent.learn()
        
        assert "value_loss" in metrics
        assert metrics["value_loss"] >= 0
    
    @pytest.mark.parametrize("num_episodes", [1, 3, 5])
    def test_multiple_episodes(self, num_episodes):
        """Test training over multiple episodes"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        episode_returns = []
        
        for episode in range(num_episodes):
            for step in range(5):
                obs = np.random.randn(8)
                action, log_prob = agent.choose_action(obs)
                reward = np.random.randn()
                agent.store_step(obs, action, reward, log_prob)
            
            metrics = agent.learn()
            episode_returns.append(metrics["episode_return"])
        
        assert len(episode_returns) == num_episodes


# Factory Function Tests

class TestCreateReinforce:
    """Test the create_reinforce factory function"""
    
    def test_create_with_defaults(self):
        """Test creating agent with default parameters"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        assert isinstance(agent, REINFORCEAlgorithm)
        assert agent.config.learning_rate == 1e-3
        assert agent.config.gamma == 0.99
        assert agent.config.use_baseline is True
        assert agent.config.use_replay is False
    
    @pytest.mark.parametrize(
        "learning_rate,gamma,use_baseline,use_replay,epochs",
        [
            (1e-4, 0.9, False, False, 1),
            (5e-4, 0.95, True, True, 2),
            (1e-3, 0.99, True, True, 4),
        ],
    )
    def test_create_with_custom_params(
        self, learning_rate, gamma, use_baseline, use_replay, epochs
    ):
        """Test creating agent with various custom parameters"""
        agent = create_reinforce(
            n_actions=6,
            input_dims=(10,),
            learning_rate=learning_rate,
            gamma=gamma,
            use_baseline=use_baseline,
            use_replay=use_replay,
            epochs_per_episode=epochs,
        )
        
        assert agent.config.learning_rate == learning_rate
        assert agent.config.gamma == gamma
        assert agent.config.use_baseline is use_baseline
        assert agent.config.use_replay is use_replay
        assert agent.config.epochs_per_episode == epochs
        
        
# Edge Cases for Replay Buffer
class TestReplayBufferEdgeCases:
    """Test edge cases for replay buffer"""
    
    def test_sample_larger_than_buffer(self):
        """Test sampling batch larger than buffer size"""
        buffer = ReplayBuffer(capacity=100)
        
        # Add only 5 transitions
        for i in range(5):
            buffer.push([i], i, 1.0, -0.5, False)
        
        # Try to sample 10 (more than available)
        with pytest.raises(ValueError):
            buffer.sample(batch_size=10)
    
    def test_buffer_overflow_overwrites_oldest(self):
        """Test that buffer overwrites oldest when full"""
        buffer = ReplayBuffer(capacity=3)
        
        # Add 5 transitions (more than capacity)
        for i in range(5):
            buffer.push([i], i, float(i), -0.5, False)
        
        # Should only have last 3
        assert len(buffer) == 3
        # Check that oldest were overwritten
        batch = buffer.sample(batch_size=3)
        rewards = sorted(batch["rewards"])
        assert rewards == [2.0, 3.0, 4.0]  # Should be last 3
    
    def test_empty_buffer_sample_raises_error(self):
        """Test sampling from empty buffer raises error"""
        buffer = ReplayBuffer(capacity=100)
        
        with pytest.raises(ValueError):
            buffer.sample(batch_size=5)


# Boundary Tests for Config Validation
class TestConfigBoundaries:
    """Test exact boundary conditions"""
    
    def test_learning_rate_zero_rejected(self):
        """Test learning_rate = 0 (exact boundary)"""
        with pytest.raises(AssertionError, match="learning_rate must be positive"):
            REINFORCEConfig(learning_rate=0.0)
    
    def test_learning_rate_epsilon_accepted(self):
        """Test learning_rate = epsilon (just above 0)"""
        epsilon = 1e-10
        config = REINFORCEConfig(learning_rate=epsilon)
        assert config.learning_rate > 0
    
    def test_gamma_exactly_one(self):
        """Test gamma = 1.0 (upper boundary)"""
        config = REINFORCEConfig(gamma=1.0)
        assert config.gamma == 1.0
    
    def test_gamma_just_above_one_rejected(self):
        """Test gamma = 1.0 + epsilon (just above upper bound)"""
        with pytest.raises(AssertionError, match="gamma must be in"):
            REINFORCEConfig(gamma=1.0 + 1e-10)
    
    def test_entropy_coeff_exactly_zero(self):
        """Test entropy_coeff = 0 (boundary)"""
        config = REINFORCEConfig(entropy_coeff=0.0)
        assert config.entropy_coeff == 0.0
    
    def test_gae_lambda_boundaries(self):
        """Test GAE lambda at boundaries [0, 1]"""
        config_zero = REINFORCEConfig(use_gae=True, gae_lambda=0.0)
        assert config_zero.gae_lambda == 0.0
        
        config_one = REINFORCEConfig(use_gae=True, gae_lambda=1.0)
        assert config_one.gae_lambda == 1.0
    
    def test_gae_lambda_outside_range(self):
        """Test GAE lambda outside [0, 1] is rejected"""
        with pytest.raises(AssertionError, match="gae_lambda must be in"):
            REINFORCEConfig(use_gae=True, gae_lambda=-0.1)
        
        with pytest.raises(AssertionError, match="gae_lambda must be in"):
            REINFORCEConfig(use_gae=True, gae_lambda=1.1)


# Algorithm State Tests
class TestAlgorithmState:
    """Test algorithm state management"""
    
    def test_choose_action_without_training(self):
        """Test action selection works before any training"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        obs = np.random.randn(8)
        
        action, log_prob = agent.choose_action(obs)
        
        assert isinstance(action, int)
        assert 0 <= action < 4
        assert isinstance(log_prob, float)
        assert np.isfinite(log_prob)
    
    def test_multiple_learn_calls(self):
        """Test learning multiple times in sequence"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for episode in range(3):
            # Collect episode
            for step in range(5):
                agent.store_step([0.1] * 8, step % 4, 1.0, -0.5)
            
            # Learn
            metrics = agent.learn()
            assert "policy_loss" in metrics
            
            # Memory should be cleared
            assert len(agent.memory) == 0
    
    def test_state_dict_contains_required_keys(self):
        """Test state dict has all necessary keys"""
        agent = create_reinforce(n_actions=4, input_dims=(8,), use_baseline=True)
        
        state = agent.state_dict()
        
        assert "config" in state
        assert "policy_net" in state
        assert "value_net" in state
    
    def test_state_dict_without_baseline(self):
        """Test state dict when baseline is disabled"""
        agent = create_reinforce(n_actions=4, input_dims=(8,), use_baseline=False)
        
        state = agent.state_dict()
        
        assert "config" in state
        assert "policy_net" in state
        # Should not have value_net when baseline disabled
        assert state.get("value_net") is None


# Numerical Stability Tests
class TestNumericalStability:
    """Test numerical stability edge cases"""
    
    def test_zero_rewards(self):
        """Test learning with all zero rewards"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 0.0, -0.5)
        
        metrics = agent.learn()
        
        # Should not crash
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
    
    def test_very_large_rewards(self):
        """Test learning with very large rewards"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 1000.0, -0.5)
        
        metrics = agent.learn()
        
        # Should handle large values
        assert np.isfinite(metrics["policy_loss"])
    
    def test_very_small_rewards(self):
        """Test learning with very small rewards"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 1e-10, -0.5)
        
        metrics = agent.learn()
        
        # Should handle small values
        assert np.isfinite(metrics["policy_loss"])
    
    def test_negative_rewards(self):
        """Test learning with negative rewards"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, -5.0, -0.5)
        
        metrics = agent.learn()
        
        assert np.isfinite(metrics["policy_loss"])
        assert metrics["episode_return"] < 0
    
    def test_mixed_positive_negative_rewards(self):
        """Test learning with mixed positive/negative rewards"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        rewards = [1.0, -2.0, 3.0, -1.0, 0.5]
        for i, reward in enumerate(rewards):
            agent.store_step([0.1] * 8, i % 4, reward, -0.5)
        
        metrics = agent.learn()
        
        assert np.isfinite(metrics["policy_loss"])
        assert metrics["episode_return"] == sum(rewards)


# GAE Edge Cases
class TestGAEEdgeCases:
    """Test GAE computation edge cases"""
    
    def test_gae_single_step(self):
        """Test GAE with single step episode"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
        )
        
        rewards = [1.0]
        values = torch.tensor([0.5])
        dones = [True]
        
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        assert advantages.shape == (1,)
        assert torch.isfinite(advantages).all()
    
    def test_gae_all_done_flags(self):
        """Test GAE when all steps are terminal"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
        )
        
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([0.5, 0.6, 0.7])
        dones = [True, True, True]
        
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        assert torch.isfinite(advantages).all()
    
    def test_gae_lambda_zero(self):
        """Test GAE with lambda=0 (no TD bootstrapping)"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
            gae_lambda=0.0,
        )
        
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([0.5, 0.6, 0.7])
        dones = [False, False, True]
        
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        assert torch.isfinite(advantages).all()
    
    def test_gae_lambda_one(self):
        """Test GAE with lambda=1 (Monte Carlo)"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
            gae_lambda=1.0,
        )
        
        rewards = [1.0, 1.0, 1.0]
        values = torch.tensor([0.5, 0.6, 0.7])
        dones = [False, False, True]
        
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        assert torch.isfinite(advantages).all()


# Importance Sampling Edge Cases
class TestImportanceSamplingEdgeCases:
    """Test importance sampling edge cases"""
    
    def test_importance_weights_identical_policies(self):
        """Test importance weights when old and new policies are identical"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            importance_sampling=True,
        )
        
        log_probs = torch.tensor([-1.0, -0.5, -2.0])
        weights = agent.compute_importance_weights(log_probs, log_probs)
        
        # Should all be 1.0 when policies are identical
        assert torch.allclose(weights, torch.ones_like(weights))
    
    def test_importance_weights_clipping(self):
        """Test that importance weights are properly clipped"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            importance_sampling=True,
            max_importance_weight=5.0,
        )
        
        # Very different log probs to trigger clipping
        old_log_probs = torch.tensor([-10.0, -10.0])
        new_log_probs = torch.tensor([-1.0, -1.0])
        
        weights = agent.compute_importance_weights(old_log_probs, new_log_probs)
        
        # Should be clipped to max_importance_weight
        assert torch.all(weights <= agent.config.max_importance_weight)
        assert torch.all(weights >= 1.0 / agent.config.max_importance_weight)
    
    def test_importance_weights_negative_clipping(self):
        """Test clipping when new policy is much worse"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            importance_sampling=True,
            max_importance_weight=10.0,
        )
        
        old_log_probs = torch.tensor([-1.0, -1.0])
        new_log_probs = torch.tensor([-10.0, -10.0])
        
        weights = agent.compute_importance_weights(old_log_probs, new_log_probs)
        
        # Should be clipped to lower bound
        assert torch.all(weights >= 1.0 / agent.config.max_importance_weight)


# Memory Leak / Resource Tests
class TestResourceManagement:
    """Test resource management and memory"""
    
    def test_memory_cleared_after_learn(self):
        """Test that memory is properly cleared after learning"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        # Store episode
        for i in range(10):
            agent.store_step([i] * 8, i % 4, 1.0, -0.5)
        
        assert len(agent.memory) == 10
        
        # Learn
        agent.learn()
        
        # Memory should be empty
        assert len(agent.memory) == 0
        assert len(agent.memory.states) == 0
        assert len(agent.memory.actions) == 0
    
    def test_replay_buffer_memory_management(self):
        """Test replay buffer doesn't grow unbounded"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=True,
            replay_buffer_size=50,
        )
        
        # Add more episodes than buffer size
        for episode in range(10):
            for i in range(10):
                agent.store_step([i * 0.1] * 8, i % 4, 1.0, -0.5)
            agent.learn()
        
        # Buffer should not exceed capacity
        assert len(agent.replay_buffer) <= agent.config.replay_buffer_size


# Configuration Consistency Tests
class TestConfigurationConsistency:
    """Test configuration consistency checks"""
    
    def test_replay_requires_valid_batch_size(self):
        """Test that replay batch size must be <= buffer size"""
        # This should ideally be caught in validation
        config = REINFORCEConfig(
            use_replay=True,
            replay_buffer_size=100,
            replay_batch_size=32,
        )
        
        assert config.replay_batch_size <= config.replay_buffer_size
    
    def test_gae_requires_baseline(self):
        """Test that GAE requires baseline to be enabled"""
        # GAE uses value estimates, so baseline should be on
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,  # Should be True for GAE
            use_gae=True,
        )
        
        assert agent.config.use_gae is True
        assert agent.config.use_baseline is True
    
    def test_importance_sampling_with_replay(self):
        """Test importance sampling works with replay buffer"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_replay=True,
            importance_sampling=True,
            epochs_per_episode=2,
        )
        
        # Should be able to train
        for i in range(5):
            agent.store_step([0.1] * 8, i % 4, 1.0, -0.5)
        
        metrics = agent.learn()
        assert "policy_loss" in metrics


# Entropy Tests
class TestEntropy:
    """Test entropy computation"""
    
    def test_entropy_increases_exploration(self):
        """Test that higher entropy coefficient affects loss"""
        agent_low = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            entropy_coeff=0.0,
        )
        
        agent_high = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            entropy_coeff=1.0,
        )
        
        # Store same episode for both
        states = [[0.1] * 8] * 5
        actions = [0, 1, 2, 3, 0]
        returns = torch.tensor([1.0] * 5)
        log_probs = [-0.5] * 5
        
        loss_low = agent_low.compute_loss(states, actions, returns, log_probs)
        loss_high = agent_high.compute_loss(states, actions, returns, log_probs)
        
        # Losses should be different
        assert loss_low["loss"] != loss_high["loss"]
    
    def test_entropy_is_non_negative(self):
        """Test that entropy is always non-negative"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        states = [[0.1] * 8] * 5
        actions = [0, 1, 2, 3, 0]
        returns = torch.tensor([1.0] * 5)
        log_probs = [-0.5] * 5
        
        losses = agent.compute_loss(states, actions, returns, log_probs)
        
        assert losses["entropy"].item() >= 0


# Single Step Episode Test
class TestSingleStepEpisode:
    """Test edge case of single-step episodes"""
    
    def test_single_step_episode_learning(self):
        """Test learning from a single-step episode"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        agent.store_step([0.1] * 8, 0, 1.0, -0.5)
        
        metrics = agent.learn()
        
        assert metrics["episode_length"] == 1
        assert np.isfinite(metrics["policy_loss"])
    
    def test_single_step_with_gae(self):
        """Test single-step episode with GAE"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
            use_baseline=True,
            use_gae=True,
        )
        
        agent.store_step([0.1] * 8, 0, 1.0, -0.5)
        
        metrics = agent.learn()
        
        assert metrics["episode_length"] == 1
        assert np.isfinite(metrics["policy_loss"])


# Very Long Episode Test
class TestLongEpisodes:
    """Test with very long episodes"""
    
    @pytest.mark.parametrize("episode_length", [100, 500, 1000])
    def test_long_episode_learning(self, episode_length):
        """Test learning from very long episodes"""
        agent = create_reinforce(n_actions=4, input_dims=(8,))
        
        for i in range(episode_length):
            agent.store_step([i * 0.001] * 8, i % 4, 1.0, -0.5)
        
        metrics = agent.learn()
        
        assert metrics["episode_length"] == episode_length
        assert np.isfinite(metrics["policy_loss"])
    
    def test_long_episode_with_normalization(self, ):
        """Test learning with long episodes"""
        agent = create_reinforce(
            n_actions=4,
            input_dims=(8,),
        )
        
        for i in range(100):
            agent.store_step([0.1] * 8, i % 4, float(i), -0.5)
        
        metrics = agent.learn()
        
        # Should handle normalization properly
        assert np.isfinite(metrics["policy_loss"])


# Invalid Configuration Combinations
class TestInvalidConfigCombinations:
    """Test invalid configuration combinations"""
    
    def test_negative_replay_buffer_size(self):
        """Test that negative buffer size is rejected"""
        with pytest.raises(AssertionError):
            REINFORCEConfig(
                use_replay=True,
                replay_buffer_size=-100,
            )
    
    def test_zero_replay_buffer_size(self):
        """Test that zero buffer size is rejected"""
        with pytest.raises(AssertionError):
            REINFORCEConfig(
                use_replay=True,
                replay_buffer_size=0,
            )
    
    def test_negative_epochs(self):
        """Test that negative epochs is rejected"""
        with pytest.raises(AssertionError):
            REINFORCEConfig(epochs_per_episode=-1)
    
    def test_zero_epochs(self):
        """Test that zero epochs is rejected"""
        with pytest.raises(AssertionError):
            REINFORCEConfig(epochs_per_episode=0)