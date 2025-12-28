from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.ppo import (
    PPOAlgorithm,
    PPOConfig,
    PPOMemory,
    ActorNetwork,
    CriticNetwork,
    create_ppo,
)


class SimplePolicy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        logits = self.head(embeds)
        return {"logits": logits}


@pytest.fixture
def policy_model():
    return SimplePolicy()


@pytest.fixture
def ppo_config():
    return PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        n_epochs=4,
        batch_size=8,
    )


@pytest.fixture
def ppo_algo(policy_model, ppo_config):
    return PPOAlgorithm(policy_model=policy_model, config=ppo_config)


@pytest.fixture
def actor_critic_networks():
    """Fixture for Actor and Critic networks."""
    n_actions = 4
    input_dims = (8,)
    actor = ActorNetwork(n_actions, input_dims, alpha=3e-4)
    critic = CriticNetwork(input_dims, alpha=3e-4)
    return actor, critic


# Test for configuration
def test_ppo_config_defaults():
    config = PPOConfig()
    assert config.learning_rate == 3e-4
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.policy_clip == 0.2
    assert config.n_epochs == 10
    assert config.batch_size == 64


def test_ppo_config_validation():
    with pytest.raises(AssertionError):
        PPOConfig(policy_clip=-0.1)
    with pytest.raises(AssertionError):
        PPOConfig(gamma=1.5)
    with pytest.raises(AssertionError):
        PPOConfig(gae_lambda=-0.1)
    with pytest.raises(AssertionError):
        PPOConfig(n_epochs=0)
    with pytest.raises(AssertionError):
        PPOConfig(batch_size=0)


def test_ppo_config_custom_values():
    config = PPOConfig(
        learning_rate=1e-3, gamma=0.95, policy_clip=0.3, n_epochs=5, batch_size=32
    )

    assert config.learning_rate == 1e-3
    assert config.gamma == 0.95
    assert config.policy_clip == 0.3
    assert config.n_epochs == 5
    assert config.batch_size == 32


# Test for PPOMemory
def test_memory_store_and_retrieve():
    memory = PPOMemory(batch_size=4)

    for i in range(8):
        state = [float(i)] * 4
        action = i
        prob = float(i) * 0.1
        val = float(i) * 0.5
        reward = float(i)
        done = float(i % 2)

        memory.store_memory(state, action, prob, val, reward, done)

    assert len(memory.states) == 8
    assert len(memory.actions) == 8
    assert len(memory.rewards) == 8


def test_memory_generate_batches():
    memory = PPOMemory(batch_size=4)

    for i in range(8):
        memory.store_memory(
            state=[float(i)] * 4,
            action=i,
            probs=float(i) * 0.1,
            vals=float(i) * 0.5,
            reward=float(i),
            done=0.0,
        )

    states, actions, probs, vals, rewards, dones, batches = memory.generate_batches()

    assert states.shape == (8, 4)
    assert actions.shape == (8,)
    assert probs.shape == (8,)
    assert vals.shape == (8,)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)

    assert len(batches) == 2
    assert all(len(batch) <= 4 for batch in batches)


def test_memory_clear():
    memory = PPOMemory(batch_size=4)

    memory.store_memory([1.0], 0, 0.1, 0.5, 1.0, 0.0)

    assert len(memory.states) == 1

    memory.clear_memory()
    assert len(memory.states) == 0
    assert len(memory.actions) == 0
    assert len(memory.rewards) == 0


def test_actor_network_initialization():
    n_actions = 4
    input_dims = (8,)
    actor = ActorNetwork(n_actions, input_dims, alpha=3e-4, fc1_dims=128, fc2_dims=64)

    assert actor.input_layer.in_features == 8
    assert actor.input_layer.out_features == 128
    assert actor.hidden_layer.out_features == 64
    assert actor.output_layer.out_features == 4


def test_actor_network_forward():
    n_actions = 4
    input_dims = (8,)
    actor = ActorNetwork(n_actions, input_dims, alpha=3e-4)
    state = torch.randn(2, 8)
    dist = actor(state)

    assert hasattr(dist, "sample")
    assert hasattr(dist, "log_prob")

    actions = dist.sample()
    assert actions.shape == (2,)
    assert all(0 <= action < n_actions for action in actions)


def test_actor_network_device():
    n_actions = 4
    input_dims = (8,)
    actor = ActorNetwork(n_actions, input_dims, alpha=3e-4)

    expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert str(actor.device) == expected_device


# Test for CriticNetwork
def test_critic_network_initialization():
    input_dims = (8,)

    critic = CriticNetwork(input_dims, alpha=3e-4, fc1_dims=128, fc2_dims=64)
    assert critic.input_layer.in_features == 8
    assert critic.input_layer.out_features == 128
    assert critic.hidden_layer.out_features == 64
    assert critic.value_head.out_features == 1


def test_critic_network_forward():
    input_dims = (8,)
    critic = CriticNetwork(input_dims, alpha=3e-4)

    state = torch.randn(2, 8)
    value = critic(state)

    assert value.shape == (2, 1)
    assert value.dtype == torch.float32


def test_critic_network_device():
    input_dims = (8,)
    critic = CriticNetwork(input_dims, alpha=3e-4)

    expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert str(critic.device) == expected_device


def test_compute_gae_advantages(ppo_algo):
    """Test Generalised Advantage Estimation computation."""

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    values = torch.tensor([0.5, 1.0, 1.5, 2.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

    advantages = ppo_algo.compute_gae_advantages(rewards, values, dones)

    assert advantages.shape == (4,)
    assert advantages.dtype == torch.float32

    assert not torch.isnan(advantages).any()


def test_get_log_probs_with_dict_output(ppo_algo):
    """Test log probability extraction from dict outputs"""
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    outputs = {"logits": logits}
    log_probs = ppo_algo.get_log_probs(outputs, labels)

    assert log_probs.shape == (batch_size, seq_len)
    assert not torch.isnan(log_probs).any()


def test_get_log_probs_with_tensor_output(ppo_algo):
    """Test log probability extraction from raw tensor"""
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    log_probs = ppo_algo.get_log_probs(logits, labels)

    assert log_probs.shape == (batch_size, seq_len)


def test_get_log_probs_with_masking(
    ppo_algo,
):
    """Test the masked positions get zero log probs"""

    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    labels[:, 2:] = -100

    log_probs = ppo_algo.get_log_probs(logits, labels)
    assert torch.allclose(log_probs[:, 3:], torch.zeros(batch_size, 2))


def test_extract_values(ppo_algo):
    """Test value extraction from logits"""

    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    values = ppo_algo._extract_values(logits)

    assert values.shape == (batch_size, seq_len, 1)
    assert not torch.isnan(values).any()


def test_compute_entropy(ppo_algo):
    """Test entropy computation"""
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    entropy = ppo_algo._compute_entropy(logits, mask)

    assert entropy.item() >= 0.0
    assert not torch.isnan(entropy)


def test_compute_entropy_with_masking(ppo_algo):
    """Test entropy computation with masked tokens"""
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    mask[:, 2:] = False

    entropy = ppo_algo._compute_entropy(logits, mask)

    assert entropy.item() >= 0.0
    assert not torch.isnan(entropy)


def test_compute_loss_unified_model(ppo_algo):
    """Test loss computation with unified policy model"""
    batch_size = 4
    seq_len = 5
    vocab_size = 10

    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "old_log_probs": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size),
        "returns": torch.randn(batch_size),
    }

    loss_dict = ppo_algo.compute_loss(batch)

    assert "loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy_loss" in loss_dict

    assert "clip_frac" in loss_dict
    assert "ratio_mean" in loss_dict
    assert "ratio_std" in loss_dict


def test_compute_loss_separate_networks():
    """Test loss computation with separate actor-critic networks"""

    n_actions = 4
    input_dims = (8,)

    config = PPOConfig(
        learning_rate=3e-4,
        policy_clip=0.2,
        n_epochs=4,
        batch_size=8,
        use_separate_value_network=True,
    )

    policy_model = nn.Linear(8, 8)

    ppo = PPOAlgorithm(
        policy_model=policy_model,
        config=config,
        n_actions=n_actions,
        input_dims=input_dims,
    )

    batch_size = 8
    batch = {
        "states": torch.randn(batch_size, 8),
        "actions": torch.randint(0, n_actions, (batch_size,)),
        "old_probs": torch.randn(batch_size),
        "advantages": torch.randn(batch_size),
        "returns": torch.randn(batch_size),
    }

    loss_dict = ppo._compute_loss_separate_networks(batch)

    assert "loss" in loss_dict
    assert "actor_loss" in loss_dict
    assert "critic_loss" in loss_dict
    assert "entropy_loss" in loss_dict
    assert "entropy" in loss_dict


def test_compute_diagnostics(ppo_algo):
    """Test diagnostic metrics computation"""

    batch_size = 4
    seq_len = 5

    ratio = torch.tensor([0.8, 1.0, 1.2, 1.5]).unsqueeze(1).expand(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    diagnostics = ppo_algo._compute_diagnostics(ratio, advantages, token_mask)

    assert "clip_frac" in diagnostics
    assert "ratio_mean" in diagnostics
    assert "ratio_std" in diagnostics
    assert "ratio_max" in diagnostics
    assert "ratio_min" in diagnostics

    assert 0 <= diagnostics["clip_frac"].item() <= 1.0


def test_training_step_unified_model(ppo_algo):
    """Test single training step with unified model"""

    batch_size = 4
    seq_len = 5
    vocab_size = 10

    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "old_log_probs": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size),
        "returns": torch.randn(batch_size),
    }

    metrics = ppo_algo.training_step(batch)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "grad_norm" in metrics

    assert all(isinstance(v, float) for v in metrics.values())


def test_training_step_backward_pass(ppo_algo):
    """Test the gradients are computed during training step"""

    batch_size = 4
    seq_len = 5
    vocab_size = 10

    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "old_log_probs": torch.randn(batch_size, seq_len),
        "advantages": torch.randn(batch_size),
        "returns": torch.randn(batch_size),
    }

    ppo_algo.optimizer.zero_grad()

    metrics = ppo_algo.training_step(batch)
    assert "loss" in metrics

    has_grad = False
    for param in ppo_algo.policy_model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradient computed during training step"


def test_learn_multi_epoch(ppo_algo):
    """test multi-epoch learning loop"""

    for i in range(16):
        state = [float(i)] * 4
        action = i % 4
        prob = 0.1
        value = 0.5
        reward = float(i)
        done = 0.0

        ppo_algo.remember(state, action, prob, value, reward, done)

    ppo_algo.config.n_epochs = 2
    metrics = ppo_algo.learn()

    assert len(metrics) > 0
    assert all("epoch" in m for m in metrics)

    epochs = [m["epoch"] for m in metrics]
    assert 0 in epochs
    assert 1 in epochs

    assert len(ppo_algo.memory.states) == 0


def test_learn_advantage_normalization(ppo_algo):
    """Test that advantages are normalized during learning"""

    ppo_algo.config.normalize_advantages = True
    for i in range(16):
        ppo_algo.remember([float(i)] * 4, i % 4, 0.1, 0.5, float(i), 0.0)

    with patch.object(ppo_algo, "training_step") as mock_step:
        mock_step.return_value = {"loss": 0.5}
        ppo_algo.learn()

        assert mock_step.call_count > 0

        first_call_batch = mock_step.call_args_list[0][0][0]

        advantages = first_call_batch["advantages"]
        assert torch.abs(advantages.mean()) < 0.1
        assert torch.abs(advantages.std() - 1.0) < 0.2


def test_choose_action_separate_networks():
    """Test action selection with separate networks"""

    n_actions = 4
    input_dims = (8,)

    config = PPOConfig(use_separate_value_network=True)
    policy_model = nn.Linear(8, 8)

    ppo = PPOAlgorithm(
        policy_model=policy_model,
        config=config,
        n_actions=n_actions,
        input_dims=input_dims,
    )

    observation = torch.randn(8).numpy()
    action, prob, value = ppo.choose_action(observation)

    assert isinstance(action, (int, float))
    assert isinstance(prob, float)
    assert isinstance(value, float)
    assert 0 <= action < n_actions


def test_remember_functionality(ppo_algo):
    """Test remember stores transitions correctly"""

    state = [1.0, 2.0, 3.0]
    action = 2
    prob = 0.25
    value = 1.5
    reward = 10.0
    done = 0.0

    ppo_algo.remember(state, action, prob, value, reward, done)

    assert len(ppo_algo.memory.states) == 1
    assert ppo_algo.memory.states[0] == state
    assert ppo_algo.memory.actions[0] == action
    assert ppo_algo.memory.probs[0] == prob
    assert ppo_algo.memory.vals[0] == value
    assert ppo_algo.memory.rewards[0] == reward
    assert ppo_algo.memory.dones[0] == done


# test for factory functions
def test_create_ppo_default():
    """Test PPO creation with default"""

    policy_model = SimplePolicy()
    ppo = create_ppo(policy_model)

    assert isinstance(ppo, PPOAlgorithm)
    assert ppo.config.learning_rate == 3e-4
    assert ppo.config.policy_clip == 0.2
    assert ppo.config.n_epochs == 10
    assert ppo.config.batch_size == 64


def test_create_ppo_custom():
    """Test PPO creation with custom parameters"""

    policy_model = SimplePolicy()
    ppo = create_ppo(
        policy_model, learning_rate=1e-3, policy_clip=0.3, n_epochs=5, batch_size=32
    )

    assert ppo.config.learning_rate == 1e-3
    assert ppo.config.policy_clip == 0.3
    assert ppo.config.n_epochs == 5
    assert ppo.config.batch_size == 32


def test_create_ppo_with_optimizer():
    """Test PPO creation with custom optimizer"""

    policy_model = SimplePolicy()
    custom_optimizer = torch.optim.SGD(policy_model.parameters(), lr=1e-2)

    ppo = create_ppo(policy_model, optimizer=custom_optimizer)

    assert ppo.optimizer is custom_optimizer


def test_full_training_loop_integration():
    """Integration test for full training workflow"""

    n_actions = 4
    input_dims = (8,)

    config = PPOConfig(
        learning_rate=1e-3,
        n_epochs=2,
        batch_size=4,
        use_separate_value_network=True,
    )

    policy_model = nn.Linear(8, 8)
    ppo = PPOAlgorithm(
        policy_model=policy_model,
        config=config,
        n_actions=n_actions,
        input_dims=input_dims,
    )

    for _ in range(16):
        observation = torch.randn(8).numpy()
        action, prob, value = ppo.choose_action(observation)
        reward = torch.randn(1).item()
        done = 0.0

        ppo.remember(observation, action, prob, value, reward, done)

    metrics = ppo.learn()

    assert len(metrics) > 0
    assert all("loss" in m for m in metrics)
    assert all("actor_loss" in m for m in metrics)
    assert all("critic_loss" in m for m in metrics)


def test_ppo_state_dict():
    """Test state dict saving and loading"""

    policy_model = SimplePolicy()
    config = PPOConfig()
    ppo1 = PPOAlgorithm(policy_model=policy_model, config=config)

    state = ppo1.state_dict()

    assert "config" in state
    assert "policy_model" in state
    assert state["config"]["policy_clip"] == 0.2

    ppo2 = PPOAlgorithm(policy_model=SimplePolicy(), config=config)
    ppo2.load_state_dict(state)

    assert ppo2.config.policy_clip == 0.2


def test_ppo_with_different_clip_ratios():
    """Test PPO behaviour with the different clip ratios"""
    policy_model = SimplePolicy()

    for clip_ratio in [0.1, 0.2, 0.3]:
        config = PPOConfig(policy_clip=clip_ratio)
        ppo = PPOAlgorithm(policy_model=policy_model, config=config)

        assert ppo.config.policy_clip == clip_ratio


def test_empty_memory_buffer():
    """Test handling of empty memory buffer (Issue #10)"""
    memory = PPOMemory(batch_size=4)

    # Empty memory should raise ValueError when generating batches
    with pytest.raises(
        ValueError, match="Cannot generate batches from empty memory buffer"
    ):
        memory.generate_batches()


def test_single_sample_memory():
    """Test handling of single sample in memory (Issue #11)"""
    memory = PPOMemory(batch_size=4)

    # Store single sample
    memory.store_memory(
        state=[1.0, 2.0], action=0, probs=-0.5, vals=1.0, reward=1.0, done=False
    )

    states, actions, old_probs, vals, rewards, dones, batches = (
        memory.generate_batches()
    )

    assert len(states) == 1
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_large_rollout_stress():
    """Test performance with large rollout buffer (Issue #12)"""
    memory = PPOMemory(batch_size=64)

    # Store 10,000 samples
    n_samples = 10000
    for i in range(n_samples):
        memory.store_memory(
            state=[float(i), float(i + 1)],
            action=i % 4,
            probs=-0.5,
            vals=1.0,
            reward=1.0,
            done=(i % 100 == 0),
        )

    states, actions, old_probs, vals, rewards, dones, batches = (
        memory.generate_batches()
    )

    assert len(states) == n_samples
    assert len(batches) > 0
    # Verify all samples are covered in batches
    total_samples_in_batches = sum(len(batch) for batch in batches)
    assert total_samples_in_batches == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
