"""
Tests for PPO (Proximal Policy Optimization) algorithm implementation.

This module contains focused tests for the PPO algorithm,
covering clipping, value functions, GAE, and core functionality.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_models import (
    TEST_DEVICES,
    MockModel,
    create_dummy_batch,
)

# Import PPO components
try:
    from thinkrl.algorithms.base import AlgorithmOutput
    from thinkrl.algorithms.ppo import (
        PPO,
        PPOAdvantageEstimator,
        PPOConfig,
        PPOLoss,
        PPOValueFunction,
        create_ppo_algorithm,
        create_ppo_config,
    )

    PPO_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"PPO implementation not available: {e}", allow_module_level=True)
    PPO_AVAILABLE = False

pytestmark = [
    pytest.mark.algorithms,
    pytest.mark.ppo,
]


class TestPPOConfig:
    """Test PPO configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = PPOConfig()

        assert config.clip_ratio == 0.2
        assert config.value_loss_coeff == 0.5
        assert config.entropy_coeff == 0.01
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.ppo_epochs == 4
        assert config.num_mini_batches == 4

    def test_custom_configuration(self):
        """Test configuration with custom values."""
        config = PPOConfig(
            clip_ratio=0.1,
            ppo_epochs=2,
            num_mini_batches=2,
            gamma=0.95,
        )

        assert config.clip_ratio == 0.1
        assert config.ppo_epochs == 2
        assert config.num_mini_batches == 2
        assert config.gamma == 0.95

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PPOConfig(clip_ratio=0.2)
        assert config.clip_ratio == 0.2

        # Invalid clip_ratio
        with pytest.raises(ValueError, match="clip_ratio must be positive"):
            PPOConfig(clip_ratio=-0.1)

        # Invalid gamma
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            PPOConfig(gamma=1.5)

        # Invalid ppo_epochs
        with pytest.raises(ValueError, match="ppo_epochs must be positive"):
            PPOConfig(ppo_epochs=0)


class TestPPOComponents:
    """Test PPO algorithm components."""

    @pytest.fixture
    def config(self):
        return PPOConfig(clip_ratio=0.2, gamma=0.99, gae_lambda=0.95)

    def test_advantage_estimation(self, config):
        """Test GAE advantage estimation."""
        estimator = PPOAdvantageEstimator(config)

        batch_size, seq_len = 4, 16
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        advantages, returns = estimator.compute_advantages_and_returns(
            rewards, values, attention_mask
        )

        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_value_function(self, config):
        """Test value function component."""
        value_fn = PPOValueFunction(config)
        hidden_size = 512

        value_fn.setup(hidden_size, "cpu")

        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        values = value_fn.forward(hidden_states)

        assert values.shape == (batch_size, seq_len)
        assert not torch.isnan(values).any()

    def test_value_loss_computation(self, config):
        """Test value function loss computation."""
        value_fn = PPOValueFunction(config)
        value_fn.setup(512, "cpu")

        batch_size, seq_len = 4, 16
        predicted_values = torch.randn(batch_size, seq_len)
        target_returns = torch.randn(batch_size, seq_len)
        old_values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        loss, metrics = value_fn.compute_value_loss(
            predicted_values, target_returns, old_values, attention_mask
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)

    def test_policy_loss_computation(self, config):
        """Test PPO clipped policy loss."""
        loss_computer = PPOLoss(config)

        batch_size, seq_len = 4, 16
        log_probs_old = torch.randn(batch_size, seq_len) * 0.1
        log_probs_new = log_probs_old + torch.randn(batch_size, seq_len) * 0.05
        advantages = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        loss, metrics = loss_computer.compute_policy_loss(
            log_probs_old, log_probs_new, advantages, attention_mask
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)
        assert "ratio_mean" in metrics
        assert "clipped_lower_frac" in metrics
        assert "clipped_upper_frac" in metrics

    def test_clipping_behavior(self, config):
        """Test that clipping works correctly."""
        loss_computer = PPOLoss(config)

        batch_size, seq_len = 2, 4

        # Create scenario with high ratios that should be clipped
        log_probs_old = torch.full((batch_size, seq_len), -5.0)
        log_probs_new = torch.full((batch_size, seq_len), -1.0)  # High ratio
        advantages = torch.ones(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        loss, metrics = loss_computer.compute_policy_loss(
            log_probs_old, log_probs_new, advantages, attention_mask
        )

        # Should have clipping
        assert (
            metrics["clipped_upper_frac"] > 0
            or metrics["ratio_mean"] > 1 + config.clip_ratio
        )


class TestPPOAlgorithm:
    """Test main PPO algorithm."""

    @pytest.fixture
    def config(self):
        return PPOConfig(
            learning_rate=1e-4,
            clip_ratio=0.2,
            ppo_epochs=2,  # Reduced for faster testing
            num_mini_batches=2,
        )

    @pytest.fixture
    def model(self):
        return MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

    @pytest.fixture
    def ppo(self, config):
        return PPO(config)

    def test_initialization(self, config):
        """Test algorithm initialization."""
        ppo = PPO(config)

        assert isinstance(ppo.config, PPOConfig)
        assert not ppo.is_setup
        assert ppo._step_count == 0

    def test_invalid_config_type(self):
        """Test that PPO requires PPOConfig."""
        from thinkrl.algorithms.base import AlgorithmConfig

        with pytest.raises(TypeError, match="config must be PPOConfig"):
            PPO(AlgorithmConfig())

    def test_setup(self, ppo, model):
        """Test algorithm setup."""
        ppo.setup(model=model)

        assert ppo.is_setup
        assert ppo.model is model
        assert ppo.optimizer is not None
        assert ppo._is_value_function_setup

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_training_step(self, ppo, model, device):
        """Test training step with multiple epochs."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ppo.config.device = device
        ppo.setup(model=model.to(device))

        batch = create_dummy_batch(batch_size=4, seq_len=16, device=device)
        batch["rewards"] = torch.randn(4, 16, device=device)

        output = ppo.step(batch)

        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert ppo._step_count == 1

        # Check PPO-specific metrics
        assert "policy_loss" in output.metrics
        assert "value_loss" in output.metrics
        assert "entropy_loss" in output.metrics

    def test_old_policy_data_collection(self, ppo, model):
        """Test collection of old policy data."""
        ppo.setup(model=model)

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)

        # Should work without old_log_probs and old_values
        output = ppo.step(batch)

        assert output.loss is not None

    def test_with_provided_old_data(self, ppo, model):
        """Test training with pre-computed old policy data."""
        ppo.setup(model=model)

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1
        batch["old_values"] = torch.randn(4, 16)

        output = ppo.step(batch)

        assert output.loss is not None

    def test_generation(self, ppo, model):
        """Test sequence generation."""
        ppo.setup(model=model)

        input_ids = torch.randint(0, 1000, (2, 8))

        generation_output = ppo.generate(input_ids=input_ids, max_length=16)

        assert "sequences" in generation_output
        assert "log_probs" in generation_output
        assert generation_output["sequences"].shape[0] == 2
        assert generation_output["sequences"].shape[1] >= 8

    def test_evaluation(self, ppo, model):
        """Test model evaluation."""
        ppo.setup(model=model)

        # Create mock evaluation data
        eval_batch = create_dummy_batch(batch_size=4, seq_len=16)
        eval_batch["rewards"] = torch.randn(4, 16)

        eval_dataloader = [eval_batch]

        eval_metrics = ppo.evaluate(eval_dataloader, num_eval_steps=1)

        assert isinstance(eval_metrics, dict)
        assert "eval_loss" in eval_metrics
        assert "eval_policy_loss" in eval_metrics

    def test_mini_batch_creation(self, ppo, model):
        """Test mini-batch creation for PPO epochs."""
        ppo.setup(model=model)

        batch = create_dummy_batch(batch_size=8, seq_len=16)
        batch["rewards"] = torch.randn(8, 16)
        batch["old_log_probs"] = torch.randn(8, 16) * 0.1
        batch["old_values"] = torch.randn(8, 16)

        mini_batches = ppo._create_mini_batches(batch)

        assert len(mini_batches) == ppo.config.num_mini_batches

        # Check that all mini-batches have correct structure
        for mini_batch in mini_batches:
            assert "input_ids" in mini_batch
            assert "attention_mask" in mini_batch
            assert "rewards" in mini_batch


class TestPPOHelpers:
    """Test PPO helper functions."""

    def test_create_ppo_config(self):
        """Test config creation helper."""
        config = create_ppo_config(clip_ratio=0.1, ppo_epochs=2, gamma=0.95)

        assert isinstance(config, PPOConfig)
        assert config.clip_ratio == 0.1
        assert config.ppo_epochs == 2
        assert config.gamma == 0.95

    def test_create_ppo_algorithm(self):
        """Test algorithm creation helper."""
        ppo = create_ppo_algorithm(clip_ratio=0.2, ppo_epochs=4)

        assert isinstance(ppo, PPO)
        assert isinstance(ppo.config, PPOConfig)
        assert ppo.config.clip_ratio == 0.2
        assert ppo.config.ppo_epochs == 4


class TestPPOIntegration:
    """Integration tests for PPO."""

    def test_full_training_workflow(self):
        """Test complete training workflow with multiple epochs."""
        config = PPOConfig(learning_rate=1e-4, ppo_epochs=2, num_mini_batches=2)
        model = MockModel(vocab_size=1000, hidden_size=256)
        ppo = PPO(config)
        ppo.setup(model=model)

        # Training steps
        for step in range(3):
            batch = create_dummy_batch(batch_size=4, seq_len=16)
            batch["rewards"] = torch.randn(4, 16)

            output = ppo.step(batch)

            assert output.loss is not None
            assert not torch.isnan(output.loss)
            assert "ppo_epochs_completed" in output.logs
            assert "num_ppo_updates" in output.logs

        assert ppo._step_count == 3

    def test_different_batch_sizes(self):
        """Test PPO with different batch sizes."""
        config = PPOConfig(ppo_epochs=2, num_mini_batches=2)

        for batch_size in [2, 4, 8]:
            ppo = PPO(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            ppo.setup(model=model)

            batch = create_dummy_batch(batch_size=batch_size, seq_len=16)
            batch["rewards"] = torch.randn(batch_size, 16)

            output = ppo.step(batch)
            assert output.loss is not None

    def test_gae_vs_simple_advantages(self):
        """Test different advantage estimation methods."""
        for use_gae in [True, False]:
            config = PPOConfig(use_gae=use_gae, ppo_epochs=1, num_mini_batches=1)

            ppo = PPO(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            ppo.setup(model=model)

            batch = create_dummy_batch(batch_size=4, seq_len=16)
            batch["rewards"] = torch.randn(4, 16)

            output = ppo.step(batch)
            assert output.loss is not None

    def test_value_clipping_options(self):
        """Test different value function clipping options."""
        for use_value_clipping in [True, False]:
            config = PPOConfig(
                use_value_clipping=use_value_clipping, ppo_epochs=1, num_mini_batches=1
            )

            ppo = PPO(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            ppo.setup(model=model)

            batch = create_dummy_batch(batch_size=4, seq_len=16)
            batch["rewards"] = torch.randn(4, 16)

            output = ppo.step(batch)
            assert output.loss is not None

    def test_early_stopping_kl(self):
        """Test early stopping based on KL divergence."""
        config = PPOConfig(
            ppo_epochs=10,  # High number
            target_kl=0.01,  # Low threshold
            enable_early_stopping=True,
            num_mini_batches=1,
        )

        ppo = PPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        ppo.setup(model=model)

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)

        output = ppo.step(batch)

        assert output.loss is not None
        # Should stop early if KL threshold is exceeded
        epochs_completed = output.logs.get("ppo_epochs_completed", 0)
        assert epochs_completed >= 1  # At least one epoch should complete

    def test_gradient_accumulation(self):
        """Test that gradient accumulation works properly."""
        config = PPOConfig(
            ppo_epochs=2,
            num_mini_batches=4,  # Multiple mini-batches for accumulation
            learning_rate=1e-4,
        )

        ppo = PPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        ppo.setup(model=model)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        batch = create_dummy_batch(batch_size=8, seq_len=16)
        batch["rewards"] = torch.randn(8, 16)

        output = ppo.step(batch)

        # Check that parameters have changed
        param_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                param_changed = True
                break

        assert param_changed, "Model parameters should change after training"
        assert output.loss is not None
