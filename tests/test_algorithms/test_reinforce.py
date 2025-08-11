"""
Tests for REINFORCE algorithm implementation.

This module contains focused tests for the REINFORCE policy gradient algorithm,
covering configuration, training steps, and core functionality.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_models import (
    TEST_DEVICES,
    MockModel,
    create_dummy_batch,
)

# Import REINFORCE components
try:
    from thinkrl.algorithms.reinforce import (
        REINFORCE,
        REINFORCEConfig,
        REINFORCEReturns,
        REINFORCEBaseline,
        REINFORCELoss,
        create_reinforce_config,
        create_reinforce_algorithm,
    )
    from thinkrl.algorithms.base import AlgorithmOutput

    REINFORCE_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"REINFORCE implementation not available: {e}", allow_module_level=True)
    REINFORCE_AVAILABLE = False

pytestmark = [
    pytest.mark.algorithms,
    pytest.mark.reinforce,
]


class TestREINFORCEConfig:
    """Test REINFORCE configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = REINFORCEConfig()
        
        assert config.gamma == 0.99
        assert config.use_baseline is True
        assert config.baseline_type == "moving_average"
        assert config.use_reward_to_go is True
        assert config.entropy_coeff == 0.01

    def test_custom_configuration(self):
        """Test configuration with custom values."""
        config = REINFORCEConfig(
            gamma=0.95,
            use_baseline=False,
            entropy_coeff=0.02,
        )
        
        assert config.gamma == 0.95
        assert config.use_baseline is False
        assert config.entropy_coeff == 0.02

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = REINFORCEConfig(gamma=0.99)
        assert config.gamma == 0.99
        
        # Invalid gamma
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            REINFORCEConfig(gamma=1.5)
        
        # Invalid baseline type
        with pytest.raises(ValueError, match="baseline_type must be one of"):
            REINFORCEConfig(baseline_type="invalid")


class TestREINFORCEComponents:
    """Test REINFORCE algorithm components."""

    @pytest.fixture
    def config(self):
        return REINFORCEConfig(gamma=0.99, use_baseline=True)

    def test_returns_computation(self, config):
        """Test Monte Carlo returns computation."""
        returns_computer = REINFORCEReturns(config)
        
        batch_size, seq_len = 4, 16
        rewards = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        
        returns = returns_computer.compute_returns(rewards, attention_mask)
        
        assert returns.shape == (batch_size, seq_len)
        assert not torch.isnan(returns).any()

    def test_baseline_estimation(self, config):
        """Test baseline estimation."""
        baseline = REINFORCEBaseline(config)
        
        batch_size, seq_len, hidden_size = 4, 16, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        baseline_values = baseline.estimate_baseline(hidden_states, attention_mask)
        
        assert baseline_values.shape == (batch_size, seq_len)
        assert not torch.isnan(baseline_values).any()

    def test_policy_loss_computation(self, config):
        """Test policy gradient loss computation."""
        loss_computer = REINFORCELoss(config)
        
        batch_size, seq_len = 4, 16
        log_probs = torch.randn(batch_size, seq_len) * 0.1
        returns = torch.randn(batch_size, seq_len)
        baseline = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        
        loss, metrics = loss_computer.compute_policy_loss(
            log_probs, returns, baseline, attention_mask
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)


class TestREINFORCEAlgorithm:
    """Test main REINFORCE algorithm."""

    @pytest.fixture
    def config(self):
        return REINFORCEConfig(
            learning_rate=1e-4,
            gamma=0.99,
            use_baseline=True,
            baseline_type="moving_average"
        )

    @pytest.fixture
    def model(self):
        return MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

    @pytest.fixture
    def reinforce(self, config):
        return REINFORCE(config)

    def test_initialization(self, config):
        """Test algorithm initialization."""
        reinforce = REINFORCE(config)
        
        assert isinstance(reinforce.config, REINFORCEConfig)
        assert not reinforce.is_setup
        assert reinforce._step_count == 0

    def test_invalid_config_type(self):
        """Test that REINFORCE requires REINFORCEConfig."""
        from thinkrl.algorithms.base import AlgorithmConfig
        
        with pytest.raises(TypeError, match="config must be REINFORCEConfig"):
            REINFORCE(AlgorithmConfig())

    def test_setup(self, reinforce, model):
        """Test algorithm setup."""
        reinforce.setup(model=model)
        
        assert reinforce.is_setup
        assert reinforce.model is model
        assert reinforce.optimizer is not None

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_training_step(self, reinforce, model, device):
        """Test training step."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        reinforce.config.device = device
        reinforce.setup(model=model.to(device))
        
        batch = create_dummy_batch(batch_size=4, seq_len=16, device=device)
        batch["rewards"] = torch.randn(4, 16, device=device)
        
        output = reinforce.step(batch)
        
        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert reinforce._step_count == 1

    def test_compute_loss(self, reinforce, model):
        """Test loss computation."""
        reinforce.setup(model=model)
        
        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        
        model_outputs = model(**batch)
        output = reinforce.compute_loss(batch, model_outputs)
        
        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None
        assert "policy_loss" in output.metrics
        assert "entropy_loss" in output.metrics

    def test_generation(self, reinforce, model):
        """Test sequence generation."""
        reinforce.setup(model=model)
        
        input_ids = torch.randint(0, 1000, (2, 8))
        
        generation_output = reinforce.generate(input_ids=input_ids, max_length=16)
        
        assert "sequences" in generation_output
        assert "log_probs" in generation_output
        assert generation_output["sequences"].shape[0] == 2
        assert generation_output["sequences"].shape[1] >= 8


class TestREINFORCEHelpers:
    """Test REINFORCE helper functions."""

    def test_create_reinforce_config(self):
        """Test config creation helper."""
        config = create_reinforce_config(gamma=0.95, use_baseline=False)
        
        assert isinstance(config, REINFORCEConfig)
        assert config.gamma == 0.95
        assert config.use_baseline is False

    def test_create_reinforce_algorithm(self):
        """Test algorithm creation helper."""
        reinforce = create_reinforce_algorithm(gamma=0.99, entropy_coeff=0.02)
        
        assert isinstance(reinforce, REINFORCE)
        assert isinstance(reinforce.config, REINFORCEConfig)
        assert reinforce.config.gamma == 0.99
        assert reinforce.config.entropy_coeff == 0.02


class TestREINFORCEIntegration:
    """Integration tests for REINFORCE."""

    def test_full_training_workflow(self):
        """Test complete training workflow."""
        config = REINFORCEConfig(learning_rate=1e-4, gamma=0.99)
        model = MockModel(vocab_size=1000, hidden_size=256)
        reinforce = REINFORCE(config)
        reinforce.setup(model=model)
        
        # Training steps
        for step in range(3):
            batch = create_dummy_batch(batch_size=4, seq_len=16)
            batch["rewards"] = torch.randn(4, 16)
            
            output = reinforce.step(batch)
            
            assert output.loss is not None
            assert not torch.isnan(output.loss)
        
        assert reinforce._step_count == 3

    def test_different_baseline_types(self):
        """Test different baseline configurations."""
        baseline_types = ["moving_average", "value_function", "none"]
        
        for baseline_type in baseline_types:
            config = REINFORCEConfig(
                baseline_type=baseline_type,
                use_baseline=(baseline_type != "none")
            )
            
            reinforce = REINFORCE(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            reinforce.setup(model=model)
            
            batch = create_dummy_batch(batch_size=2, seq_len=8)
            batch["rewards"] = torch.randn(2, 8)
            
            output = reinforce.step(batch)
            assert output.loss is not None

    def test_reward_to_go_vs_full_episode(self):
        """Test different return computation methods."""
        for use_reward_to_go in [True, False]:
            config = REINFORCEConfig(use_reward_to_go=use_reward_to_go)
            reinforce = REINFORCE(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            reinforce.setup(model=model)
            
            batch = create_dummy_batch(batch_size=2, seq_len=8)
            batch["rewards"] = torch.randn(2, 8)
            
            output = reinforce.step(batch)
            assert output.loss is not None