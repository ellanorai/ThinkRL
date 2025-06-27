"""
Complete tests for DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) algorithm.

This module contains comprehensive tests for all DAPO components that we've implemented:
- Configuration validation and serialization
- Advantage estimation with group relative methods
- Loss computation with Clip-Higher technique
- Dynamic sampling for training efficiency
- Complete algorithm integration and training workflows
- Checkpoint saving/loading
- Performance and compatibility tests

Run with: pytest tests/test_algorithms/test_dapo.py -v
"""

import math
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

# Import test utilities from the project structure
from tests.test_models import (
    TEST_DEVICES,
    MockModel,
    assert_model_output,
    create_dummy_batch,
)

# Import DAPO components - handle gracefully if not implemented
try:
    from thinkrl.algorithms import (
        create_algorithm,
        get_algorithm,
        get_algorithm_status,
        list_stable_algorithms,
    )
    from thinkrl.algorithms.base import AlgorithmConfig, AlgorithmOutput, BaseAlgorithm
    from thinkrl.algorithms.dapo import (
        DAPO,
        DAPOAdvantageEstimator,
        DAPOConfig,
        DAPOLoss,
        DAPOSampler,
        create_dapo_algorithm,
        create_dapo_config,
    )

    DAPO_AVAILABLE = True

except ImportError as e:
    # If DAPO is not implemented, skip all tests
    pytest.skip(f"DAPO implementation not available: {e}", allow_module_level=True)
    DAPO_AVAILABLE = False


# Test markers for organizing test runs
pytestmark = [
    pytest.mark.algorithms,
    pytest.mark.dapo,
]


class TestDAPOConfig:
    """Test DAPO configuration class and validation."""

    def test_default_configuration(self):
        """Test that default configuration values are correct."""
        config = DAPOConfig()

        # Core DAPO parameters
        assert config.clip_ratio_lower == 0.2
        assert config.clip_ratio_higher == 5.0
        assert config.dynamic_sampling is True
        assert config.token_level_loss is True

        # Group relative advantage
        assert config.use_group_relative_advantage is True
        assert config.group_size == 8
        assert config.advantage_normalization is True

        # Overlong reward shaping
        assert config.overlong_penalty == 0.1
        assert config.max_sequence_length == 2048
        assert config.truncation_reward_adjustment is True

        # Training dynamics
        assert config.entropy_coeff == 0.01
        assert config.gae_lambda == 0.95

        # Base algorithm config
        assert config.learning_rate == 3e-4
        assert config.batch_size == 32

    def test_custom_configuration(self):
        """Test configuration with custom values."""
        config = DAPOConfig(
            clip_ratio_lower=0.1,
            clip_ratio_higher=3.0,
            dynamic_sampling=False,
            group_size=16,
            learning_rate=1e-4,
            batch_size=64,
        )

        assert config.clip_ratio_lower == 0.1
        assert config.clip_ratio_higher == 3.0
        assert config.dynamic_sampling is False
        assert config.group_size == 16
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64

    def test_clip_ratio_validation(self):
        """Test validation of clip ratio parameters."""
        # Valid configuration
        config = DAPOConfig(clip_ratio_lower=0.1, clip_ratio_higher=2.0)
        assert config.clip_ratio_lower == 0.1
        assert config.clip_ratio_higher == 2.0

        # Test that invalid configurations would raise errors in a full implementation
        # For now, we test the logic without actual validation
        with pytest.raises(ValueError, match="clip_ratio_lower must be positive"):
            DAPOConfig(clip_ratio_lower=-0.1)

        with pytest.raises(ValueError, match="clip_ratio_higher.*must be greater"):
            DAPOConfig(clip_ratio_lower=0.5, clip_ratio_higher=0.3)

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = DAPOConfig(
            clip_ratio_lower=0.15,
            clip_ratio_higher=4.0,
            dynamic_sampling=False,
            learning_rate=2e-4,
        )

        # Serialize to dict
        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["clip_ratio_lower"] == 0.15
        assert config_dict["clip_ratio_higher"] == 4.0
        assert config_dict["dynamic_sampling"] is False
        assert config_dict["learning_rate"] == 2e-4

        # Deserialize from dict
        restored_config = DAPOConfig.from_dict(config_dict)
        assert restored_config.clip_ratio_lower == 0.15
        assert restored_config.clip_ratio_higher == 4.0
        assert restored_config.dynamic_sampling is False
        assert restored_config.learning_rate == 2e-4

    def test_config_update(self):
        """Test configuration update method."""
        config = DAPOConfig()

        updated_config = config.update(
            clip_ratio_higher=10.0, learning_rate=5e-5, dynamic_sampling=False
        )

        # Original config should be unchanged
        assert config.clip_ratio_higher == 5.0
        assert config.learning_rate == 3e-4
        assert config.dynamic_sampling is True

        # Updated config should have new values
        assert updated_config.clip_ratio_higher == 10.0
        assert updated_config.learning_rate == 5e-5
        assert updated_config.dynamic_sampling is False

    def test_device_resolution(self):
        """Test automatic device resolution."""
        config = DAPOConfig(device="auto")
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected_device

        # Test explicit device setting
        config_cpu = DAPOConfig(device="cpu")
        assert config_cpu.device == "cpu"


class TestDAPOAdvantageEstimator:
    """Test DAPO advantage estimation with group relative methods."""

    @pytest.fixture
    def config(self):
        """Create test configuration for advantage estimator."""
        return DAPOConfig(
            group_size=4,
            gae_lambda=0.95,
            use_group_relative_advantage=True,
            advantage_normalization=True,
        )

    @pytest.fixture
    def estimator(self, config):
        """Create advantage estimator instance."""
        return DAPOAdvantageEstimator(config)

    def test_basic_advantage_computation(self, estimator):
        """Test basic advantage computation functionality."""
        batch_size, seq_len = 8, 16

        rewards = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        advantages = estimator.compute_advantages(
            rewards=rewards, attention_mask=attention_mask
        )

        # Validate output shape and properties
        assert advantages.shape == (batch_size, seq_len)
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()
        assert advantages.dtype == torch.float32

    def test_group_relative_advantages(self, config):
        """Test group relative advantage estimation."""
        config.use_group_relative_advantage = True
        config.group_size = 4
        estimator = DAPOAdvantageEstimator(config)

        batch_size, seq_len = 8, 16
        rewards = torch.zeros(batch_size, seq_len)

        # Create clear group differences
        rewards[:4] = 1.0  # High reward group
        rewards[4:] = -1.0  # Low reward group

        attention_mask = torch.ones(batch_size, seq_len)

        advantages = estimator.compute_advantages(
            rewards=rewards, attention_mask=attention_mask
        )

        # First group should have positive advantages
        group1_advantages = advantages[:4].mean()
        assert (
            group1_advantages > 0
        ), f"Group 1 advantages should be positive, got {group1_advantages}"

        # Second group should have negative advantages
        group2_advantages = advantages[4:].mean()
        assert (
            group2_advantages < 0
        ), f"Group 2 advantages should be negative, got {group2_advantages}"

    def test_attention_mask_handling(self, estimator):
        """Test proper handling of attention masks."""
        batch_size, seq_len = 4, 16

        rewards = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        # Mask out second half of sequences
        attention_mask[:, seq_len // 2 :] = 0

        advantages = estimator.compute_advantages(
            rewards=rewards, attention_mask=attention_mask
        )

        # Masked positions should have zero advantages
        masked_advantages = advantages[:, seq_len // 2 :]
        assert torch.allclose(
            masked_advantages, torch.zeros_like(masked_advantages), atol=1e-6
        ), "Masked positions should have zero advantages"

        # Non-masked positions should have non-zero advantages (generally)
        unmasked_advantages = advantages[:, : seq_len // 2]
        assert not torch.allclose(
            unmasked_advantages, torch.zeros_like(unmasked_advantages), atol=1e-6
        ), "Unmasked positions should have non-zero advantages"

    def test_overlong_penalty_application(self, config):
        """Test overlong sequence penalty mechanism."""
        config.truncation_reward_adjustment = True
        config.max_sequence_length = 8
        config.overlong_penalty = 0.2
        estimator = DAPOAdvantageEstimator(config)

        batch_size, seq_len = 4, 16
        rewards = torch.ones(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        # Some sequences exceed max length
        sequence_lengths = torch.tensor([6, 8, 12, 16])

        advantages = estimator.compute_advantages(
            rewards=rewards,
            attention_mask=attention_mask,
            sequence_lengths=sequence_lengths,
        )

        # Test that overlong sequences are handled
        assert advantages.shape == (batch_size, seq_len)
        # The exact penalty application would depend on implementation details

    def test_advantage_normalization(self, config):
        """Test advantage normalization functionality."""
        config.advantage_normalization = True
        estimator = DAPOAdvantageEstimator(config)

        batch_size, seq_len = 8, 16
        # Create rewards with high variance
        rewards = torch.randn(batch_size, seq_len) * 10
        attention_mask = torch.ones(batch_size, seq_len)

        advantages = estimator.compute_advantages(
            rewards=rewards, attention_mask=attention_mask
        )

        # Normalized advantages should have approximately zero mean and unit variance
        valid_advantages = advantages[attention_mask.bool()]

        mean_val = valid_advantages.mean().item()
        std_val = valid_advantages.std().item()

        assert abs(mean_val) < 0.1, f"Mean should be close to 0, got {mean_val}"
        assert abs(std_val - 1.0) < 0.2, f"Std should be close to 1, got {std_val}"

    def test_gae_computation(self, config):
        """Test Generalized Advantage Estimation computation."""
        config.gae_lambda = 0.9
        estimator = DAPOAdvantageEstimator(config)

        # Simple test case
        batch_size, seq_len = 1, 4
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        attention_mask = torch.ones(batch_size, seq_len)

        advantages = estimator.compute_advantages(
            rewards=rewards, attention_mask=attention_mask
        )

        assert advantages.shape == (batch_size, seq_len)
        # GAE should create discounted advantage estimates
        # Later rewards should contribute more to earlier advantages


class TestDAPOLoss:
    """Test DAPO loss computation with Clip-Higher technique."""

    @pytest.fixture
    def config(self):
        """Create test configuration for loss computation."""
        return DAPOConfig(
            clip_ratio_lower=0.2,
            clip_ratio_higher=3.0,
            entropy_coeff=0.01,
            token_level_loss=True,
        )

    @pytest.fixture
    def loss_computer(self, config):
        """Create loss computer instance."""
        return DAPOLoss(config)

    def test_policy_loss_computation(self, loss_computer):
        """Test basic policy loss computation."""
        batch_size, seq_len = 4, 16

        old_log_probs = torch.randn(batch_size, seq_len) * 0.1
        new_log_probs = old_log_probs + torch.randn(batch_size, seq_len) * 0.05
        advantages = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        loss, metrics = loss_computer.compute_policy_loss(
            log_probs_old=old_log_probs,
            log_probs_new=new_log_probs,
            advantages=advantages,
            attention_mask=attention_mask,
        )

        # Validate loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Should be scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Validate metrics
        assert isinstance(metrics, dict)
        expected_metrics = [
            "ratio_mean",
            "ratio_std",
            "clipped_lower_frac",
            "clipped_upper_frac",
            "clipped_higher_frac",
            "advantage_mean",
            "advantage_std",
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(
                metrics[metric], float
            ), f"Metric {metric} should be float"

    def test_clip_higher_mechanism(self, config):
        """Test the Clip-Higher technique with extreme ratios."""
        loss_computer = DAPOLoss(config)

        batch_size, seq_len = 2, 8

        # Create scenario with very high probability ratios
        old_log_probs = torch.full((batch_size, seq_len), -5.0)  # Low probability
        new_log_probs = torch.full((batch_size, seq_len), -1.0)  # High probability
        advantages = torch.ones(batch_size, seq_len)  # Positive advantages
        attention_mask = torch.ones(batch_size, seq_len)

        loss, metrics = loss_computer.compute_policy_loss(
            old_log_probs, new_log_probs, advantages, attention_mask
        )

        # Ratio should be exp(-1 - (-5)) = exp(4) ≈ 54.6
        # This exceeds clip_ratio_higher=3.0, so higher clipping should be applied
        expected_ratio = math.exp(4)
        assert (
            metrics["ratio_mean"] > config.clip_ratio_higher
        ), f"Ratio {metrics['ratio_mean']} should exceed clip_ratio_higher {config.clip_ratio_higher}"

        assert (
            metrics["clipped_higher_frac"] > 0
        ), "Some tokens should be clipped by higher bound"

    def test_entropy_loss_computation(self, loss_computer):
        """Test entropy regularization loss."""
        batch_size, seq_len, vocab_size = 4, 16, 1000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        attention_mask = torch.ones(batch_size, seq_len)

        entropy_loss = loss_computer.compute_entropy_loss(
            logits=logits, attention_mask=attention_mask
        )

        assert isinstance(entropy_loss, torch.Tensor)
        assert entropy_loss.ndim == 0
        assert not torch.isnan(entropy_loss)
        assert not torch.isinf(entropy_loss)

    def test_kl_divergence_loss(self, loss_computer):
        """Test KL divergence loss computation."""
        batch_size, seq_len = 4, 16

        new_log_probs = torch.randn(batch_size, seq_len) * 0.1
        ref_log_probs = new_log_probs + torch.randn(batch_size, seq_len) * 0.05
        attention_mask = torch.ones(batch_size, seq_len)

        kl_loss = loss_computer.compute_kl_loss(
            log_probs_new=new_log_probs,
            log_probs_ref=ref_log_probs,
            attention_mask=attention_mask,
        )

        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.ndim == 0
        assert not torch.isnan(kl_loss)
        assert kl_loss.item() >= 0  # KL divergence is non-negative

    def test_token_vs_sequence_level_loss(self, config):
        """Test difference between token-level and sequence-level loss."""
        batch_size, seq_len = 4, 16

        old_log_probs = torch.randn(batch_size, seq_len) * 0.1
        new_log_probs = old_log_probs + torch.randn(batch_size, seq_len) * 0.05
        advantages = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        # Token-level loss
        config.token_level_loss = True
        loss_computer_token = DAPOLoss(config)
        loss_token, _ = loss_computer_token.compute_policy_loss(
            old_log_probs, new_log_probs, advantages, attention_mask
        )

        # Sequence-level loss
        config.token_level_loss = False
        loss_computer_seq = DAPOLoss(config)
        loss_seq, _ = loss_computer_seq.compute_policy_loss(
            old_log_probs, new_log_probs, advantages, attention_mask
        )

        # Both should be valid tensors
        assert isinstance(loss_token, torch.Tensor)
        assert isinstance(loss_seq, torch.Tensor)
        assert not torch.isnan(loss_token)
        assert not torch.isnan(loss_seq)

        # They might be different due to different averaging
        # We just ensure both are computed successfully


class TestDAPOSampler:
    """Test DAPO dynamic sampling functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration for sampler."""
        return DAPOConfig(
            dynamic_sampling=True,
            min_gradient_threshold=1e-6,
            sample_efficiency_target=0.8,
            adaptive_threshold=True,
        )

    @pytest.fixture
    def sampler(self, config):
        """Create sampler instance."""
        return DAPOSampler(config)

    def test_no_filtering_when_disabled(self, config):
        """Test that no filtering occurs when dynamic sampling is disabled."""
        config.dynamic_sampling = False
        sampler = DAPOSampler(config)

        batch = create_dummy_batch(batch_size=8, seq_len=16)

        filtered_batch, sample_mask = sampler.filter_samples(batch)

        # Should return original batch unchanged
        assert torch.equal(filtered_batch["input_ids"], batch["input_ids"])
        assert sample_mask.all(), "All samples should be kept when sampling is disabled"

    def test_filtering_with_gradients(self, sampler):
        """Test sample filtering when gradients are provided."""
        batch = create_dummy_batch(batch_size=8, seq_len=16)

        # Mock gradients with different magnitudes
        gradients = [
            torch.randn(100, 50) * 0.01,  # Small gradients
            torch.zeros(50, 25),  # Zero gradients
            torch.randn(25, 10) * 0.1,  # Larger gradients
        ]

        filtered_batch, sample_mask = sampler.filter_samples(batch, gradients)

        # Validate filtering results
        assert isinstance(sample_mask, torch.Tensor)
        assert sample_mask.dtype == torch.bool
        assert len(sample_mask) == batch["input_ids"].shape[0]

        # Filtered batch should have same or fewer samples
        assert filtered_batch["input_ids"].shape[0] <= batch["input_ids"].shape[0]

        # All filtered samples should correspond to True mask values
        num_kept = sample_mask.sum().item()
        assert filtered_batch["input_ids"].shape[0] == num_kept

    def test_adaptive_threshold_behavior(self, sampler):
        """Test adaptive threshold adjustment."""
        batch = create_dummy_batch(batch_size=8, seq_len=16)
        gradients = [torch.randn(100, 50) * 0.01]

        initial_threshold = sampler.gradient_threshold

        # Run multiple filtering operations to trigger adaptation
        for _ in range(10):
            _, sample_mask = sampler.filter_samples(batch, gradients)
            # Simulate threshold update
            sampler._update_threshold(sample_mask, torch.randn(len(sample_mask)))

        final_threshold = sampler.gradient_threshold

        # Threshold should remain within reasonable bounds
        assert (
            1e-8 <= final_threshold <= 1e-3
        ), f"Threshold {final_threshold} outside reasonable range"

    def test_efficiency_target_maintenance(self, config):
        """Test that sampler tries to maintain target efficiency."""
        sampler = DAPOSampler(config)

        # Test low efficiency scenario
        low_efficiency_mask = torch.tensor([True, False, False, False])
        gradient_norms = torch.randn(4)

        initial_threshold = sampler.gradient_threshold
        sampler._update_threshold(low_efficiency_mask, gradient_norms)

        # Threshold should decrease to include more samples
        assert (
            sampler.gradient_threshold <= initial_threshold
        ), "Threshold should decrease when efficiency is low"

        # Test high efficiency scenario
        high_efficiency_mask = torch.tensor([True, True, True, False])
        sampler._update_threshold(high_efficiency_mask, gradient_norms)

        # After high efficiency, threshold behavior depends on history


class TestDAPOAlgorithm:
    """Test main DAPO algorithm class and integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration for DAPO algorithm."""
        return DAPOConfig(
            learning_rate=1e-4,
            batch_size=4,
            clip_ratio_lower=0.2,
            clip_ratio_higher=3.0,
            dynamic_sampling=True,
        )

    @pytest.fixture
    def model(self):
        """Create test model for DAPO."""
        return MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

    @pytest.fixture
    def reference_model(self):
        """Create reference model for KL divergence."""
        return MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

    @pytest.fixture
    def dapo(self, config):
        """Create DAPO algorithm instance."""
        return DAPO(config)

    def test_dapo_initialization(self, config):
        """Test DAPO algorithm initialization."""
        dapo = DAPO(config)

        assert isinstance(dapo.config, DAPOConfig)
        assert not dapo.is_setup
        assert dapo._step_count == 0

        # Check that components are created
        assert isinstance(dapo.advantage_estimator, DAPOAdvantageEstimator)
        assert isinstance(dapo.loss_computer, DAPOLoss)
        assert isinstance(dapo.sampler, DAPOSampler)

    def test_invalid_config_type(self):
        """Test that DAPO requires DAPOConfig specifically."""
        with pytest.raises(TypeError, match="config must be DAPOConfig"):
            DAPO(AlgorithmConfig())

    def test_setup_with_models(self, dapo, model, reference_model):
        """Test algorithm setup with models."""
        dapo.setup(model=model, reference_model=reference_model)

        assert dapo.is_setup
        assert dapo.model is model
        assert dapo._reference_model is reference_model
        assert dapo.optimizer is not None

    def test_setup_without_reference_model(self, dapo, model):
        """Test setup without reference model."""
        dapo.setup(model=model)

        assert dapo.is_setup
        assert dapo._reference_model is None

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_training_step(self, dapo, model, device):
        """Test complete training step on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Setup with device
        dapo.config.device = device
        dapo.setup(model=model.to(device))

        # Create batch with required DAPO fields
        batch = create_dummy_batch(batch_size=4, seq_len=16, device=device)
        batch["rewards"] = torch.randn(4, 16, device=device)
        batch["old_log_probs"] = torch.randn(4, 16, device=device) * 0.1

        # Perform training step
        output = dapo.step(batch)

        # Validate output
        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None
        assert isinstance(output.metrics, dict)
        assert isinstance(output.logs, dict)

        # Validate loss properties
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)
        assert output.loss.device.type == device.split(":")[0]

        # Check step count incremented
        assert dapo._step_count == 1

    def test_compute_loss_functionality(self, dapo, model):
        """Test loss computation functionality."""
        dapo.setup(model=model)

        # Create batch with all required fields
        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        # Forward pass through model
        model_outputs = model(**batch)

        # Compute loss
        output = dapo.compute_loss(batch, model_outputs)

        # Validate output structure
        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None

        # Check required metrics
        required_metrics = ["policy_loss", "entropy_loss", "total_loss"]
        for metric in required_metrics:
            assert metric in output.metrics, f"Missing metric: {metric}"

    def test_compute_loss_with_reference_model(self, dapo, model, reference_model):
        """Test loss computation with reference model for KL divergence."""
        dapo.setup(model=model, reference_model=reference_model)

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        model_outputs = model(**batch)
        output = dapo.compute_loss(batch, model_outputs)

        # Should include KL loss when reference model is used
        assert "kl_loss" in output.metrics

    def test_missing_required_batch_fields(self, dapo, model):
        """Test error handling for missing required batch fields."""
        dapo.setup(model=model)

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        model_outputs = model(**batch)

        # Test missing rewards
        with pytest.raises(ValueError, match="rewards"):
            dapo.compute_loss(batch, model_outputs)

        # Test missing old_log_probs
        batch["rewards"] = torch.randn(4, 16)
        with pytest.raises(ValueError, match="old_log_probs"):
            dapo.compute_loss(batch, model_outputs)

    def test_generation_functionality(self, dapo, model):
        """Test text generation capability."""
        dapo.setup(model=model)

        input_ids = torch.randint(0, 1000, (2, 8))

        generation_output = dapo.generate(input_ids=input_ids, max_length=16)

        # Validate generation output
        assert "sequences" in generation_output
        assert "log_probs" in generation_output
        assert "attention_mask" in generation_output

        sequences = generation_output["sequences"]
        assert sequences.shape[0] == 2  # Batch size preserved
        assert sequences.shape[1] >= 8  # At least input length

    def test_model_evaluation(self, dapo, model):
        """Test model evaluation on validation data."""
        dapo.setup(model=model)

        # Create mock evaluation data
        eval_batch = create_dummy_batch(batch_size=4, seq_len=16)
        eval_batch["rewards"] = torch.randn(4, 16)
        eval_batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        eval_dataloader = [eval_batch, eval_batch]  # Simple mock dataloader

        eval_metrics = dapo.evaluate(eval_dataloader, num_eval_steps=2)

        # Validate evaluation metrics
        assert isinstance(eval_metrics, dict)
        expected_metrics = ["eval_loss", "eval_policy_loss"]
        for metric in expected_metrics:
            assert metric in eval_metrics

    def test_checkpoint_save_and_load(self, dapo, model):
        """Test checkpoint saving and loading functionality."""
        dapo.setup(model=model)

        # Take a training step to change state
        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1
        dapo.step(batch)

        original_step_count = dapo._step_count

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save checkpoint
            dapo.save_checkpoint(checkpoint_path)

            # Create new algorithm and load
            new_dapo = DAPO(dapo.config)
            new_model = MockModel(vocab_size=1000, hidden_size=256, num_layers=2)
            new_dapo.setup(model=new_model)
            new_dapo.load_checkpoint(checkpoint_path)

            # Verify state restoration
            assert new_dapo._step_count == original_step_count

        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_algorithm_info(self, dapo, model):
        """Test algorithm information retrieval."""
        dapo.setup(model=model)

        info = dapo.get_info()

        # Validate info structure
        assert isinstance(info, dict)
        assert info["algorithm"] == "DAPO"

        # DAPO-specific info
        dapo_specific_keys = [
            "clip_ratio_lower",
            "clip_ratio_higher",
            "dynamic_sampling",
            "token_level_loss",
            "group_relative_advantage",
            "step_count",
        ]
        for key in dapo_specific_keys:
            assert key in info

    def test_step_before_setup_error(self, dapo):
        """Test error when trying to step before setup."""
        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        with pytest.raises(RuntimeError, match="must be setup"):
            dapo.step(batch)


class TestDAPOHelperFunctions:
    """Test DAPO helper functions and utilities."""

    def test_create_dapo_config(self):
        """Test DAPO config creation helper."""
        config = create_dapo_config(
            clip_ratio_lower=0.15,
            clip_ratio_higher=4.0,
            learning_rate=2e-4,
            dynamic_sampling=False,
        )

        assert isinstance(config, DAPOConfig)
        assert config.clip_ratio_lower == 0.15
        assert config.clip_ratio_higher == 4.0
        assert config.learning_rate == 2e-4
        assert config.dynamic_sampling is False

    def test_create_dapo_algorithm(self):
        """Test DAPO algorithm creation helper."""
        dapo = create_dapo_algorithm(
            clip_ratio_lower=0.1, clip_ratio_higher=2.0, learning_rate=5e-5
        )

        assert isinstance(dapo, DAPO)
        assert isinstance(dapo.config, DAPOConfig)
        assert dapo.config.clip_ratio_lower == 0.1
        assert dapo.config.clip_ratio_higher == 2.0
        assert dapo.config.learning_rate == 5e-5

    def test_create_dapo_algorithm_with_config(self):
        """Test algorithm creation with existing config."""
        config = DAPOConfig(clip_ratio_lower=0.3, clip_ratio_higher=6.0)
        dapo = create_dapo_algorithm(config=config)

        assert isinstance(dapo, DAPO)
        assert dapo.config is config


class TestDAPOIntegration:
    """Integration tests for DAPO algorithm components."""

    @pytest.fixture
    def full_setup(self):
        """Create a fully configured DAPO setup for integration testing."""
        config = DAPOConfig(
            learning_rate=1e-4,
            clip_ratio_lower=0.2,
            clip_ratio_higher=3.0,
            dynamic_sampling=True,
            token_level_loss=True,
            use_group_relative_advantage=True,
            group_size=4,
        )

        model = MockModel(vocab_size=1000, hidden_size=256, num_layers=2)
        reference_model = MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

        dapo = DAPO(config)
        dapo.setup(model=model, reference_model=reference_model)

        return dapo, model, reference_model

    def test_end_to_end_training_workflow(self, full_setup):
        """Test complete end-to-end training workflow."""
        dapo, model, reference_model = full_setup

        # Create training batches
        training_batches = []
        for _ in range(3):
            batch = create_dummy_batch(batch_size=8, seq_len=16)
            batch["rewards"] = torch.randn(8, 16)
            batch["old_log_probs"] = torch.randn(8, 16) * 0.1
            training_batches.append(batch)

        # Training loop
        losses = []
        for batch in training_batches:
            output = dapo.step(batch)
            losses.append(output.loss.item())

            # Validate output components
            assert "policy_loss" in output.metrics
            assert "entropy_loss" in output.metrics
            assert "total_loss" in output.metrics
            assert "advantage_mean" in output.metrics
            assert "ratio_mean" in output.metrics

        # Validate training progression
        assert len(losses) == 3
        assert all(not math.isnan(loss) for loss in losses)
        assert all(math.isfinite(loss) for loss in losses)
        assert dapo._step_count == 3

    def test_different_batch_sizes(self, full_setup):
        """Test training with varying batch sizes."""
        dapo, model, reference_model = full_setup

        batch_sizes = [2, 4, 8, 16]

        for batch_size in batch_sizes:
            batch = create_dummy_batch(batch_size=batch_size, seq_len=12)
            batch["rewards"] = torch.randn(batch_size, 12)
            batch["old_log_probs"] = torch.randn(batch_size, 12) * 0.1

            output = dapo.step(batch)

            assert output.loss is not None
            assert not torch.isnan(output.loss)
            assert output.metrics["total_loss"] == output.loss.item()

    def test_different_sequence_lengths(self, full_setup):
        """Test training with varying sequence lengths."""
        dapo, model, reference_model = full_setup

        seq_lengths = [8, 16, 32, 64]

        for seq_len in seq_lengths:
            batch = create_dummy_batch(batch_size=4, seq_len=seq_len)
            batch["rewards"] = torch.randn(4, seq_len)
            batch["old_log_probs"] = torch.randn(4, seq_len) * 0.1

            output = dapo.step(batch)

            assert output.loss is not None
            assert not torch.isnan(output.loss)

    def test_masked_sequences(self, full_setup):
        """Test training with attention masks for variable-length sequences."""
        dapo, model, reference_model = full_setup

        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        # Create variable length sequences with masks
        for i in range(4):
            mask_start = 8 + i * 2  # 8, 10, 12, 14
            batch["attention_mask"][i, mask_start:] = 0

        output = dapo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_gradient_flow_validation(self, full_setup):
        """Test that gradients flow properly through the algorithm."""
        dapo, model, reference_model = full_setup

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        batch = create_dummy_batch(batch_size=4, seq_len=16)
        batch["rewards"] = torch.randn(4, 16)
        batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        output = dapo.step(batch)

        # Verify parameters changed
        param_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                param_changed = True
                break

        assert param_changed, "Model parameters should change after training step"

    def test_training_stability(self, full_setup):
        """Test training stability over multiple steps."""
        dapo, model, reference_model = full_setup

        losses = []
        for step in range(10):
            batch = create_dummy_batch(batch_size=4, seq_len=16)
            batch["rewards"] = torch.randn(4, 16)
            batch["old_log_probs"] = torch.randn(4, 16) * 0.1

            output = dapo.step(batch)
            losses.append(output.loss.item())

        # Check for stability
        assert len(losses) == 10
        assert all(not math.isnan(loss) for loss in losses)
        assert all(abs(loss) < 100 for loss in losses), "Loss should remain bounded"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self):
        """Test DAPO training on GPU."""
        config = DAPOConfig(device="cuda", learning_rate=1e-4)
        model = MockModel(vocab_size=1000, hidden_size=256).cuda()
        reference_model = MockModel(vocab_size=1000, hidden_size=256).cuda()

        dapo = DAPO(config)
        dapo.setup(model=model, reference_model=reference_model)

        batch = create_dummy_batch(batch_size=4, seq_len=16, device="cuda")
        batch["rewards"] = torch.randn(4, 16, device="cuda")
        batch["old_log_probs"] = torch.randn(4, 16, device="cuda") * 0.1

        output = dapo.step(batch)

        assert output.loss.device.type == "cuda"
        assert not torch.isnan(output.loss)

    def test_memory_efficiency(self, full_setup):
        """Test memory efficiency with larger batches."""
        dapo, model, reference_model = full_setup

        # Test with larger batch to check memory usage
        large_batch = create_dummy_batch(batch_size=32, seq_len=64)
        large_batch["rewards"] = torch.randn(32, 64)
        large_batch["old_log_probs"] = torch.randn(32, 64) * 0.1

        # Should handle large batches without issues
        output = dapo.step(large_batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_error_handling_robustness(self, full_setup):
        """Test robust error handling in various scenarios."""
        dapo, model, reference_model = full_setup

        # Test with malformed batch
        bad_batch = {"input_ids": torch.randn(4, 16)}  # Wrong tensor type/content

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            dapo.step(bad_batch)

        # Test with mismatched tensor sizes
        mismatched_batch = create_dummy_batch(batch_size=4, seq_len=16)
        mismatched_batch["rewards"] = torch.randn(4, 8)  # Wrong sequence length
        mismatched_batch["old_log_probs"] = torch.randn(4, 16) * 0.1

        with pytest.raises((ValueError, RuntimeError)):
            dapo.step(mismatched_batch)


class TestDAPORegistryIntegration:
    """Test DAPO integration with algorithm registry."""

    def test_algorithm_registry_functionality(self):
        """Test DAPO works with the algorithm registry."""
        # Test getting algorithm from registry
        algorithm_class = get_algorithm("dapo")
        assert algorithm_class == DAPO

        # Test creating algorithm via registry
        dapo = create_algorithm("dapo", learning_rate=1e-4, clip_ratio_higher=5.0)
        assert isinstance(dapo, DAPO)
        assert dapo.config.learning_rate == 1e-4
        assert dapo.config.clip_ratio_higher == 5.0

    def test_stable_algorithms_list(self):
        """Test that DAPO appears in stable algorithms."""
        stable_algos = list_stable_algorithms()
        assert "dapo" in stable_algos

    def test_algorithm_status(self):
        """Test DAPO algorithm status."""
        status = get_algorithm_status("dapo")
        assert status == "stable"


class TestDAPOPerformance:
    """Performance and benchmarking tests for DAPO."""

    @pytest.mark.slow
    def test_training_performance(self):
        """Test DAPO training performance (marked as slow)."""
        config = DAPOConfig(learning_rate=1e-4, batch_size=16)
        model = MockModel(vocab_size=5000, hidden_size=512, num_layers=4)
        dapo = DAPO(config)
        dapo.setup(model=model)

        import time

        batch = create_dummy_batch(batch_size=16, seq_len=32, vocab_size=5000)
        batch["rewards"] = torch.randn(16, 32)
        batch["old_log_probs"] = torch.randn(16, 32) * 0.1

        # Warmup
        for _ in range(3):
            dapo.step(batch)

        # Performance measurement
        start_time = time.time()
        num_steps = 10

        for _ in range(num_steps):
            output = dapo.step(batch)
            assert output.loss is not None

        end_time = time.time()

        avg_step_time = (end_time - start_time) / num_steps
        print(f"Average DAPO step time: {avg_step_time:.4f} seconds")

        # Performance threshold (adjust based on hardware)
        assert avg_step_time < 5.0, f"Training step too slow: {avg_step_time:.4f}s"

    def test_memory_usage_monitoring(self):
        """Test memory usage during DAPO training."""
        config = DAPOConfig()
        model = MockModel(vocab_size=1000, hidden_size=256, num_layers=2)
        dapo = DAPO(config)
        dapo.setup(model=model)

        batch = create_dummy_batch(batch_size=8, seq_len=32)
        batch["rewards"] = torch.randn(8, 32)
        batch["old_log_probs"] = torch.randn(8, 32) * 0.1

        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Multiple training steps
        for _ in range(10):
            output = dapo.step(batch)
            del output

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many objects created: {object_growth}"


# Module-level test configuration
def test_dapo_module_imports():
    """Test that all DAPO components can be imported correctly."""
    # This test verifies the import structure
    assert DAPO_AVAILABLE, "DAPO should be available for testing"

    # Test basic instantiation
    config = DAPOConfig()
    dapo = DAPO(config)

    assert isinstance(config, DAPOConfig)
    assert isinstance(dapo, DAPO)
    assert isinstance(dapo, BaseAlgorithm)


def test_dapo_in_algorithm_list():
    """Test that DAPO appears in algorithm listings."""
    try:
        from thinkrl.algorithms import list_algorithms

        algorithms = list_algorithms()
        assert "dapo" in algorithms
    except ImportError:
        pytest.skip("Algorithm listing not available")


# Performance markers for test organization
slow_tests = pytest.mark.slow
gpu_tests = pytest.mark.gpu
integration_tests = pytest.mark.integration


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
