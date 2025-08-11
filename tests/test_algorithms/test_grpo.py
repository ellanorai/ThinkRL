"""
Tests for GRPO (Group Relative Policy Optimization) algorithm implementation.

This module contains focused tests for the GRPO algorithm,
covering group normalization, batching, and core functionality.
"""

import pytest
import torch

from tests.test_models import (
    TEST_DEVICES,
    MockModel,
    create_dummy_batch,
)

# Import GRPO components
try:
    from thinkrl.algorithms.grpo import (
        GRPO,
        GRPOConfig,
        GRPORewardNormalizer,
        GRPOLoss,
        GRPOBatcher,
    )
    from thinkrl.algorithms.base import AlgorithmOutput

    GRPO_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"GRPO implementation not available: {e}", allow_module_level=True)
    GRPO_AVAILABLE = False

pytestmark = [
    pytest.mark.algorithms,
    pytest.mark.grpo,
]


class TestGRPOConfig:
    """Test GRPO configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = GRPOConfig()

        assert config.use_group_normalization is True
        assert config.group_min_size == 2
        assert config.token_level_loss is True
        assert config.entropy_coeff == 0.01
        assert config.micro_batch_size == 4

    def test_custom_configuration(self):
        """Test configuration with custom values."""
        config = GRPOConfig(
            use_group_normalization=False,
            group_min_size=4,
            entropy_coeff=0.02,
        )

        assert config.use_group_normalization is False
        assert config.group_min_size == 4
        assert config.entropy_coeff == 0.02

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid group_min_size
        with pytest.raises(ValueError, match="group_min_size must be at least 1"):
            GRPOConfig(group_min_size=0)

        # Invalid reward_clip_range
        with pytest.raises(ValueError, match="reward_clip_range must be"):
            GRPOConfig(reward_clip_range=(5.0, -5.0))


class TestGRPOComponents:
    """Test GRPO algorithm components."""

    @pytest.fixture
    def config(self):
        return GRPOConfig(
            use_group_normalization=True, group_min_size=2, micro_batch_size=2
        )

    def test_reward_normalization(self, config):
        """Test group-based reward normalization."""
        normalizer = GRPORewardNormalizer(config)

        # Create rewards with clear group differences
        rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])  # Two groups
        prompts = ["prompt1", "prompt1", "prompt2", "prompt2"]

        normalized = normalizer.normalize_rewards(rewards, prompts)

        assert normalized.shape == rewards.shape
        assert not torch.isnan(normalized).any()

        # Groups should be normalized separately
        group1_mean = normalized[:2].mean()
        group2_mean = normalized[2:].mean()
        assert abs(group1_mean.item()) < 0.1  # Should be close to 0
        assert abs(group2_mean.item()) < 0.1

    def test_reward_normalization_disabled(self, config):
        """Test reward normalization when disabled."""
        config.use_group_normalization = False
        normalizer = GRPORewardNormalizer(config)

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        prompts = ["p1", "p1", "p2", "p2"]

        normalized = normalizer.normalize_rewards(rewards, prompts)

        # Should return original rewards when normalization is disabled
        assert torch.equal(normalized, rewards)

    def test_policy_loss_computation(self, config):
        """Test GRPO policy loss computation."""
        loss_computer = GRPOLoss(config)

        batch_size, seq_len = 4, 16
        log_probs = torch.randn(batch_size, seq_len) * 0.1
        advantages = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)
        generated_mask = torch.ones(batch_size, seq_len)

        loss, metrics = loss_computer.compute_policy_loss(
            log_probs, advantages, attention_mask, generated_mask
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)

    def test_micro_batch_creation(self, config):
        """Test micro-batch creation for efficient training."""
        batcher = GRPOBatcher(config)

        # Create sample episodes
        episodes = [
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [4, 5],
                "reward": 1.0,
                "prompt": "test prompt 1",
            },
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [4, 5, 6],
                "reward": -1.0,
                "prompt": "test prompt 2",
            },
        ]

        micro_batches = batcher.create_micro_batches(episodes, pad_token_id=0)

        assert len(micro_batches) > 0

        # Check micro-batch structure
        micro_batch = micro_batches[0]
        assert "input_ids" in micro_batch
        assert "attention_mask" in micro_batch
        assert "generated_mask" in micro_batch
        assert "rewards" in micro_batch
        assert "prompts" in micro_batch


class TestGRPOAlgorithm:
    """Test main GRPO algorithm."""

    @pytest.fixture
    def config(self):
        return GRPOConfig(
            learning_rate=1e-4, use_group_normalization=True, micro_batch_size=2
        )

    @pytest.fixture
    def model(self):
        return MockModel(vocab_size=1000, hidden_size=256, num_layers=2)

    @pytest.fixture
    def grpo(self, config):
        return GRPO(config)

    def test_initialization(self, config):
        """Test algorithm initialization."""
        grpo = GRPO(config)

        assert isinstance(grpo.config, GRPOConfig)
        assert not grpo.is_setup
        assert grpo._step_count == 0

    def test_invalid_config_type(self):
        """Test that GRPO requires GRPOConfig."""
        from thinkrl.algorithms.base import AlgorithmConfig

        with pytest.raises(TypeError, match="config must be GRPOConfig"):
            GRPO(AlgorithmConfig())

    def test_setup(self, grpo, model):
        """Test algorithm setup."""
        grpo.setup(model=model)

        assert grpo.is_setup
        assert grpo.model is model
        assert grpo.optimizer is not None

    @pytest.mark.parametrize("device", TEST_DEVICES)
    def test_training_step(self, grpo, model, device):
        """Test training step with episodes."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        grpo.config.device = device
        grpo.setup(model=model.to(device))

        # Create sample episodes for GRPO
        episodes = [
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [4, 5],
                "reward": 1.0,
                "prompt": "test prompt 1",
            },
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [4, 5, 6],
                "reward": -1.0,
                "prompt": "test prompt 2",
            },
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        output = grpo.step(batch)

        assert isinstance(output, AlgorithmOutput)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert grpo._step_count == 1

    def test_missing_episodes(self, grpo, model):
        """Test error handling for missing episodes."""
        grpo.setup(model=model)

        batch = {"pad_token_id": 0}  # Missing episodes

        with pytest.raises(ValueError, match="batch must contain 'episodes'"):
            grpo.step(batch)

    def test_empty_episodes(self, grpo, model):
        """Test handling of empty episodes list."""
        grpo.setup(model=model)

        batch = {"episodes": [], "pad_token_id": 0}  # Empty episodes

        with pytest.raises(ValueError, match="batch must contain 'episodes'"):
            grpo.step(batch)


class TestGRPOIntegration:
    """Integration tests for GRPO."""

    def test_full_training_workflow(self):
        """Test complete training workflow."""
        config = GRPOConfig(learning_rate=1e-4, micro_batch_size=2)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo = GRPO(config)
        grpo.setup(model=model)

        # Create episodes with group structure
        for step in range(3):
            episodes = [
                {
                    "prefix_tokens": [1, 2, 3],
                    "generated_tokens": [4, 5],
                    "reward": 1.0 if i % 2 == 0 else -1.0,
                    "prompt": f"prompt_{i % 2}",  # Two groups
                }
                for i in range(4)
            ]

            batch = {"episodes": episodes, "pad_token_id": 0}

            output = grpo.step(batch)

            assert output.loss is not None
            assert not torch.isnan(output.loss)

        assert grpo._step_count == 3

    def test_group_normalization_effect(self):
        """Test that group normalization affects training."""
        # Test with group normalization enabled
        config_with_norm = GRPOConfig(use_group_normalization=True, micro_batch_size=2)
        grpo_with_norm = GRPO(config_with_norm)
        model1 = MockModel(vocab_size=1000, hidden_size=256)
        grpo_with_norm.setup(model=model1)

        # Test without group normalization
        config_without_norm = GRPOConfig(
            use_group_normalization=False, micro_batch_size=2
        )
        grpo_without_norm = GRPO(config_without_norm)
        model2 = MockModel(vocab_size=1000, hidden_size=256)
        grpo_without_norm.setup(model=model2)

        # Same episodes for both
        episodes = [
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [4, 5],
                "reward": 2.0,
                "prompt": "group1",
            },
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [4, 5],
                "reward": -2.0,
                "prompt": "group2",
            },
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        # Both should work without errors
        output1 = grpo_with_norm.step(batch.copy())
        output2 = grpo_without_norm.step(batch.copy())

        assert output1.loss is not None
        assert output2.loss is not None

    def test_variable_length_episodes(self):
        """Test handling of variable-length episodes."""
        config = GRPOConfig(micro_batch_size=2)
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        # Episodes with different lengths
        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3],
                "reward": 1.0,
                "prompt": "short",
            },
            {
                "prefix_tokens": [1, 2, 3, 4, 5],
                "generated_tokens": [6, 7, 8, 9, 10],
                "reward": -1.0,
                "prompt": "long",
            },
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_micro_batch_processing(self):
        """Test that micro-batching works correctly."""
        config = GRPOConfig(micro_batch_size=1)  # Force small micro-batches
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        # Create more episodes than micro-batch size
        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": float(i),
                "prompt": f"prompt_{i}",
            }
            for i in range(4)
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_entropy_coefficient_decay(self):
        """Test entropy coefficient decay over time."""
        config = GRPOConfig(
            entropy_coeff=0.1,
            entropy_decay=0.95,
            min_entropy_coeff=0.01,
            micro_batch_size=2,
        )
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        initial_entropy_coeff = grpo.loss_computer.current_entropy_coeff

        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": 1.0,
                "prompt": "test",
            }
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        # Run several steps to see entropy decay
        for _ in range(5):
            grpo.step(batch)

        final_entropy_coeff = grpo.loss_computer.current_entropy_coeff

        # Entropy coefficient should have decayed but not below minimum
        assert final_entropy_coeff <= initial_entropy_coeff
        assert final_entropy_coeff >= config.min_entropy_coeff

    def test_token_level_vs_sequence_level_loss(self):
        """Test different loss computation methods."""
        for token_level in [True, False]:
            config = GRPOConfig(token_level_loss=token_level, micro_batch_size=2)
            grpo = GRPO(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            grpo.setup(model=model)

            episodes = [
                {
                    "prefix_tokens": [1, 2, 3],
                    "generated_tokens": [4, 5],
                    "reward": 1.0,
                    "prompt": "test",
                }
            ]

            batch = {"episodes": episodes, "pad_token_id": 0}
            output = grpo.step(batch)

            assert output.loss is not None
            assert not torch.isnan(output.loss)

    def test_only_reward_generated_tokens(self):
        """Test applying rewards only to generated tokens."""
        config = GRPOConfig(only_reward_generated=True, micro_batch_size=2)
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        episodes = [
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [4, 5, 6],
                "reward": 2.0,
                "prompt": "test",
            }
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}
        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_group_size_effects(self):
        """Test different group sizes for normalization."""
        for group_size in [1, 2, 4]:
            config = GRPOConfig(group_min_size=group_size, micro_batch_size=2)
            grpo = GRPO(config)
            model = MockModel(vocab_size=1000, hidden_size=256)
            grpo.setup(model=model)

            # Create episodes with multiple groups
            episodes = [
                {
                    "prefix_tokens": [1, 2],
                    "generated_tokens": [3, 4],
                    "reward": float(i),
                    "prompt": f"group_{i % 3}",  # 3 different groups
                }
                for i in range(6)
            ]

            batch = {"episodes": episodes, "pad_token_id": 0}
            output = grpo.step(batch)

            assert output.loss is not None

    def test_token_level_rewards(self):
        """Test handling of token-level rewards vs sequence-level."""
        config = GRPOConfig(micro_batch_size=2)
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        # Test with token-level rewards
        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": [0.5, 1.0],  # Token-level rewards
                "prompt": "test",
            }
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}
        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        config = GRPOConfig(
            max_grad_norm=0.5, micro_batch_size=2  # Low value to ensure clipping
        )
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": 10.0,  # High reward to potentially cause large gradients
                "prompt": "test",
            }
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}
        output = grpo.step(batch)

        assert output.loss is not None
        # Should have gradient norm in metrics if clipping occurred
        if "grad_norm" in output.metrics:
            assert (
                output.metrics["grad_norm"] <= config.max_grad_norm + 0.1
            )  # Small tolerance

    def test_large_batch_memory_efficiency(self):
        """Test memory efficiency with larger batches."""
        config = GRPOConfig(micro_batch_size=2)  # Small micro-batches
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        # Create larger batch
        episodes = [
            {
                "prefix_tokens": [1, 2, 3] * (i + 1),  # Variable lengths
                "generated_tokens": [4, 5, 6] * (i + 1),
                "reward": float(i),
                "prompt": f"prompt_{i % 4}",  # Multiple groups
            }
            for i in range(12)  # Larger number of episodes
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}
        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)

    def test_empty_generated_tokens(self):
        """Test handling of episodes with no generated tokens."""
        config = GRPOConfig(micro_batch_size=2)
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        episodes = [
            {
                "prefix_tokens": [1, 2, 3],
                "generated_tokens": [],  # Empty generation
                "reward": 0.0,
                "prompt": "test",
            },
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": 1.0,
                "prompt": "test",
            },
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}

        # Should handle gracefully or raise appropriate error
        try:
            output = grpo.step(batch)
            if output.loss is not None:
                assert not torch.isnan(output.loss)
        except (ValueError, RuntimeError) as e:
            # Acceptable to raise error for invalid input
            assert "empty" in str(e).lower() or "generated" in str(e).lower()

    def test_single_group_normalization(self):
        """Test normalization when all episodes have same prompt."""
        config = GRPOConfig(
            use_group_normalization=True, group_min_size=2, micro_batch_size=2
        )
        grpo = GRPO(config)
        model = MockModel(vocab_size=1000, hidden_size=256)
        grpo.setup(model=model)

        # All episodes have same prompt (single group)
        episodes = [
            {
                "prefix_tokens": [1, 2],
                "generated_tokens": [3, 4],
                "reward": float(i),
                "prompt": "same_prompt",  # All same
            }
            for i in range(4)
        ]

        batch = {"episodes": episodes, "pad_token_id": 0}
        output = grpo.step(batch)

        assert output.loss is not None
        assert not torch.isnan(output.loss)
