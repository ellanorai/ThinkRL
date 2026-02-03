"""
Tests for DeepSpeed Strategies
==============================

Comprehensive tests for ZeRO optimization strategies.
"""

import pytest

from thinkrl.distributed.strategies import (
    STRATEGIES,
    DeepSpeedStrategy,
    ZeRO1Strategy,
    ZeRO2Strategy,
    ZeRO3Strategy,
    get_strategy,
)


class TestZeRO1Strategy:
    """Tests for ZeRO Stage 1 strategy."""

    def test_default_initialization(self):
        """Test default ZeRO1 strategy initialization."""
        strategy = ZeRO1Strategy()

        assert strategy.gradient_accumulation_steps == 1
        assert strategy.gradient_clipping == 1.0
        assert strategy.train_micro_batch_size_per_gpu == 4
        assert strategy.bf16_enabled is True
        assert strategy.fp16_enabled is False

    def test_custom_initialization(self):
        """Test ZeRO1 with custom parameters."""
        strategy = ZeRO1Strategy(
            gradient_accumulation_steps=4,
            gradient_clipping=0.5,
            train_micro_batch_size_per_gpu=8,
            bf16_enabled=False,
            fp16_enabled=True,
        )

        assert strategy.gradient_accumulation_steps == 4
        assert strategy.gradient_clipping == 0.5
        assert strategy.train_micro_batch_size_per_gpu == 8
        assert strategy.bf16_enabled is False
        assert strategy.fp16_enabled is True

    def test_get_zero_config(self):
        """Test ZeRO config generation."""
        strategy = ZeRO1Strategy()
        config = strategy.get_zero_config()

        assert config["stage"] == 1
        assert "reduce_scatter" in config
        assert "overlap_comm" in config

    def test_get_config_with_bf16(self):
        """Test full config with BF16."""
        strategy = ZeRO1Strategy(bf16_enabled=True)
        config = strategy.get_config()

        assert "bf16" in config
        assert config["bf16"]["enabled"] is True
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 1

    def test_get_config_with_fp16(self):
        """Test full config with FP16."""
        strategy = ZeRO1Strategy(bf16_enabled=False, fp16_enabled=True)
        config = strategy.get_config()

        assert "fp16" in config
        assert config["fp16"]["enabled"] is True
        assert "loss_scale" in config["fp16"]

    def test_activation_checkpointing(self):
        """Test activation checkpointing config."""
        strategy = ZeRO1Strategy(
            activation_checkpointing=True,
            partition_activations=True,
        )
        config = strategy.get_config()

        assert "activation_checkpointing" in config
        assert config["activation_checkpointing"]["partition_activations"] is True


class TestZeRO2Strategy:
    """Tests for ZeRO Stage 2 strategy."""

    def test_default_initialization(self):
        """Test default ZeRO2 strategy initialization."""
        strategy = ZeRO2Strategy()

        assert strategy.overlap_comm is True
        assert strategy.reduce_scatter is True
        assert strategy.offload_optimizer is False
        assert strategy.contiguous_gradients is True

    def test_get_zero_config_without_offload(self):
        """Test ZeRO2 config without offloading."""
        strategy = ZeRO2Strategy(offload_optimizer=False)
        config = strategy.get_zero_config()

        assert config["stage"] == 2
        assert "offload_optimizer" not in config
        assert config["overlap_comm"] is True

    def test_get_zero_config_with_offload(self):
        """Test ZeRO2 config with optimizer offloading."""
        strategy = ZeRO2Strategy(
            offload_optimizer=True,
            offload_optimizer_device="cpu",
            offload_optimizer_pin_memory=True,
        )
        config = strategy.get_zero_config()

        assert config["stage"] == 2
        assert "offload_optimizer" in config
        assert config["offload_optimizer"]["device"] == "cpu"
        assert config["offload_optimizer"]["pin_memory"] is True

    def test_bucket_sizes(self):
        """Test bucket size configuration."""
        strategy = ZeRO2Strategy(
            reduce_bucket_size=1_000_000_000,
            allgather_bucket_size=1_000_000_000,
        )
        config = strategy.get_zero_config()

        assert config["reduce_bucket_size"] == 1_000_000_000
        assert config["allgather_bucket_size"] == 1_000_000_000


class TestZeRO3Strategy:
    """Tests for ZeRO Stage 3 strategy."""

    def test_default_initialization(self):
        """Test default ZeRO3 strategy initialization."""
        strategy = ZeRO3Strategy()

        assert strategy.offload_optimizer is True
        assert strategy.offload_param is False
        assert strategy.stage3_gather_16bit_weights_on_model_save is True

    def test_get_zero_config_basic(self):
        """Test basic ZeRO3 config."""
        strategy = ZeRO3Strategy()
        config = strategy.get_zero_config()

        assert config["stage"] == 3
        assert "stage3_prefetch_bucket_size" in config
        assert "stage3_param_persistence_threshold" in config
        assert "stage3_max_live_parameters" in config

    def test_get_zero_config_full_offload(self):
        """Test ZeRO3 with full offloading."""
        strategy = ZeRO3Strategy(
            offload_optimizer=True,
            offload_param=True,
        )
        config = strategy.get_zero_config()

        assert "offload_optimizer" in config
        assert "offload_param" in config
        assert config["offload_param"]["device"] == "cpu"

    def test_subgroup_size(self):
        """Test sub_group_size configuration."""
        strategy = ZeRO3Strategy(sub_group_size=500_000_000)
        config = strategy.get_zero_config()

        assert config["sub_group_size"] == 500_000_000


class TestGetStrategy:
    """Tests for get_strategy factory function."""

    def test_get_zero1_strategy(self):
        """Test getting ZeRO1 strategy."""
        strategy = get_strategy("zero1")
        assert isinstance(strategy, ZeRO1Strategy)

    def test_get_zero2_strategy(self):
        """Test getting ZeRO2 strategy."""
        strategy = get_strategy("zero2")
        assert isinstance(strategy, ZeRO2Strategy)

    def test_get_zero3_strategy(self):
        """Test getting ZeRO3 strategy."""
        strategy = get_strategy("zero3")
        assert isinstance(strategy, ZeRO3Strategy)

    def test_get_zero2_offload_preset(self):
        """Test getting ZeRO2 with offload preset."""
        strategy = get_strategy("zero2_offload")
        assert isinstance(strategy, ZeRO2Strategy)
        assert strategy.offload_optimizer is True

    def test_get_zero3_offload_preset(self):
        """Test getting ZeRO3 with offload preset."""
        strategy = get_strategy("zero3_offload")
        assert isinstance(strategy, ZeRO3Strategy)
        assert strategy.offload_optimizer is True
        assert strategy.offload_param is True

    def test_get_strategy_with_kwargs(self):
        """Test getting strategy with custom kwargs."""
        strategy = get_strategy("zero2", gradient_accumulation_steps=8)
        assert strategy.gradient_accumulation_steps == 8

    def test_get_strategy_case_insensitive(self):
        """Test case insensitivity."""
        strategy = get_strategy("ZERO2")
        assert isinstance(strategy, ZeRO2Strategy)

    def test_get_strategy_invalid(self):
        """Test invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("zero4")

    def test_strategies_registry(self):
        """Test STRATEGIES registry contains all strategies."""
        assert "zero1" in STRATEGIES
        assert "zero2" in STRATEGIES
        assert "zero3" in STRATEGIES
        assert "zero2_offload" in STRATEGIES
        assert "zero3_offload" in STRATEGIES


class TestStrategyIntegration:
    """Integration tests for strategies."""

    def test_config_is_valid_json_serializable(self):
        """Test that generated config is JSON serializable."""
        import json

        for name in ["zero1", "zero2", "zero3"]:
            strategy = get_strategy(name)
            config = strategy.get_config()
            # Should not raise
            json_str = json.dumps(config)
            assert isinstance(json_str, str)

    def test_all_strategies_have_required_keys(self):
        """Test all strategies produce valid configs."""
        required_keys = [
            "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "gradient_clipping",
            "zero_optimization",
        ]

        for name in ["zero1", "zero2", "zero3"]:
            strategy = get_strategy(name)
            config = strategy.get_config()

            for key in required_keys:
                assert key in config, f"Missing key {key} in {name} config"
