"""
Tests for LoRA Configuration and Utilities
==========================================

Comprehensive tests for PEFT/LoRA integration.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from thinkrl.peft.lora import (
    LoRAConfig,
    ARCHITECTURE_TARGETS,
    is_peft_available,
    get_trainable_parameters,
    get_lora_config_for_model,
)


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_initialization(self):
        """Test default LoRA config initialization."""
        config = LoRAConfig()

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.target_modules is None
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

    def test_custom_initialization(self):
        """Test custom LoRA config initialization."""
        config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="all",
        )

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "all"

    def test_for_architecture_llama(self):
        """Test LoRA config for Llama architecture."""
        config = LoRAConfig.for_architecture("llama")

        assert config.target_modules is not None
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_for_architecture_qwen(self):
        """Test LoRA config for Qwen architecture."""
        config = LoRAConfig.for_architecture("qwen")

        assert config.target_modules is not None
        assert "c_attn" in config.target_modules

    def test_for_architecture_mistral(self):
        """Test LoRA config for Mistral architecture."""
        config = LoRAConfig.for_architecture("mistral")

        assert config.target_modules is not None
        assert "q_proj" in config.target_modules

    def test_for_architecture_unknown(self):
        """Test LoRA config for unknown architecture uses default."""
        config = LoRAConfig.for_architecture("unknown_arch")

        assert config.target_modules is not None
        # Should use default targets
        assert config.target_modules == ARCHITECTURE_TARGETS["default"]

    def test_for_llama_shortcut(self):
        """Test for_llama shortcut method."""
        config = LoRAConfig.for_llama(r=16)

        assert config.r == 16
        assert config.lora_alpha == 32  # 2 * r
        assert "q_proj" in config.target_modules

    def test_for_qwen_shortcut(self):
        """Test for_qwen shortcut method."""
        config = LoRAConfig.for_qwen(r=8)

        assert config.r == 8
        assert "c_attn" in config.target_modules

    def test_for_qwen2_shortcut(self):
        """Test for_qwen2 shortcut method."""
        config = LoRAConfig.for_qwen2(r=8)

        assert config.r == 8
        assert "q_proj" in config.target_modules

    def test_for_mistral_shortcut(self):
        """Test for_mistral shortcut method."""
        config = LoRAConfig.for_mistral(r=8)

        assert config.r == 8
        assert "q_proj" in config.target_modules

    def test_alpha_is_2x_rank(self):
        """Test that alpha is set to 2x rank by default in for_architecture."""
        config = LoRAConfig.for_architecture("llama", r=8)
        assert config.lora_alpha == 16

        config = LoRAConfig.for_architecture("llama", r=16)
        assert config.lora_alpha == 32

    def test_custom_kwargs_override(self):
        """Test custom kwargs override defaults."""
        config = LoRAConfig.for_architecture(
            "llama",
            r=16,
            lora_dropout=0.2,
            use_rslora=True,
        )

        assert config.r == 16
        assert config.lora_dropout == 0.2
        assert config.use_rslora is True


class TestArchitectureTargets:
    """Tests for architecture target modules."""

    def test_all_architectures_have_targets(self):
        """Test all defined architectures have target modules."""
        for arch, targets in ARCHITECTURE_TARGETS.items():
            assert isinstance(targets, list)
            assert len(targets) > 0

    def test_llama_targets(self):
        """Test Llama target modules."""
        targets = ARCHITECTURE_TARGETS["llama"]

        expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for target in expected:
            assert target in targets

    def test_default_targets_exist(self):
        """Test default targets exist."""
        assert "default" in ARCHITECTURE_TARGETS
        assert len(ARCHITECTURE_TARGETS["default"]) > 0


class TestIsPeftAvailable:
    """Tests for is_peft_available function."""

    def test_returns_boolean(self):
        """Test that is_peft_available returns a boolean."""
        result = is_peft_available()
        assert isinstance(result, bool)


class TestGetTrainableParameters:
    """Tests for get_trainable_parameters function."""

    def test_simple_model(self):
        """Test with a simple model."""
        model = nn.Linear(10, 5)
        result = get_trainable_parameters(model)

        assert "trainable" in result
        assert "frozen" in result
        assert "total" in result
        assert "trainable_percent" in result

    def test_all_trainable(self):
        """Test model with all trainable parameters."""
        model = nn.Linear(10, 5)
        result = get_trainable_parameters(model)

        assert result["trainable"] == result["total"]
        assert result["frozen"] == 0
        assert result["trainable_percent"] == 100.0

    def test_all_frozen(self):
        """Test model with all frozen parameters."""
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        result = get_trainable_parameters(model)

        assert result["trainable"] == 0
        assert result["frozen"] == result["total"]
        assert result["trainable_percent"] == 0.0

    def test_mixed_parameters(self):
        """Test model with mixed trainable/frozen parameters."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        result = get_trainable_parameters(model)

        assert result["trainable"] > 0
        assert result["frozen"] > 0
        assert result["trainable"] + result["frozen"] == result["total"]


class TestGetLoRAConfigForModel:
    """Tests for get_lora_config_for_model function."""

    def test_with_model_type_attribute(self):
        """Test with model that has model_type attribute."""
        # Mock model with config
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"

        config = get_lora_config_for_model(mock_model, r=8)

        assert config.r == 8
        assert config.target_modules is not None

    def test_with_architectures_attribute(self):
        """Test with model that has architectures attribute."""
        mock_model = MagicMock()
        mock_model.config.model_type = None
        mock_model.config.architectures = ["LlamaForCausalLM"]

        config = get_lora_config_for_model(mock_model, r=8)

        assert config.r == 8

    def test_without_config(self):
        """Test with model without config attribute."""
        model = nn.Linear(10, 5)
        config = get_lora_config_for_model(model, r=8)

        assert config.r == 8
        # Should use default targets
        assert config.target_modules is None or len(config.target_modules) > 0

    def test_custom_kwargs(self):
        """Test passing custom kwargs."""
        model = nn.Linear(10, 5)
        config = get_lora_config_for_model(
            model,
            r=16,
            lora_dropout=0.2,
        )

        assert config.r == 16
        assert config.lora_dropout == 0.2


class TestLoRAConfigToPeftConfig:
    """Tests for to_peft_config method."""

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT not installed")
    def test_to_peft_config(self):
        """Test conversion to peft config."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        peft_config = config.to_peft_config()

        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16
        assert peft_config.lora_dropout == 0.05

    def test_to_peft_config_without_peft(self):
        """Test to_peft_config raises when PEFT not available."""
        if is_peft_available():
            pytest.skip("PEFT is available")

        config = LoRAConfig()

        with pytest.raises(ImportError):
            config.to_peft_config()


class TestLoRAConfigIntegration:
    """Integration tests for LoRA configuration."""

    def test_config_serialization(self):
        """Test config can be serialized to dict."""
        from dataclasses import asdict

        config = LoRAConfig.for_llama(r=8)
        config_dict = asdict(config)

        assert config_dict["r"] == 8
        assert isinstance(config_dict["target_modules"], list)

    def test_multiple_architectures_have_unique_targets(self):
        """Test different architectures have appropriate targets."""
        llama_config = LoRAConfig.for_architecture("llama")
        gpt2_config = LoRAConfig.for_architecture("gpt2")

        # Llama uses q_proj, gpt2 uses c_attn
        assert "q_proj" in llama_config.target_modules
        assert "c_attn" in gpt2_config.target_modules
