"""
Tests for LoRA Configuration and Utilities
==========================================

Comprehensive tests for PEFT/LoRA integration.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.peft.lora import (
    ARCHITECTURE_TARGETS,
    LoRAConfig,
    get_lora_config_for_model,
    get_trainable_parameters,
    is_peft_available,
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
        for _, targets in ARCHITECTURE_TARGETS.items():
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
        bloom_config = LoRAConfig.for_architecture("bloom")

        # Llama uses q_proj, bloom uses query_key_value
        assert "q_proj" in llama_config.target_modules
        assert "query_key_value" in bloom_config.target_modules


class TestLoRAUtilities:
    """Tests for LoRA utility functions."""

    def test_inject_lora_peft_missing(self):
        """Test inject_lora raises ImportError when PEFT is missing."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError, match="peft library is required"):
                from thinkrl.peft.lora import inject_lora

                inject_lora(MagicMock(), MagicMock())

    def test_merge_lora_weights_peft_missing(self):
        """Test merge_lora_weights raises ImportError when PEFT is missing."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError, match="peft library is required"):
                from thinkrl.peft.lora import merge_lora_weights

                merge_lora_weights(MagicMock())

    def test_merge_lora_weights_no_method(self):
        """Test merge_lora_weights returns model as-is if no merge_and_unload."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", True):
            model = MagicMock()
            del model.merge_and_unload  # Ensure attribute doesn't exist

            # Need to patch hasattr to return False for merge_and_unload
            # Or just rely on MagicMock behavior if we didn't add it.
            # MagicMock usually creates attributes on access.
            # We can use a real object or spec.
            class SimpleModel:
                pass

            model = SimpleModel()

            from thinkrl.peft.lora import merge_lora_weights

            result = merge_lora_weights(model)
            assert result is model

    def test_unload_lora_peft_missing(self):
        """Test unload_lora raises ImportError when PEFT is missing."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", False):
            with pytest.raises(ImportError, match="peft library is required"):
                from thinkrl.peft.lora import unload_lora

                unload_lora(MagicMock())

    def test_unload_lora_fallback(self):
        """Test unload_lora fallback when unload() is missing."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", True):
            # Model with base_model but no unload
            class MockPeftModel:
                def __init__(self):
                    self.base_model = MagicMock()
                    self.base_model.model = "inner_model"

            model = MockPeftModel()
            from thinkrl.peft.lora import unload_lora

            result = unload_lora(model)
            assert result == "inner_model"

    def test_prepare_model_for_deepspeed_lora_zero3(self):
        """Test prepare_model_for_deepspeed_lora with ZeRO-3."""
        with patch("thinkrl.peft.lora.inject_lora") as mock_inject:
            mock_inject.return_value = "injected_model"
            from thinkrl.peft.lora import prepare_model_for_deepspeed_lora

            result = prepare_model_for_deepspeed_lora(MagicMock(), MagicMock(), zero_stage=3)
            assert result == "injected_model"


class TestLoRAConfigCoverage:
    """Extra coverage tests for LoRAConfig."""

    def test_all_architectures_have_targets(self):
        """Test all defined architectures have target modules."""
        from thinkrl.peft.lora import ARCHITECTURE_TARGETS

        for _, targets in ARCHITECTURE_TARGETS.items():
            assert isinstance(targets, list)
            assert len(targets) > 0

    def test_task_types(self):
        """Test mapping of different task types."""
        with patch("thinkrl.peft.lora.PEFT_AVAILABLE", True):
            with patch("thinkrl.peft.lora.PeftLoraConfig") as _:
                with patch("thinkrl.peft.lora.TaskType") as MockTaskType:
                    # Setup MockTaskType attributes
                    MockTaskType.CAUSAL_LM = "CAUSAL_LM"
                    MockTaskType.SEQ_CLS = "SEQ_CLS"
                    MockTaskType.SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
                    MockTaskType.TOKEN_CLS = "TOKEN_CLS"

                    types = ["SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"]
                    for t in types:
                        config = LoRAConfig(task_type=t)
                        config.to_peft_config()

    def test_architecture_substring_match(self):
        """Test fuzzy matching for architecture."""
        # "mistral-7b" should match "mistral"
        config = LoRAConfig.for_architecture("mistral-7b-v0.1")
        assert "q_proj" in config.target_modules

    def test_get_lora_config_architectures_list(self):
        """Test getting config from model.config.architectures list."""
        from thinkrl.peft.lora import get_lora_config_for_model

        model = MagicMock()
        del model.config.model_type  # Ensure fallback to architectures
        model.config.architectures = ["MistralForCausalLM"]

        config = get_lora_config_for_model(model)
        assert "q_proj" in config.target_modules
