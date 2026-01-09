from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.models.model import (
    compute_model_size,
    create_reference_model,
    get_actor_model,
    get_llm_for_sequence_regression,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.param = nn.Parameter(torch.randn(10))


class ModelConfig:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path


@pytest.fixture
def model_config():
    return ModelConfig(name_or_path="meta-llama/Llama-2-7b-hf")


@pytest.fixture
def mock_classes():
    # Since imports happen inside functions, we patch the classes in their definition modules
    with patch("thinkrl.models.reward_model.RewardModel") as MockRewardModel, patch(
        "thinkrl.models.critic.Critic"
    ) as MockCritic, patch("thinkrl.models.actor.Actor") as MockActor:
        MockRewardModel.return_value = MagicMock(spec=nn.Module)
        MockCritic.return_value = MagicMock(spec=nn.Module)

        # Setup Actor mock to behave like a module for modification usage
        actor_instance = MockModel()
        MockActor.return_value = actor_instance

        yield MockRewardModel, MockCritic, MockActor


def test_get_llm_for_sequence_regression_reward(mock_classes):
    MockRewardModel, _, _ = mock_classes

    _ = get_llm_for_sequence_regression(
        "meta-llama/Llama-2-7b-hf", model_type="reward", lora_rank=8, normalize_reward=True
    )

    MockRewardModel.assert_called_once()
    call_kwargs = MockRewardModel.call_args[1]
    assert call_kwargs["pretrained_model"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["lora_rank"] == 8
    assert call_kwargs["normalize_reward"] is True


def test_get_llm_for_sequence_regression_critic(mock_classes):
    _, MockCritic, _ = mock_classes

    _ = get_llm_for_sequence_regression("meta-llama/Llama-2-7b-hf", model_type="critic", use_flash_attention=False)

    MockCritic.assert_called_once()
    call_kwargs = MockCritic.call_args[1]
    assert call_kwargs["pretrained_model"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["use_flash_attention"] is False


def test_get_llm_for_sequence_regression_invalid():
    with pytest.raises(ValueError, match="Unknown model_type"):
        get_llm_for_sequence_regression("meta-llama/Llama-2-7b-hf", model_type="invalid")


def test_get_actor_model(mock_classes):
    _, _, MockActor = mock_classes

    _ = get_actor_model("meta-llama/Llama-2-7b-hf", lora_rank=16, bf16=False)

    MockActor.assert_called_once()
    call_kwargs = MockActor.call_args[1]
    assert call_kwargs["pretrained_model"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["lora_rank"] == 16
    assert call_kwargs["bf16"] is False


def test_create_reference_model(mock_classes):
    _, _, MockActor = mock_classes

    # Verify lora_rank is forced to 0 and model is frozen
    ref_model = create_reference_model("meta-llama/Llama-2-7b-hf")

    MockActor.assert_called_once()
    call_kwargs = MockActor.call_args[1]
    assert call_kwargs["pretrained_model"] == "meta-llama/Llama-2-7b-hf"
    assert call_kwargs["lora_rank"] == 0

    # Check if parameters are frozen
    for param in ref_model.parameters():
        assert param.requires_grad is False

    # Check eval mode
    assert not ref_model.training


def test_compute_model_size():
    model = nn.Sequential(
        nn.Linear(10, 10),  # 110 params
        nn.Linear(10, 5),  # 55 params
    )
    # Total = 165

    # Freeze second layer
    for param in model[1].parameters():
        param.requires_grad = False

    stats = compute_model_size(model)

    assert stats["total_params"] == 165
    assert stats["trainable_params"] == 110
    assert stats["frozen_params"] == 55
    assert abs(stats["trainable_percent"] - (110 / 165) * 100) < 1e-5


def test_compute_model_size_empty():
    model = nn.Sequential()
    stats = compute_model_size(model)
    assert stats["total_params"] == 0
    assert stats["trainable_params"] == 0
    assert stats["trainable_percent"] == 0
