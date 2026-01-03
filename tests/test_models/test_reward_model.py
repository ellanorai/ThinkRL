from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.models.reward_model import RewardModel


class MockConfig:
    def __init__(self):
        self.hidden_size = 100


class MockModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        batch_size, seq_len = input_ids.shape
        # Return hidden states
        hidden_states = torch.randn(batch_size, seq_len, 100)
        return MockModelOutput(hidden_states)

    def gradient_checkpointing_enable(self, kwargs):
        pass

    def gradient_checkpointing_disable(self):
        pass

    def save_pretrained(self, path, **kwargs):
        pass


@pytest.fixture
def mock_transformers():
    with patch("thinkrl.models.reward_model.AutoConfig") as MockAutoConfig:
        with patch("thinkrl.models.reward_model.AutoModel") as MockAutoModel:
            MockAutoModel.from_pretrained.return_value = MockModel()
            MockAutoConfig.from_pretrained.return_value = MockConfig()
            yield MockAutoModel


class TestRewardModel:
    def test_init_with_module(self):
        model = MockModel()
        rm = RewardModel(model)
        assert rm.model is model
        assert rm.lora_rank == 0
        assert isinstance(rm.reward_head, nn.Linear)

    def test_init_with_string(self, mock_transformers):
        rm = RewardModel("gpt2")
        message = mock_transformers.from_pretrained.call_args[0][0]
        assert message == "gpt2"

    def test_init_with_normalization(self):
        model = MockModel()
        rm = RewardModel(model, normalize_reward=True)
        assert hasattr(rm, "reward_mean")
        assert hasattr(rm, "reward_std")

    def test_init_with_lora(self):
        with patch("thinkrl.models.reward_model._PEFT_AVAILABLE", True):
            with patch("thinkrl.models.reward_model.get_peft_model", create=True) as mock_get_peft:
                with patch("thinkrl.models.reward_model.LoraConfig", create=True) as MockLoraConfig:
                    with patch("thinkrl.models.reward_model.TaskType", create=True) as MockTaskType:
                        model = MockModel()
                        # Mock named_parameters for bf16 cast loop
                        model.named_parameters = MagicMock(return_value=[("layer.weight", torch.randn(10))])

                        MockTaskType.SEQ_CLS = "SEQ_CLS"

                        mock_get_peft.return_value = model

                        rm = RewardModel(model, lora_rank=8, bf16=True)

                        assert rm.lora_rank == 8
                        mock_get_peft.assert_called()

    def test_forward(self):
        model = MockModel()
        rm = RewardModel(model)
        input_ids = torch.randint(0, 100, (2, 10))

        # Test basic forward
        rewards = rm(input_ids)
        # output shape is (batch)
        assert rewards.shape == (2,)

        # Test return_output
        rewards, output = rm(input_ids, return_output=True)
        assert output is not None

    def test_forward_with_normalization(self):
        model = MockModel()
        rm = RewardModel(model, normalize_reward=True)
        rm.reward_mean = torch.tensor([0.0])
        rm.reward_std = torch.tensor([1.0])
        rm.training = False  # normalization applied during inference

        input_ids = torch.randint(0, 100, (2, 10))
        rewards = rm(input_ids)
        assert rewards.shape == (2,)

    def test_compute_pairwise_rewards(self):
        model = MockModel()
        rm = RewardModel(model)
        chosen_ids = torch.randint(0, 100, (2, 10))
        chosen_mask = torch.ones(2, 10, dtype=torch.long)
        rejected_ids = torch.randint(0, 100, (2, 10))
        rejected_mask = torch.ones(2, 10, dtype=torch.long)

        chosen, rejected = rm.compute_pairwise_rewards(chosen_ids, chosen_mask, rejected_ids, rejected_mask)
        assert chosen.shape == (2,)
        assert rejected.shape == (2,)

    def test_update_normalization(self):
        model = MockModel()
        rm = RewardModel(model, normalize_reward=True)

        rewards = torch.randn(10)
        rm.update_normalization(rewards)

        assert rm.reward_mean != 0.0
        assert rm.reward_std != 1.0

    def test_gradient_checkpointing(self):
        model = MockModel()
        rm = RewardModel(model)
        rm.model.gradient_checkpointing_enable = MagicMock()
        rm.gradient_checkpointing_enable()
        rm.model.gradient_checkpointing_enable.assert_called()

    def test_save_pretrained(self):
        model = MockModel()
        rm = RewardModel(model, normalize_reward=True)
        rm.model.save_pretrained = MagicMock()
        with patch("torch.save") as mock_save:
            rm.save_pretrained("path")
            rm.model.save_pretrained.assert_called_with("path")
            mock_save.assert_called()

    def test_from_pretrained(self, mock_transformers):
        with patch("os.path.exists", return_value=True):
            with patch("torch.load") as mock_load:
                mock_load.return_value = {
                    "reward_head": {},
                    "reward_mean": torch.tensor([0.0]),
                    "reward_std": torch.tensor([1.0]),
                }
                # We need to mock load_state_dict since we return empty dict
                with patch.object(nn.Linear, "load_state_dict") as mock_load_sd:
                    rm = RewardModel.from_pretrained("gpt2")
                    assert isinstance(rm, RewardModel)
                    assert hasattr(rm, "reward_mean")

    def test_init_transformers_missing(self):
        with patch("thinkrl.models.reward_model._TRANSFORMERS_AVAILABLE", False):
            with pytest.raises(ImportError):
                RewardModel(MockModel())
