from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.models.critic import Critic


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
    with patch("thinkrl.models.critic.AutoConfig") as MockAutoConfig:
        with patch("thinkrl.models.critic.AutoModel") as MockAutoModel:
            MockAutoModel.from_pretrained.return_value = MockModel()
            MockAutoConfig.from_pretrained.return_value = MockConfig()
            yield MockAutoModel


class TestCritic:
    def test_init_with_module(self):
        model = MockModel()
        critic = Critic(model)
        assert critic.model is model
        assert critic.lora_rank == 0
        assert isinstance(critic.value_head, nn.Linear)

    def test_init_with_string(self, mock_transformers):
        critic = Critic("gpt2")
        message = mock_transformers.from_pretrained.call_args[0][0]
        assert message == "gpt2"

    def test_init_with_lora(self):
        with patch("thinkrl.models.critic._PEFT_AVAILABLE", True):
            with patch("thinkrl.models.critic.get_peft_model", create=True) as mock_get_peft:
                with patch("thinkrl.models.critic.LoraConfig", create=True) as MockLoraConfig:
                    with patch("thinkrl.models.critic.TaskType", create=True) as MockTaskType:
                        model = MockModel()
                        # Mock named_parameters for bf16 cast loop
                        model.named_parameters = MagicMock(return_value=[("layer.weight", torch.randn(10))])

                        MockTaskType.SEQ_CLS = "SEQ_CLS"

                        mock_get_peft.return_value = model

                        critic = Critic(model, lora_rank=8, bf16=True)

                        assert critic.lora_rank == 8
                        mock_get_peft.assert_called()

    def test_forward(self):
        model = MockModel()
        critic = Critic(model)
        input_ids = torch.randint(0, 100, (2, 10))

        # Test basic forward
        values = critic(input_ids)
        # output shape is (batch, seq_len-1)
        assert values.shape == (2, 9)

        # Test return_output
        values, output = critic(input_ids, return_output=True)
        assert output is not None

    def test_forward_action_mask(self):
        model = MockModel()
        critic = Critic(model)
        input_ids = torch.randint(0, 100, (2, 10))
        action_mask = torch.ones(2, 10)

        values = critic(input_ids, action_mask=action_mask)
        assert values.shape == (2, 9)

    def test_get_value_at_position(self):
        model = MockModel()
        critic = Critic(model)
        input_ids = torch.randint(0, 100, (2, 10))

        # Test last position (default)
        values = critic.get_value_at_position(input_ids)
        assert values.shape == (2,)

        # Test specific position
        values_pos = critic.get_value_at_position(input_ids, position=0)
        assert values_pos.shape == (2,)

        # Test with attention mask
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[0, -1] = 0  # first seq len 9
        values_mask = critic.get_value_at_position(input_ids, attention_mask=attention_mask)
        assert values_mask.shape == (2,)

    def test_gradient_checkpointing(self):
        model = MockModel()
        critic = Critic(model)
        critic.model.gradient_checkpointing_enable = MagicMock()
        critic.gradient_checkpointing_enable()
        critic.model.gradient_checkpointing_enable.assert_called()

    def test_save_pretrained(self):
        model = MockModel()
        critic = Critic(model)
        critic.model.save_pretrained = MagicMock()
        with patch("torch.save") as mock_save:
            critic.save_pretrained("path")
            critic.model.save_pretrained.assert_called_with("path")
            mock_save.assert_called()

    def test_from_pretrained(self, mock_transformers):
        with patch("os.path.exists", return_value=True):
            with patch("torch.load") as mock_load:
                mock_load.return_value = {"value_head": {}}
                # We need to mock load_state_dict since we return empty dict for value_head
                with patch.object(nn.Linear, "load_state_dict") as mock_load_sd:
                    critic = Critic.from_pretrained("gpt2")
                    assert isinstance(critic, Critic)
                    mock_load.assert_called()

    def test_init_transformers_missing(self):
        with patch("thinkrl.models.critic._TRANSFORMERS_AVAILABLE", False):
            with pytest.raises(ImportError):
                Critic(MockModel())
