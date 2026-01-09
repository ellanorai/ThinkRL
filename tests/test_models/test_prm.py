import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import AutoConfig

from thinkrl.models.prm import PRMConfig, ProcessRewardModel, create_prm


class TestProcessRewardModel(unittest.TestCase):
    def setUp(self):
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.config = PRMConfig(model_name_or_path=self.model_name)

    @patch("thinkrl.models.prm.AutoModel")
    @patch("thinkrl.models.prm.AutoConfig")
    def test_init(self, mock_config, mock_model):
        """Test PRM initialization."""
        mock_config_inst = MagicMock()
        mock_config_inst.hidden_size = 768
        mock_config.from_pretrained.return_value = mock_config_inst

        mock_model_inst = MagicMock()
        mock_model_inst.config.hidden_size = 768
        mock_model.from_pretrained.return_value = mock_model_inst

        prm = ProcessRewardModel(config=self.config)

        self.assertIsInstance(prm.score_head, nn.Linear)
        self.assertEqual(prm.score_head.in_features, 768)
        self.assertEqual(prm.score_head.out_features, 1)

    @patch("thinkrl.models.prm.AutoModel")
    @patch("thinkrl.models.prm.AutoConfig")
    def test_forward_dense(self, mock_config, mock_model):
        """Test forward pass without step positions (dense)."""
        # Setup mocks
        mock_config_inst = MagicMock()
        mock_config_inst.hidden_size = 768
        mock_config.from_pretrained.return_value = mock_config_inst

        mock_model_inst = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_inst

        # Mock forward output
        batch_size = 2
        seq_len = 10
        hidden = torch.randn(batch_size, seq_len, 768)

        # Mock the attribute access to retrieve last_hidden_state
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_inst.return_value = mock_output

        prm = ProcessRewardModel(config=self.config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = prm(input_ids)

        self.assertEqual(output["step_rewards"].shape, (batch_size, seq_len))

    @patch("thinkrl.models.prm.AutoModel")
    @patch("thinkrl.models.prm.AutoConfig")
    def test_forward_with_steps(self, mock_config, mock_model):
        """Test forward pass with step positions."""
        # Setup mocks
        mock_config_inst = MagicMock()
        mock_config_inst.hidden_size = 768
        mock_config.from_pretrained.return_value = mock_config_inst

        mock_model_inst = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_inst

        batch_size = 2
        seq_len = 10
        hidden = torch.randn(batch_size, seq_len, 768)

        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model_inst.return_value = mock_output

        prm = ProcessRewardModel(config=self.config)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        step_positions = [[3, 7], [5, 9]]  # 2 steps per sample

        output = prm(input_ids, step_positions=step_positions)

        self.assertEqual(output["step_rewards"].shape, (batch_size, 2))

        # Test padding in step positions
        step_positions_uneven = [[3], [5, 9]]
        output_uneven = prm(input_ids, step_positions=step_positions_uneven)
        self.assertEqual(output_uneven["step_rewards"].shape, (batch_size, 2))
        # Ensure padding value is -inf for first sample's second step
        self.assertEqual(output_uneven["step_rewards"][0, 1], float("-inf"))

    def test_detect_steps(self):
        """Test step detection logic."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [10]  # newline token

        prm = ProcessRewardModel(base_model=MagicMock())  # lightweight mock
        prm.config.step_tag = "\n"

        # [1, 2, 10, 3, 4, 10] -> steps at indices 2 and 5
        input_ids = torch.tensor([[1, 2, 10, 3, 4, 10], [10, 1, 1, 1, 1, 1]])

        positions = prm.detect_steps(input_ids, mock_tokenizer)

        self.assertEqual(positions[0], [2, 5])
        self.assertEqual(positions[1], [0])

    def test_aggregation(self):
        """Test reward aggregation logic."""
        prm = ProcessRewardModel(base_model=MagicMock())

        # [batch, steps]
        step_rewards = torch.tensor(
            [
                [0.5, 0.8, 0.2],
                [0.9, 0.1, float("-inf")],  # 2nd sample has padding
            ]
        )

        # Min
        agg_min = prm.aggregate_rewards(step_rewards, method="min")
        self.assertTrue(torch.allclose(agg_min, torch.tensor([0.2, 0.1])))

        # Mean
        agg_mean = prm.aggregate_rewards(step_rewards, method="mean")
        self.assertTrue(torch.allclose(agg_mean, torch.tensor([0.5, 0.5])))  # (0.9+0.1)/2

        # Last
        agg_last = prm.aggregate_rewards(step_rewards, method="last")
        self.assertTrue(torch.allclose(agg_last, torch.tensor([0.2, 0.1])))

    def test_find_first_error(self):
        """Test first error detection."""
        prm = ProcessRewardModel(base_model=MagicMock())
        prm.config.threshold = 0.5

        # < 0.5 is error
        step_rewards = torch.tensor(
            [
                [0.8, 0.9, 0.4],  # Error at index 2
                [0.2, 0.9, 0.9],  # Error at index 0
                [0.8, 0.9, 0.9],  # No error
                [0.8, 0.2, float("-inf")],  # Error at index 1, ignore padding
            ]
        )

        errors = prm.find_first_error(step_rewards)

        expected = torch.tensor([2, 0, -1, 1])
        self.assertTrue(torch.equal(errors, expected))


if __name__ == "__main__":
    unittest.main()
