import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from thinkrl.models.actor import Actor
from thinkrl.models.critic import Critic
from thinkrl.models.loader import _load_from_cloud, _load_from_hf, _load_from_local, get_model
from thinkrl.models.reward_model import RewardModel


class TestModelLoader(unittest.TestCase):
    @patch("thinkrl.models.loader._load_from_hf")
    def test_get_model_hf_actor(self, mock_load_hf):
        """Test loading actor model from HF via get_model."""
        mock_model = MagicMock(spec=Actor)
        mock_load_hf.return_value = mock_model

        # Test basic HF loading
        model = get_model("meta-llama/Llama-2-7b-hf", "actor")

        assert model == mock_model
        mock_load_hf.assert_called_with("meta-llama/Llama-2-7b-hf", "actor")

    @patch("thinkrl.models.loader._load_from_hf")
    def test_get_model_hf_reward(self, mock_load_hf):
        """Test loading reward model from HF via get_model."""
        mock_model = MagicMock(spec=RewardModel)
        mock_load_hf.return_value = mock_model

        model = get_model("my-reward-model", "reward", normalize_reward=True)

        assert model == mock_model
        mock_load_hf.assert_called_with("my-reward-model", "reward", normalize_reward=True)

    def test_get_model_cloud_placeholder(self):
        """Test that cloud URIs raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            get_model("s3://bucket/model", "actor")

        with self.assertRaises(NotImplementedError):
            get_model("gs://bucket/model", "actor")

    @patch("thinkrl.models.loader._load_from_local")
    def test_get_model_local(self, mock_load_local):
        """Test local path detection."""
        mock_model = MagicMock()
        mock_load_local.return_value = mock_model

        # Test file:// prefix
        model = get_model("file:///tmp/model", "actor")
        assert model == mock_model
        mock_load_local.assert_called_with("/tmp/model", "actor")

        # Test directory detection (mock os.path.isdir in loader if needed,
        # but here we rely on the logic that falls through or check starts with / or .\)
        # However, loader.py uses os.path.isdir. To test that branch without actually creating dirs,
        # we can patch os.path.isdir

    @patch("thinkrl.models.loader.os.path.isdir")
    @patch("thinkrl.models.loader._load_from_local")
    def test_get_model_local_dir(self, mock_load_local, mock_isdir):
        """Test directory detection logic."""
        mock_isdir.return_value = True
        mock_model = MagicMock()
        mock_load_local.return_value = mock_model

        model = get_model("./local_checkpoint", "critic")
        assert model == mock_model

        mock_load_local.assert_called_with("./local_checkpoint", "critic")

    @patch("thinkrl.models.actor.Actor")
    def test_load_from_hf_dispatch(self, MockActor):
        """Test internal dispatch of _load_from_hf."""
        mock_instance = MagicMock()
        MockActor.return_value = mock_instance

        model = _load_from_hf("model-id", "actor", lora_rank=16)

        MockActor.assert_called_with(
            pretrained_model="model-id",
            use_flash_attention=True,
            bf16=True,
            load_in_4bit=False,
            lora_rank=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=None,
            device_map=None,
            trust_remote_code=False,
            fp16=False,
            lora_init_type="default",
        )
        assert model == mock_instance

    @patch("thinkrl.models.reward_model.RewardModel")
    def test_load_from_hf_reward(self, MockRewardModel):
        """Test internal dispatch of _load_from_hf for reward."""
        model = _load_from_hf("model-id", "reward", normalize_reward=True)
        assert model == MockRewardModel.return_value
        MockRewardModel.assert_called()
        call_kwargs = MockRewardModel.call_args[1]
        assert call_kwargs["normalize_reward"] is True

    @patch("thinkrl.models.loader._load_from_hf")
    def test_load_from_local_wrapper(self, mock_load_hf):
        """Test _load_from_local wrapping HF loader."""
        with patch("thinkrl.models.loader.os.path.exists") as mock_exists:
            # Simulate config.json existing
            mock_exists.return_value = True

            _load_from_local("/path/to/model", "actor")

            mock_load_hf.assert_called_with("/path/to/model", "actor")

    def test_load_from_local_invalid(self):
        """Test local loader raises error for unknown format."""
        with patch("thinkrl.models.loader.os.path.exists") as mock_exists:
            # Simulate config.json MISSING
            mock_exists.return_value = False

            with self.assertRaises(NotImplementedError):
                _load_from_local("/path/to/model", "actor")


class TestLegacyCompatibility(unittest.TestCase):
    """Test that old functions in model.py correctly call the new loader."""

    @patch("thinkrl.models.loader._load_from_hf")
    def test_get_actor_model_compatibility(self, mock_load_hf):
        from thinkrl.models import get_actor_model

        get_actor_model("llama", lora_rank=8)

        mock_load_hf.assert_called()
        args, kwargs = mock_load_hf.call_args
        assert args[0] == "llama"
        assert args[1] == "actor"
        assert kwargs["lora_rank"] == 8

    @patch("thinkrl.models.loader._load_from_hf")
    def test_get_llm_compatibility(self, mock_load_hf):
        from thinkrl.models import get_llm_for_sequence_regression

        get_llm_for_sequence_regression("meta-llama/Llama-2-7b-hf", model_type="critic")

        mock_load_hf.assert_called()
        args, kwargs = mock_load_hf.call_args
        assert args[1] == "critic"

    @patch("thinkrl.models.loader._load_from_hf")
    def test_create_reference_model_compatibility(self, mock_load_hf):
        from thinkrl.models.model import create_reference_model

        # Should force lora_rank=0 and model_type='ref'
        create_reference_model("meta-llama/Llama-2-7b-hf")

        mock_load_hf.assert_called()
        args, kwargs = mock_load_hf.call_args
        assert args[1] == "ref"
        assert kwargs["lora_rank"] == 0


if __name__ == "__main__":
    unittest.main()
