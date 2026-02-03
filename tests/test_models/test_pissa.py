from unittest.mock import MagicMock, patch

import pytest

from thinkrl.models.actor import Actor
from thinkrl.models.loader import get_model
from thinkrl.models.reward_model import RewardModel


@pytest.fixture
def mock_peft():
    with patch("thinkrl.models.actor.get_peft_model", create=True) as mock:
        yield mock


@pytest.fixture
def mock_transformers():
    with patch("thinkrl.models.actor.AutoModelForCausalLM") as mock:
        yield mock


def test_actor_pissa_init(mock_transformers, mock_peft):
    """Test Actor initializes with PiSSA config."""
    with patch("thinkrl.models.actor._PEFT_AVAILABLE", True):
        Actor(pretrained_model="gpt2", lora_rank=8, lora_init_type="pissa")

        # Check that LoraConfig was created with init_lora_weights="pissa"
        # We need to mock LoraConfig to check its args
        with patch("thinkrl.models.actor.LoraConfig") as mock_config:
            # Re-init to capture mock
            Actor(pretrained_model="gpt2", lora_rank=8, lora_init_type="pissa")
            _, kwargs = mock_config.call_args
            assert kwargs.get("init_lora_weights") == "pissa"


def test_actor_default_lora_init(mock_transformers, mock_peft):
    """Test Actor maps 'default' to True."""
    with patch("thinkrl.models.actor._PEFT_AVAILABLE", True):
        with patch("thinkrl.models.actor.LoraConfig") as mock_config:
            Actor(pretrained_model="gpt2", lora_rank=8, lora_init_type="default")
            _, kwargs = mock_config.call_args
            assert kwargs.get("init_lora_weights") is True
