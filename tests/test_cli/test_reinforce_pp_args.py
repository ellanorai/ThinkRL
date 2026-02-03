from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.reinforce_pp_trainer.ReinforcePPTrainer") as mock:
        yield mock


@pytest.fixture
def mock_get_model():
    with patch("thinkrl.models.loader.get_model") as mock:
        yield mock


@pytest.fixture
def mock_dataset():
    with patch("thinkrl.data.datasets.RLHFDataset") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_distributed():
    with patch("thinkrl.utils.distributed_util.init_distributed"), patch(
        "thinkrl.utils.distributed_util.get_local_rank", return_value=0
    ):
        yield


@pytest.fixture
def mock_tokenizer():
    with patch("transformers.AutoTokenizer") as mock:
        mock.from_pretrained.return_value.pad_token = None
        yield mock


def test_reinforce_pp_args_fp16(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Test that --fp16 flag is correctly passed."""
    result = runner.invoke(
        app,
        [
            "reinforce-pp",
            "--model",
            "gpt2",
            "--ref-model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--fp16",
            "--batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    # Verify fp16=True passed to get_model
    _, kwargs = mock_get_model.call_args
    assert kwargs.get("fp16") is True
    # Verify bf16 turned off
    assert kwargs.get("bf16") is False


def test_reinforce_pp_args_lora_init(mock_trainer, mock_dataset, mock_tokenizer, mock_get_model):
    """Test that --lora-init flag is passed."""
    result = runner.invoke(
        app,
        [
            "reinforce-pp",
            "--model",
            "gpt2",
            "--ref-model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--lora-init",
            "pissa",
        ],
    )

    assert result.exit_code == 0
    _, kwargs = mock_get_model.call_args
    assert kwargs.get("lora_init_type") == "pissa"


def test_reinforce_pp_args_max_samples(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Test that --max-samples and --max-length are passed."""
    result = runner.invoke(
        app,
        [
            "reinforce-pp",
            "--model",
            "gpt2",
            "--ref-model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--max-samples",
            "100",
            "--max-length",
            "256",
        ],
    )

    assert result.exit_code == 0

    # Check dataset initialization
    _, kwargs = mock_dataset.call_args
    assert kwargs.get("max_samples") == 100
    assert kwargs.get("max_length") == 256
