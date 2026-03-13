from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.star_trainer.STaRTrainer") as mock:
        yield mock


@pytest.fixture
def mock_get_model():
    with patch("thinkrl.models.loader.get_model") as mock:
        yield mock


@pytest.fixture
def mock_dataset():
    with patch("thinkrl.data.datasets.RLHFDataset") as mock:
        mock.return_value.__len__.return_value = 100
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


def test_star_args_fp16(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Test that --fp16 flag is correctly passed."""
    result = runner.invoke(
        app,
        [
            "star",
            "--model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--fp16",
        ],
    )

    assert result.exit_code == 0
    # Verify fp16=True passed to get_model
    _, kwargs = mock_get_model.call_args
    assert kwargs.get("fp16") is True
    # Verify bf16 turned off
    assert kwargs.get("bf16") is False


def test_star_args_lora_init(mock_trainer, mock_dataset, mock_tokenizer, mock_get_model):
    """Test that --lora-init flag is passed."""
    result = runner.invoke(
        app,
        [
            "star",
            "--model",
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


def test_star_args_staR_params(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Test that STaR-specific hyperparams are passed to STaRConfig."""
    result = runner.invoke(
        app,
        [
            "star",
            "--model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--max-iterations",
            "25",
            "--scaling-factor",
            "1.5",
            "--warmup-steps",
            "50",
            "--base-steps",
            "100",
        ],
    )

    assert result.exit_code == 0

    # Check config initialization in trainer
    _, kwargs_trainer = mock_trainer.call_args
    config = kwargs_trainer.get("config")
    
    assert config is not None
    assert config.max_iterations == 25
    assert config.step_scaling_factor == 1.5
    assert config.warmup_steps == 50
    assert config.base_training_steps == 100
