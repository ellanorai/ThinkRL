import importlib.machinery
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_distributed():
    with patch("thinkrl.utils.distributed_util.init_distributed"), patch(
        "thinkrl.utils.distributed_util.get_local_rank", return_value=0
    ):
        yield


@pytest.fixture
def mock_kto_deps():
    """
    Mock all heavy imports that happen inside the kto CLI command function.
    """
    mock_get_model = MagicMock()
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value.pad_token = None
    mock_tokenizer_cls.from_pretrained.return_value.eos_token = "<eos>"
    mock_tokenizer_cls.from_pretrained.return_value.pad_token_id = 0
    mock_tokenizer_cls.from_pretrained.return_value.eos_token_id = 1
    mock_dataset_cls = MagicMock()
    mock_dataset_cls.return_value.__len__ = MagicMock(return_value=100)
    mock_trainer_cls = MagicMock()

    # Pre-seed sys.modules to prevent __init__.py import cascade into deepspeed
    kto_trainer_mod = types.ModuleType("thinkrl.training.kto_trainer")
    kto_trainer_mod.KTOTrainer = mock_trainer_cls

    # Save originals
    saved = {}
    for key in ["thinkrl.training.kto_trainer"]:
        saved[key] = sys.modules.get(key)

    sys.modules["thinkrl.training.kto_trainer"] = kto_trainer_mod

    with patch("thinkrl.models.loader.get_model", mock_get_model), patch(
        "transformers.AutoTokenizer", mock_tokenizer_cls
    ), patch("thinkrl.data.datasets.RLHFDataset", mock_dataset_cls):
        yield {
            "get_model": mock_get_model,
            "tokenizer": mock_tokenizer_cls,
            "dataset": mock_dataset_cls,
            "trainer": mock_trainer_cls,
        }

    # Restore
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


def test_kto_args_fp16(mock_kto_deps):
    """Test that --fp16 flag is correctly passed."""
    result = runner.invoke(
        app,
        [
            "kto",
            "--model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--fp16",
        ],
    )

    assert result.exit_code == 0, result.output
    # Verify fp16=True passed to get_model
    _, kwargs = mock_kto_deps["get_model"].call_args_list[0] # actor model
    assert kwargs.get("fp16") is True
    # Verify bf16 turned off
    assert kwargs.get("bf16") is False


def test_kto_args_beta(mock_kto_deps):
    """Test that KTO-specific hyperparams are passed to KTOConfig."""
    result = runner.invoke(
        app,
        [
            "kto",
            "--model",
            "gpt2",
            "--dataset",
            "fake_dataset",
            "--beta",
            "0.5",
            "--lambda-d",
            "1.2",
            "--lambda-u",
            "0.8",
            "--entropy-coef",
            "0.01",
            "--reward-threshold",
            "0.7",
        ],
    )

    assert result.exit_code == 0, result.output

    # Check config initialization in trainer
    _, kwargs_trainer = mock_kto_deps["trainer"].call_args
    config = kwargs_trainer.get("config")
    reward_threshold = kwargs_trainer.get("reward_threshold")

    assert config is not None
    assert config.beta == 0.5
    assert config.lambda_d == 1.2
    assert config.lambda_u == 0.8
    assert config.entropy_coef == 0.01
    assert reward_threshold == 0.7
