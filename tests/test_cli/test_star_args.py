import importlib.machinery
import sys
import types
from unittest.mock import MagicMock, patch


# Mock deepspeed *before* importing CLI to prevent DeprecationWarning from distutils
if "deepspeed" not in sys.modules:
    mock_deepspeed = MagicMock()
    mock_deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    sys.modules["deepspeed"] = mock_deepspeed

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
def mock_star_deps():
    """
    Mock all heavy imports that happen inside the star CLI command function.

    The CLI star command does lazy imports (inside the function body):
        from thinkrl.algorithms.star import STaRConfig
        from thinkrl.data.datasets import RLHFDataset
        from thinkrl.models.loader import get_model
        from thinkrl.training.star_trainer import STaRTrainer

    We need to pre-seed sys.modules for thinkrl.training.star_trainer to avoid
    the thinkrl.training.__init__ → sft_trainer → deepspeed cascade.
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
    star_trainer_mod = types.ModuleType("thinkrl.training.star_trainer")
    star_trainer_mod.STaRTrainer = mock_trainer_cls

    # Save originals
    saved = {}
    for key in ["thinkrl.training.star_trainer"]:
        saved[key] = sys.modules.get(key)

    sys.modules["thinkrl.training.star_trainer"] = star_trainer_mod

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


def test_star_args_fp16(mock_star_deps):
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

    assert result.exit_code == 0, result.output
    # Verify fp16=True passed to get_model
    _, kwargs = mock_star_deps["get_model"].call_args
    assert kwargs.get("fp16") is True
    # Verify bf16 turned off
    assert kwargs.get("bf16") is False


def test_star_args_lora_init(mock_star_deps):
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

    assert result.exit_code == 0, result.output
    _, kwargs = mock_star_deps["get_model"].call_args
    assert kwargs.get("lora_init_type") == "pissa"


def test_star_args_staR_params(mock_star_deps):
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

    assert result.exit_code == 0, result.output

    # Check config initialization in trainer
    _, kwargs_trainer = mock_star_deps["trainer"].call_args
    config = kwargs_trainer.get("config")

    assert config is not None
    assert config.max_iterations == 25
    assert config.step_scaling_factor == 1.5
    assert config.warmup_steps == 50
    assert config.base_training_steps == 100
