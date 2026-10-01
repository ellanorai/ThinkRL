"""`thinkrl reward` used to echo its flags and call _not_implemented (#193), even
though RMTrainer (Bradley-Terry pairwise loss, checkpointing) already existed."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["reward", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.rm_trainer.RMTrainer") as mock:
        mock.return_value.train.return_value = {"loss": 0.1}
        yield mock


@pytest.fixture
def mock_get_model():
    with patch("thinkrl.models.loader.get_model") as mock:
        yield mock


@pytest.fixture
def mock_dataset():
    with patch("thinkrl.data.datasets.PreferenceDataset") as mock:
        mock.return_value.__len__.return_value = 100
        yield mock


@pytest.fixture
def mock_tokenizer():
    with patch("transformers.AutoTokenizer") as mock:
        mock.from_pretrained.return_value.pad_token = "<eos>"
        yield mock


def test_reward_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


def test_reward_loads_the_model_as_a_reward_model(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    _, kwargs = mock_get_model.call_args
    assert kwargs["model_type"] == "reward"


def test_reward_builds_the_dataset_with_the_given_columns(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--chosen-column", "good", "--rejected-column", "bad"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_dataset.call_args
    assert kwargs["chosen_column"] == "good"
    assert kwargs["rejected_column"] == "bad"


def test_reward_margin_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--margin", "0.5"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["args"].margin == 0.5


def test_reward_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
