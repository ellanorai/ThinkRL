"""`thinkrl dapo` did not exist at all (#124/#193), even though DAPOAlgorithm
(Clip-Higher, token-level loss, dynamic sampling) already existed."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["dapo", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.dapo_trainer.DAPOTrainer") as mock:
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


@pytest.fixture
def mock_tokenizer():
    with patch("transformers.AutoTokenizer") as mock:
        mock.from_pretrained.return_value.pad_token = "<eos>"
        yield mock


def test_dapo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_dapo_loads_only_the_policy_without_a_ref_model(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "actor" in model_types
    assert "ref" not in model_types


def test_dapo_kl_coeff_without_ref_model_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--kl-coeff", "0.04"])

    assert result.exit_code != 0
    assert "requires a reference model" in result.output


def test_dapo_epsilon_range_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--epsilon-low", "0.1", "--epsilon-high", "0.3"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].epsilon_low == 0.1
    assert kwargs["config"].epsilon_high == 0.3


def test_dapo_group_size_reaches_generation_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--group-size", "8"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["generation_config"].num_return_sequences == 8
    assert kwargs["config"].group_size == 8


def test_dapo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
