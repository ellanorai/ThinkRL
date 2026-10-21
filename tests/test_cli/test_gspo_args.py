"""`thinkrl gspo` did not exist at all (#124/#193/#195), even though GSPOAlgorithm
(sequence-level, length-normalized ratio) already existed on its own branch."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["gspo", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.gspo_trainer.GSPOTrainer") as mock:
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


def test_gspo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_gspo_loads_only_the_policy_no_ref_model_flag_exists(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--ref-model", "gpt2"])

    # No --ref-model option: GSPOAlgorithm.compute_loss never applies a KL term,
    # so a flag that silently did nothing would be the same dummy-knob mistake
    # exposing --kl-coeff on this command would be.
    assert result.exit_code != 0


def test_gspo_clip_range_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--epsilon-low", "0.001", "--epsilon-high", "0.002"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].epsilon_low == 0.001
    assert kwargs["config"].epsilon_high == 0.002


def test_gspo_group_size_reaches_generation_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--group-size", "8"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["generation_config"].num_return_sequences == 8
    assert kwargs["config"].group_size == 8


def test_gspo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
