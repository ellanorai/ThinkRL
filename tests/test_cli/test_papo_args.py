"""`thinkrl papo` did not exist at all (#194), even though PAPOAlgorithm already
existed -- it was blocked on `masked_input_ids` having no producer anywhere in the
rollout pipeline. cli/papo.py wires it through the new PAPOTrainer, which builds
that field itself (see test_papo_trainer.py)."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["papo", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.papo_trainer.PAPOTrainer") as mock:
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


def test_papo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--kl-coeff", "0"])

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_papo_kl_coeff_without_ref_model_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--kl-coeff", "0.1"])

    assert result.exit_code != 0
    assert "requires a reference model" in result.output


def test_papo_perception_knobs_reach_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--kl-coeff", "0", "--gamma", "0.5", "--eta", "0.25", "--kl-prcp-cap", "2.0"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].gamma == 0.5
    assert kwargs["config"].eta == 0.25
    assert kwargs["config"].kl_prcp_cap == 2.0


def test_papo_mask_prob_reaches_the_trainer_not_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--kl-coeff", "0", "--mask-prob", "0.3"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["mask_prob"] == 0.3


def test_papo_group_size_reaches_generation_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--kl-coeff", "0", "--group-size", "8"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["generation_config"].num_return_sequences == 8
    assert kwargs["config"].group_size == 8


def test_papo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--kl-coeff", "0", "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
