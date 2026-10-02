"""`thinkrl dpo` used to echo its flags and call _not_implemented (#193), even
though DPOAlgorithm (sigmoid/hinge/IPO losses, reference-model KL) already
existed -- only the DPOTrainer training loop (#124) and the dataset's
prompt_length (used to mask the prompt out of DPO's labels) were missing."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

# --ref-model given so the mocked get_model() call covers both models; the
# default clone-the-policy path is exercised separately against a real model.
BASE = ["dpo", "--model", "gpt2", "--dataset", "fake_dataset", "--ref-model", "gpt2"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.dpo_trainer.DPOTrainer") as mock:
        mock.return_value.train.return_value = {"loss": 0.1}
        yield mock


@pytest.fixture
def mock_algorithm():
    with patch("thinkrl.algorithms.dpo.DPOAlgorithm") as mock:
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


def test_dpo_reaches_the_trainer(mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


def test_dpo_loads_the_model_as_an_actor(mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "actor" in model_types
    assert "ref" in model_types


def test_dpo_builds_the_dataset_with_the_given_columns(
    mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer
):
    result = runner.invoke(app, [*BASE, "--chosen-column", "good", "--rejected-column", "bad"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_dataset.call_args
    assert kwargs["chosen_column"] == "good"
    assert kwargs["rejected_column"] == "bad"


def test_dpo_beta_and_loss_type_reach_the_algorithm_config(
    mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer
):
    runner.invoke(app, [*BASE, "--beta", "0.5", "--loss-type", "ipo"])

    _, kwargs = mock_algorithm.call_args
    assert kwargs["config"].beta == 0.5
    assert kwargs["config"].loss_type == "ipo"


def test_dpo_dry_run_does_not_train(mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()


def test_dpo_without_ref_model_clones_the_policy_as_a_real_frozen_reference():
    """The only path real models (not mocks) can exercise: cloning must freeze
    the copy, or the "reference" would drift with every optimizer step."""
    from transformers import AutoModelForCausalLM

    with (
        patch("thinkrl.training.dpo_trainer.DPOTrainer") as mock_trainer,
        patch("thinkrl.algorithms.dpo.DPOAlgorithm") as mock_algorithm,
        patch("thinkrl.data.datasets.PreferenceDataset") as mock_dataset,
        patch(
            "thinkrl.models.loader.get_model",
            return_value=AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2"),
        ),
        patch("transformers.AutoTokenizer") as mock_tokenizer,
    ):
        mock_trainer.return_value.train.return_value = {"loss": 0.1}
        mock_dataset.return_value.__len__.return_value = 100
        mock_tokenizer.from_pretrained.return_value.pad_token = "<eos>"

        result = runner.invoke(app, ["dpo", "--model", "sshleifer/tiny-gpt2", "--dataset", "fake_dataset"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_algorithm.call_args
        ref_model = kwargs["ref_model"]
        assert not ref_model.training
        assert all(not p.requires_grad for p in ref_model.parameters())
