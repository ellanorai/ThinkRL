"""`thinkrl ipo` did not exist before #124's IPO sub-task: IPOAlgorithm was
registered in ALGORITHMS and had a `create_ipo` factory, but no command and no
trainer, so running IPO meant assembling DPOTrainer by hand.

Mirrors test_dpo_args.py, plus the two validations the DPO command has no
equivalent of: IPO's loss names and its positive-tau requirement."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

# --ref-model given so one mocked get_model() covers both models; the
# clone-the-policy path is exercised separately against a real model.
BASE = ["ipo", "--model", "gpt2", "--dataset", "fake_dataset", "--ref-model", "gpt2"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.ipo_trainer.IPOTrainer") as mock:
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


def test_ipo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


def test_ipo_loads_a_policy_and_a_reference(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "actor" in model_types
    assert "ref" in model_types


def test_ipo_builds_the_dataset_with_the_given_columns(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--chosen-column", "good", "--rejected-column", "bad"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_dataset.call_args
    assert kwargs["chosen_column"] == "good"
    assert kwargs["rejected_column"] == "bad"


def test_tau_and_loss_type_reach_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--tau", "0.25", "--loss-type", "ipo_hinge"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].tau == 0.25
    assert kwargs["config"].loss_type == "ipo_hinge"


def test_length_normalization_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--length-normalization"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].length_normalization is True


def test_no_algorithm_surface_at_the_cli(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Per #124: the trainer owns which algorithm it drives, the CLI does not
    pick one."""
    result = runner.invoke(app, [*BASE, "--help"])

    assert "--algorithm" not in result.output
    runner.invoke(app, BASE)
    _, kwargs = mock_trainer.call_args
    assert "algorithm" not in kwargs


def test_a_dpo_loss_name_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """`--loss-type sigmoid` is valid for dpo and meaningless for ipo. Taking it
    silently would have IPOLoss raise deep in the first backward pass."""
    result = runner.invoke(app, [*BASE, "--loss-type", "sigmoid"])

    assert result.exit_code == 1
    assert "is not an IPO loss" in result.output
    mock_trainer.assert_not_called()


def test_non_positive_tau_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """The loss target is 1/(2*tau); tau=0 is a ZeroDivisionError and tau<0
    inverts the margin the policy is pushed towards."""
    result = runner.invoke(app, [*BASE, "--tau", "0"])

    assert result.exit_code == 1
    assert "must be positive" in result.output
    mock_trainer.assert_not_called()


def test_ipo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()


def test_ipo_without_ref_model_clones_the_policy_as_a_real_frozen_reference():
    """The only path real models (not mocks) can exercise: cloning must freeze
    the copy, or the "reference" would drift with every optimizer step."""
    from transformers import AutoModelForCausalLM

    with (
        patch("thinkrl.training.ipo_trainer.IPOTrainer") as mock_trainer,
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

        result = runner.invoke(app, ["ipo", "--model", "sshleifer/tiny-gpt2", "--dataset", "fake_dataset"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_trainer.call_args
        ref_model = kwargs["ref_model"]
        assert not ref_model.training
        assert all(not p.requires_grad for p in ref_model.parameters())
