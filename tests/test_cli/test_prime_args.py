"""`thinkrl prime` did not exist at all (#124/#193), even though PRIMEAlgorithm
(online Implicit PRM + RLOO-baselined PPO) already existed."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["prime", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.prime_trainer.PRIMETrainer") as mock:
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


def test_prime_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_prime_loads_only_the_policy_without_ref_or_prm_models(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert model_types == ["actor"]


def test_prime_ref_and_prm_flags_load_their_own_models(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--ref-model", "some/ref", "--prm-model", "some/prm"])

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert model_types.count("actor") == 1
    assert model_types.count("ref") == 2


def test_prime_beta_and_clip_epsilon_reach_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--beta", "0.1", "--clip-epsilon", "0.3"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].beta == 0.1
    assert kwargs["config"].clip_epsilon == 0.3


def test_prime_num_generations_per_prompt_reaches_generation_config(
    mock_trainer, mock_get_model, mock_dataset, mock_tokenizer
):
    runner.invoke(app, [*BASE, "--num-generations-per-prompt", "4"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["generation_config"].num_return_sequences == 4
    assert kwargs["config"].num_generations_per_prompt == 4


def test_prime_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
