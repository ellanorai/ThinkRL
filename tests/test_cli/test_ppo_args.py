"""`thinkrl ppo` used to echo its flags and call _not_implemented (#124/#193), even
though PPOAlgorithm (actor-critic, GAE, clipped surrogate) already existed."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["ppo", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.ppo_trainer.PPOTrainer") as mock:
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


def test_ppo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_ppo_loads_policy_and_critic(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "actor" in model_types
    assert "critic" in model_types


def test_ppo_without_reward_model_warns_and_does_not_load_one(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert "dummy len-based reward" in result.output
    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "reward" not in model_types


def test_ppo_reward_model_flag_loads_a_reward_model(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--reward-model", "some/reward-model"])

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "reward" in model_types


def test_ppo_clip_range_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--clip-range", "0.3"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].policy_clip == 0.3


def test_ppo_rollouts_per_prompt_reaches_generation_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--rollouts-per-prompt", "4"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["generation_config"].num_return_sequences == 4


def test_ppo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()
