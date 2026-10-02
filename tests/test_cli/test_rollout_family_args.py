"""ppo/dapo/vapo/prime/dr-grpo had no code path at all (#124, #193): dapo, vapo, prime
and dr-grpo had no CLI command whatsoever, and ppo echoed its flags into
_not_implemented. All five now drive GRPOTrainer's generic algorithm= path.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["--model", "gpt2", "--dataset", "fake_dataset"]

ALGO_CLASS_PATH = {
    "ppo": "thinkrl.algorithms.ppo.PPOAlgorithm",
    "dapo": "thinkrl.algorithms.dapo.DAPOAlgorithm",
    "vapo": "thinkrl.algorithms.vapo.VAPOAlgorithm",
    "prime": "thinkrl.algorithms.prime.PRIMEAlgorithm",
    "dr-grpo": "thinkrl.algorithms.dr_grpo.DrGRPOAlgorithm",
}


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.grpo_trainer.GRPOTrainer") as mock:
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


@pytest.mark.parametrize("command", ["ppo", "dapo", "vapo", "prime", "dr-grpo"])
def test_the_command_reaches_the_trainer(command, mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH[command]) as mock_algo:
        result = runner.invoke(app, [command, *BASE])

        assert result.exit_code == 0, result.output
        mock_algo.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer.return_value.train.assert_called_once()


@pytest.mark.parametrize("command", ["ppo", "dapo", "vapo", "prime", "dr-grpo"])
def test_dry_run_does_not_train(command, mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH[command]):
        result = runner.invoke(app, [command, *BASE, "--dry-run"])

        assert result.exit_code == 0, result.output
        mock_trainer.return_value.train.assert_not_called()


def test_ppo_builds_a_separate_critic(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH["ppo"]):
        runner.invoke(app, ["ppo", *BASE])

        model_types = [kwargs.get("model_type") for _, kwargs in mock_get_model.call_args_list]
        assert "critic" in model_types


def test_dapo_passes_clip_bounds_into_its_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH["dapo"]) as mock_algo:
        runner.invoke(app, ["dapo", *BASE, "--epsilon-low", "0.1", "--epsilon-high", "0.4"])

        _, kwargs = mock_algo.call_args
        assert kwargs["config"].epsilon_low == 0.1
        assert kwargs["config"].epsilon_high == 0.4


def test_prime_beta_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH["prime"]) as mock_algo:
        runner.invoke(app, ["prime", *BASE, "--beta", "0.25"])

        _, kwargs = mock_algo.call_args
        assert kwargs["config"].beta == 0.25


def test_dr_grpo_group_size_reaches_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH["dr-grpo"]) as mock_algo:
        runner.invoke(app, ["dr-grpo", *BASE, "--group-size", "8"])

        _, kwargs = mock_algo.call_args
        assert kwargs["config"].group_size == 8


def test_ref_model_is_loaded_and_forwarded_when_given(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    with patch(ALGO_CLASS_PATH["dapo"]) as mock_algo:
        runner.invoke(app, ["dapo", *BASE, "--ref-model", "gpt2-ref"])

        model_calls = {c.args[0]: c.kwargs for c in mock_get_model.call_args_list}
        assert "gpt2-ref" in model_calls
        assert model_calls["gpt2-ref"]["model_type"] == "ref"
        _, algo_kwargs = mock_algo.call_args
        assert algo_kwargs["ref_model"] is mock_get_model.return_value


def test_remote_rm_url_takes_precedence_over_reward_model(
    mock_trainer, mock_get_model, mock_dataset, mock_tokenizer
):
    with patch(ALGO_CLASS_PATH["ppo"]), patch("thinkrl.rewards.RemoteRewardScorer") as mock_scorer:
        result = runner.invoke(
            app, ["ppo", *BASE, "--remote-rm-url", "http://rm:8000", "--reward-model", "some/reward-model"]
        )

        assert result.exit_code == 0, result.output
        mock_scorer.assert_called_once_with(remote_urls="http://rm:8000")
