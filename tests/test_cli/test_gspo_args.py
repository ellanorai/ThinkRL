"""`thinkrl gspo` (#195): GSPO had no CLI command at all -- this is the new one,
built the same way as the rollout family in #193 (GRPOTrainer(algorithm=...)).
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.gspo import app
from thinkrl.cli.main import app as main_app


runner = CliRunner()

BASE = ["--model", "gpt2", "--ref-model", "gpt2", "--dataset", "d.jsonl", "--source", "json"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.grpo_trainer.GRPOTrainer") as mock:
        mock.return_value.train.return_value = None
        yield mock


@pytest.fixture
def mock_algorithm():
    with patch("thinkrl.algorithms.gspo.GSPOAlgorithm") as mock:
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


def test_ref_model_is_required():
    result = runner.invoke(app, ["--model", "gpt2", "--dataset", "d.jsonl"])

    assert result.exit_code == 1
    assert "--ref-model" in result.output


def test_gspo_reaches_the_trainer(mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_gspo_passes_an_algorithm_not_model_to_the_trainer(
    mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer
):
    """GRPOTrainer raises if both `algorithm` and `model` are given; this command
    must build the algorithm itself, not hand the trainer a bare model."""
    runner.invoke(app, BASE)

    _, kwargs = mock_trainer.call_args
    assert "algorithm" in kwargs
    assert "model" not in kwargs


def test_epsilon_and_beta_reach_the_algorithm_config(
    mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer
):
    runner.invoke(app, [*BASE, "--epsilon-low", "0.001", "--epsilon-high", "0.002", "--beta", "0.05"])

    _, kwargs = mock_algorithm.call_args
    config = kwargs["config"]
    assert config.epsilon_low == 0.001
    assert config.epsilon_high == 0.002
    assert config.beta == 0.05


def test_dry_run_does_not_train(mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()


def test_an_unknown_logging_backend_is_rejected():
    result = runner.invoke(app, [*BASE, "--logging-backend", "mlflow"])

    assert result.exit_code == 1
    assert "mlflow" in result.output


def test_gspo_is_mounted_on_the_main_app():
    result = CliRunner().invoke(main_app, ["gspo", "--help"])

    assert result.exit_code == 0
    assert "Group Sequence Policy Optimization" in result.output


def test_gspo_is_exposed_as_a_console_script():
    import pathlib

    setup_py = pathlib.Path(__file__).resolve().parents[2] / "setup.py"

    assert '"gspo": "thinkrl.cli.gspo:main"' in setup_py.read_text()


def test_malformed_reward_fn_spec_errors_cleanly_instead_of_crashing_on_none(
    mock_trainer, mock_algorithm, mock_get_model, mock_dataset, mock_tokenizer
):
    """spec_from_file_location can return None for a path with no matching loader;
    module_from_spec(None) would otherwise crash before this reaches typer's error
    handling."""
    with patch("importlib.util.spec_from_file_location", return_value=None):
        result = runner.invoke(app, [*BASE, "--reward-fn", "/nonexistent/path.py:reward_fn"])

    assert result.exit_code == 1
    assert "Error loading reward function" in result.output
