"""`thinkrl copo` did not exist before #124's COPO sub-task: COPOAlgorithm was
registered in ALGORITHMS, had a `create_copo` factory and 655 lines of
implementation, and nothing in the package could run it.

Mirrors test_ipo_args.py, plus the validations specific to COPO's online loop:
it needs a reward model, at least two generations to form a pair, and a
non-zero temperature for the exploration bonus to mean anything."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = [
    "copo",
    "--model",
    "gpt2",
    "--dataset",
    "fake_dataset",
    "--reward-model",
    "gpt2-rm",
    "--ref-model",
    "gpt2",
]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.copo_trainer.COPOTrainer") as mock:
        mock.return_value.config.hidden_size = 768
        mock.return_value.train.return_value = {"loss": 0.1}
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


def test_copo_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()


def test_copo_loads_a_policy_a_reward_model_and_a_reference(
    mock_trainer, mock_get_model, mock_dataset, mock_tokenizer
):
    runner.invoke(app, BASE)

    model_types = [kwargs["model_type"] for _, kwargs in mock_get_model.call_args_list]
    assert "actor" in model_types
    assert "reward" in model_types
    assert "ref" in model_types


def test_reward_model_is_mandatory(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """COPO builds its own preference pairs by scoring its own generations, so
    there is no chosen/rejected column to fall back on."""
    result = runner.invoke(app, ["copo", "--model", "gpt2", "--dataset", "fake_dataset"])

    assert result.exit_code != 0
    assert "--reward-model" in result.output
    mock_trainer.assert_not_called()


def test_alpha_beta_and_generations_reach_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--alpha", "0.3", "--beta", "0.2", "-g", "8"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].alpha == 0.3
    assert kwargs["config"].beta == 0.2
    assert kwargs["config"].num_generations_per_prompt == 8


def test_cfn_flags_reach_the_config(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--cfn-output-dim", "40", "--cfn-buffer-size", "500", "--cfn-lr", "0.01"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["config"].cfn_output_dim == 40
    assert kwargs["config"].cfn_buffer_size == 500
    assert kwargs["config"].cfn_learning_rate == 0.01


def test_loop_flags_reach_the_trainer_args(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    runner.invoke(app, [*BASE, "--iterations", "7", "--prompts-per-iteration", "3"])

    _, kwargs = mock_trainer.call_args
    assert kwargs["args"].num_iterations == 7
    assert kwargs["args"].prompts_per_iteration == 3


def test_no_algorithm_surface_at_the_cli(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """Per #124: the trainer owns which algorithm it drives."""
    result = runner.invoke(app, [*BASE, "--help"])

    assert "--algorithm" not in result.output
    runner.invoke(app, BASE)
    _, kwargs = mock_trainer.call_args
    assert "algorithm" not in kwargs


def test_a_single_generation_cannot_form_a_pair(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """COPO pairs the best-scoring completion against the worst. With one
    sample per prompt create_preference_pairs yields nothing and every
    iteration logs "No preference pairs created" while training nothing."""
    result = runner.invoke(app, [*BASE, "-g", "1"])

    assert result.exit_code == 1
    assert "cannot produce a preference pair" in result.output
    mock_trainer.assert_not_called()


def test_greedy_decoding_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """The exploration bonus rewards visiting rare states, which needs sampling.
    Temperature 0 makes every generation per prompt identical."""
    result = runner.invoke(app, [*BASE, "--temperature", "0"])

    assert result.exit_code == 1
    assert "nothing to explore" in result.output
    mock_trainer.assert_not_called()


def test_an_unknown_loss_type_is_refused(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--loss-type", "kto"])

    assert result.exit_code == 1
    assert "not a COPO preference loss" in result.output
    mock_trainer.assert_not_called()


def test_copo_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()


def test_copo_without_ref_model_clones_the_policy_as_a_real_frozen_reference():
    """The only path real models (not mocks) can exercise: cloning must freeze
    the copy, or the "reference" would drift with every optimizer step."""
    from transformers import AutoModelForCausalLM

    with (
        patch("thinkrl.training.copo_trainer.COPOTrainer") as mock_trainer,
        patch("thinkrl.data.datasets.RLHFDataset") as mock_dataset,
        patch(
            "thinkrl.models.loader.get_model",
            return_value=AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2"),
        ),
        patch("transformers.AutoTokenizer") as mock_tokenizer,
    ):
        mock_trainer.return_value.config.hidden_size = 2
        mock_trainer.return_value.train.return_value = {"loss": 0.1}
        mock_dataset.return_value.__len__.return_value = 100
        mock_tokenizer.from_pretrained.return_value.pad_token = "<eos>"

        result = runner.invoke(
            app,
            [
                "copo",
                "--model",
                "sshleifer/tiny-gpt2",
                "--dataset",
                "fake_dataset",
                "--reward-model",
                "sshleifer/tiny-gpt2",
            ],
        )

        assert result.exit_code == 0, result.output
        _, kwargs = mock_trainer.call_args
        ref_model = kwargs["reference_model"]
        assert not ref_model.training
        assert all(not p.requires_grad for p in ref_model.parameters())
