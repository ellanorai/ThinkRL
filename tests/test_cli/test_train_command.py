"""`thinkrl train` builds a trainer from a config instead of printing and exiting 0."""

import inspect

import pytest
import yaml

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from thinkrl.cli.main import TRAINABLE_FROM_CONFIG, _resolve_algorithm_name, app  # noqa: E402
from thinkrl.config.base import ThinkRLConfig  # noqa: E402


runner = CliRunner()


def _config_dict(algorithm="grpo", dataset="/tmp/does-not-exist.jsonl"):
    return {
        "model": {"name_or_path": "facebook/opt-125m", "use_flash_attention": False},
        "algorithm": {"name": algorithm, "learning_rate": 1e-6, "group_size": 2},
        "data": {"dataset": dataset, "max_prompt_length": 32, "max_response_length": 32},
        "distributed": {"micro_batch_size": 2, "bf16": False},
        "logging": {"output_dir": "/tmp/thinkrl-test-out"},
    }


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_train_is_no_longer_a_stub():
    """The regression: the body ended at `# TODO: Implement actual training loop`."""
    source = inspect.getsource(app.registered_commands[0].callback)

    assert "TODO: Implement actual training loop" not in source


def test_algorithm_name_survives_parsing():
    """No algorithm config declares a `name` field, so it used to be dropped on parse."""
    config = ThinkRLConfig.from_dict(_config_dict())

    assert type(config.algorithm).__name__ == "GRPOConfig"
    assert config.algorithm.name == "grpo"


def test_algorithm_name_survives_a_round_trip():
    """to_dict/from_dict silently turned every algorithm back into the ppo default."""
    original = ThinkRLConfig.from_dict(_config_dict(algorithm="dapo"))
    restored = ThinkRLConfig.from_dict(original.to_dict())

    assert restored.algorithm.name == "dapo"
    assert type(restored.algorithm).__name__ == "DAPOConfig"


def test_resolve_algorithm_name_reads_both_shapes():
    parsed = ThinkRLConfig.from_dict(_config_dict())
    assert _resolve_algorithm_name(parsed) == "grpo"

    fallback = ThinkRLConfig.from_dict({"algorithm": {"name": "not-an-algorithm"}})
    assert _resolve_algorithm_name(fallback) == "not-an-algorithm"


def test_unsupported_algorithm_fails_by_name(tmp_path):
    config = _write(tmp_path, _config_dict(algorithm="dpo"))

    result = runner.invoke(app, ["train", "--config", config])

    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "dpo" in combined
    assert "grpo" in combined, "the error should name what is supported"


def test_missing_dataset_is_rejected_before_loading_a_model(tmp_path):
    data = _config_dict()
    data["data"]["dataset"] = None
    config = _write(tmp_path, data)

    result = runner.invoke(app, ["train", "--config", config])

    assert result.exit_code != 0
    assert "dataset" in result.stdout + (result.stderr or "")


def test_dry_run_still_validates_without_training(tmp_path):
    result = runner.invoke(app, ["train", "--config", _write(tmp_path, _config_dict()), "--dry-run"])

    assert result.exit_code == 0
    assert "Configuration is valid" in result.stdout


def test_resume_is_rejected_rather_than_silently_ignored(tmp_path):
    result = runner.invoke(
        app, ["train", "--config", _write(tmp_path, _config_dict()), "--resume", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "resume" in (result.stdout + (result.stderr or "")).lower()


def test_supported_set_is_not_empty():
    assert TRAINABLE_FROM_CONFIG
