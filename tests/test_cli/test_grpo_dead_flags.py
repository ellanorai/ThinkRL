"""Flags that are parsed and then ignored (#79), and the vLLM plumbing gap (#85).

A silently dropped flag is worse than one that does not exist, because the run looks
configured. --grad-accum was the sharpest case: it reached the W&B run config and nothing
else, so a run was *recorded* as though gradient accumulation had been applied.
"""

import inspect

import pytest


typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from thinkrl.cli.grpo import app  # noqa: E402
from thinkrl.training.grpo_trainer import GRPOTrainer  # noqa: E402


runner = CliRunner()

BASE = ["--model", "gpt2", "--dataset", "d.jsonl", "--source", "json"]


def test_grad_accum_is_rejected_rather_than_dropped():
    result = runner.invoke(app, [*BASE, "--grad-accum", "4"])

    assert result.exit_code == 1
    assert "--grad-accum" in result.output


def test_grad_accum_of_one_is_the_no_op_and_still_allowed():
    """Rejecting the default would break every existing invocation."""
    result = runner.invoke(app, [*BASE, "--grad-accum", "1", "--dry-run"])

    assert "not supported" not in result.output


def test_deepspeed_is_rejected_and_names_the_reason():
    result = runner.invoke(app, [*BASE, "--deepspeed", "configs/ds_zero2.json"])

    assert result.exit_code == 1
    assert "#128" in result.output


def test_an_unknown_logging_backend_is_rejected():
    """It used to accept anything and log nothing."""
    result = runner.invoke(app, [*BASE, "--logging-backend", "mlflow"])

    assert result.exit_code == 1
    assert "mlflow" in result.output


def test_grad_accum_no_longer_reaches_the_wandb_config():
    """The actively harmful half: the value was recorded as applied."""
    from thinkrl.cli import grpo

    source = inspect.getsource(grpo)
    wandb_config = source.split("wandb.init(")[1].split(")")[0]

    assert "grad_accum" not in wandb_config


def test_tensorboard_is_actually_constructed():
    """It is the default value, and until now it logged nothing at all while a working
    TensorBoardLogger sat in the package with its own tests."""
    from thinkrl.cli import grpo

    source = inspect.getsource(grpo)

    assert "TensorBoardLogger" in source


def test_trainer_forwards_the_vllm_url_and_world_size():
    """VLLMClient accepted both and the trainer passed neither, so a remote worker or a
    different topology was unreachable (#85)."""
    params = inspect.signature(GRPOTrainer.__init__).parameters

    assert "vllm_url" in params
    assert "vllm_sync_world_size" in params

    source = inspect.getsource(GRPOTrainer.__init__)
    assert "url=vllm_url" in source
    assert "sync_world_size=vllm_sync_world_size" in source


def test_the_cli_exposes_the_vllm_options():
    params = {p.name for p in typer.main.get_command(app).params}

    assert "vllm_url" in params
    assert "vllm_sync_world_size" in params


def test_the_worker_is_exposed_as_a_console_script():
    """It was a complete FastAPI server with its own main() that nothing ran."""
    import pathlib

    setup_py = pathlib.Path(__file__).resolve().parents[2] / "setup.py"

    assert "thinkrl-vllm-worker" in setup_py.read_text()
