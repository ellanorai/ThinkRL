"""CLI basics: version, completion, honest algorithm listing, non-zero exits on stubs."""

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

import thinkrl  # noqa: E402
from thinkrl.algorithms import ALGORITHMS  # noqa: E402
from thinkrl.cli.main import _is_stub, app  # noqa: E402


runner = CliRunner()

STUBS = {"kto", "orpo", "rloo"}
STUB_COMMANDS = ["sft", "dpo", "ppo", "reward", "orpo", "kto"]


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_the_version_and_exits_zero(flag):
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert thinkrl.__version__ in result.stdout


def test_completion_is_available():
    """add_completion=False previously removed functionality Typer supplies for free."""
    result = runner.invoke(app, ["--help"])

    assert "--install-completion" in result.stdout


def test_stub_detection_matches_the_registry():
    detected = {name for name, cls in ALGORITHMS.items() if _is_stub(cls)}

    assert detected == STUBS


def test_info_separates_implemented_from_stub_algorithms():
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    available, _, unimplemented = result.stdout.partition("Registered but not implemented")

    assert unimplemented, "info did not distinguish stub algorithms"
    for name in STUBS:
        assert f"- {name}" in unimplemented
        assert f"- {name}" not in available
    assert "- grpo" in available


def test_info_reports_the_version():
    result = runner.invoke(app, ["info"])

    assert thinkrl.__version__ in result.stdout


@pytest.mark.parametrize("command", STUB_COMMANDS)
def test_unimplemented_commands_exit_non_zero(command):
    """Exiting 0 makes a command that did nothing indistinguishable from one that worked."""
    result = runner.invoke(app, [command, "--model", "gpt2", "--dataset", "foo"])

    assert result.exit_code != 0, f"`thinkrl {command}` reported success without doing anything"


def test_the_failure_message_points_somewhere_useful():
    result = runner.invoke(app, ["sft", "--model", "gpt2", "--dataset", "foo"])

    combined = result.stdout + (result.stderr or "")
    assert "not implemented" in combined
    assert "issues/" in combined


def test_implemented_commands_are_still_listed():
    result = runner.invoke(app, ["--help"])

    for command in ("grpo", "star", "info"):
        assert command in result.stdout
