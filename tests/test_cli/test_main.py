"""
Tests for CLI Main Module
=========================

Tests for command-line interface.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestCLIAvailability:
    """Tests for CLI module availability."""

    def test_import_cli_module(self):
        """Test that CLI module can be imported."""
        from thinkrl.cli import main
        assert main is not None

    def test_app_exists_or_none(self):
        """Test that app exists (if typer installed) or is None."""
        from thinkrl.cli.main import app
        # App is either a Typer app or None
        assert app is None or hasattr(app, "command")


class TestCLIWithTyper:
    """Tests for CLI when typer is available."""

    @pytest.fixture
    def typer_available(self):
        """Check if typer is available."""
        try:
            import typer
            return True
        except ImportError:
            return False

    def test_info_command(self, typer_available):
        """Test info command execution."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        runner = CliRunner()
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "ThinkRL" in result.stdout

    def test_info_shows_algorithms(self, typer_available):
        """Test info command shows available algorithms."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        runner = CliRunner()
        result = runner.invoke(app, ["info"])

        assert "ppo" in result.stdout.lower()

    def test_train_requires_config(self, typer_available):
        """Test train command requires config file."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        runner = CliRunner()
        result = runner.invoke(app, ["train"])

        # Should fail without --config
        assert result.exit_code != 0

    def test_train_dry_run(self, typer_available):
        """Test train command with dry run."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        # Create temp config file
        config_data = {"max_steps": 100}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(app, ["train", "--config", config_path, "--dry-run"])

            assert result.exit_code == 0
            assert "valid" in result.stdout.lower() or "max_steps" in result.stdout
        finally:
            Path(config_path).unlink()

    def test_train_with_override(self, typer_available):
        """Test train command with config override."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        config_data = {"max_steps": 100}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(app, [
                "train",
                "--config", config_path,
                "--override", "max_steps=500",
                "--dry-run",
            ])

            assert result.exit_code == 0
            # Should show overridden value
            assert "500" in result.stdout
        finally:
            Path(config_path).unlink()

    def test_info_with_config(self, typer_available):
        """Test info command with config file."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        config_data = {"seed": 12345}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(app, ["info", "--config", config_path])

            assert result.exit_code == 0
            assert "12345" in result.stdout
        finally:
            Path(config_path).unlink()


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_without_typer(self):
        """Test main function behavior without typer."""
        # This is hard to test without actually removing typer
        # Just ensure main function exists and is callable
        from thinkrl.cli.main import main
        assert callable(main)


class TestCLIConfigValidation:
    """Tests for CLI config validation."""

    @pytest.fixture
    def typer_available(self):
        """Check if typer is available."""
        try:
            import typer
            return True
        except ImportError:
            return False

    def test_invalid_config_format_error(self, typer_available):
        """Test error on invalid config format."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        # Create file with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"invalid")
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(app, ["train", "--config", config_path])

            assert result.exit_code != 0
        finally:
            Path(config_path).unlink()

    def test_invalid_override_format(self, typer_available):
        """Test error on invalid override format."""
        if not typer_available:
            pytest.skip("Typer not installed")

        from typer.testing import CliRunner
        from thinkrl.cli.main import app

        if app is None:
            pytest.skip("CLI app not available")

        config_data = {"max_steps": 100}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(app, [
                "train",
                "--config", config_path,
                "--override", "invalid_format_no_equals",
                "--dry-run",
            ])

            assert result.exit_code != 0
            assert "expected key=value" in result.stdout.lower() or "invalid" in result.stdout.lower()
        finally:
            Path(config_path).unlink()
