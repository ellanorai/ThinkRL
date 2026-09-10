"""Every directory shipped under thinkrl/ must be a real, importable package.

setup.py uses find_packages(), which skips any directory without an
__init__.py. thinkrl/generation/ had none, so it was silently left out of the
wheel and unimportable on a clean install.
"""

import importlib
from pathlib import Path

import pytest
from setuptools import find_packages

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "thinkrl"


def _shipped_directories():
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_dir() or path.name == "__pycache__":
            continue
        if any(part == "__pycache__" for part in path.parts):
            continue
        if list(path.glob("*.py")):
            yield path


@pytest.mark.parametrize("directory", list(_shipped_directories()), ids=lambda p: p.name)
def test_directory_with_modules_is_a_package(directory):
    assert (directory / "__init__.py").exists(), (
        f"{directory.relative_to(PACKAGE_ROOT.parent)} contains modules but no __init__.py, "
        "so find_packages() will not ship it"
    )


def test_every_package_imports():
    for name in sorted(find_packages(where=str(PACKAGE_ROOT.parent), include=["thinkrl*"])):
        importlib.import_module(name)
