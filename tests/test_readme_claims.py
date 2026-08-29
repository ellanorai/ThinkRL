"""The README must not present unimplemented APIs as runnable.

It documented `from thinkrl.training import CoTTrainer, CoTConfig`, which raises
ImportError, and a `thinkrl.cli.train_rl` entry point that is not a module.
"""

import importlib
import re
from pathlib import Path

import pytest


README = Path(__file__).resolve().parents[1] / "README.md"
TEXT = README.read_text()

# Imports that appear in fenced python blocks and are expected to work.
PYTHON_IMPORTS = re.findall(r"^from (thinkrl[\w.]*) import ([^\n]+)$", TEXT, flags=re.MULTILINE)


def test_readme_exists_and_mentions_thinkrl():
    assert "thinkrl" in TEXT.lower()


@pytest.mark.parametrize(("module", "names"), PYTHON_IMPORTS, ids=lambda v: str(v)[:40])
def test_every_documented_import_resolves(module, names):
    imported = importlib.import_module(module)
    for name in (n.strip() for n in names.split(",")):
        assert hasattr(imported, name), f"README documents {module}.{name}, which does not exist"


def test_unimplemented_trainers_are_marked_as_such():
    assert "CoT / ToT trainers are not implemented yet" in TEXT


def test_no_bare_cot_trainer_import_is_presented_as_working():
    working_imports = {f"{module}.{name.strip()}" for module, names in PYTHON_IMPORTS for name in names.split(",")}
    assert "thinkrl.training.CoTTrainer" not in working_imports
