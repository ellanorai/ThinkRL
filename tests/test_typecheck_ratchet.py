"""#145: `mypy thinkrl` reported 191 errors across 43 of 107 modules and the CI step
carried continue-on-error, so the type check had never gated anything and new code was
never checked.

The 64 clean modules are gated now, and the debt is an explicit list whose length is the
metric. These tests exist so the list can only shrink.
"""

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]

# The count when the ratchet went in. Lower it when you clear a module; never raise it.
DEBT_AT_RATCHET = 43


def _ratchet_modules() -> list[str]:
    """Read the list out of pyproject.toml without a TOML parser.

    tomllib is 3.11+ and this project supports 3.10, so importing it here passed locally
    and broke the 3.10 job. The list is a flat array of quoted strings, so pulling it out
    directly is enough and costs no dependency.
    """
    text = (ROOT / "pyproject.toml").read_text()

    marker = "# --- #145 type-check ratchet"
    assert marker in text, "the #145 ratchet block is gone from pyproject.toml"

    block = text.split(marker, 1)[1]
    array = re.search(r"module = \[(.*?)\]", block, re.DOTALL)
    assert array, "the #145 ratchet list is gone from pyproject.toml"

    return re.findall(r'"([^"]+)"', array.group(1))


def test_the_debt_list_never_grows():
    """A new module with type errors should be fixed, not appended to the list."""
    modules = _ratchet_modules()

    assert len(modules) <= DEBT_AT_RATCHET, (
        f"{len(modules)} modules are excluded from type checking, up from {DEBT_AT_RATCHET}. "
        "Fix the new module rather than adding it here."
    )


def test_the_list_is_kept_tidy():
    modules = _ratchet_modules()

    assert modules == sorted(modules), "keep the ratchet list sorted so diffs stay readable"
    assert len(modules) == len(set(modules)), "duplicate entry in the ratchet list"


def test_every_excluded_module_still_exists():
    """An entry for a deleted module is silent dead weight that inflates the metric."""
    for module in _ratchet_modules():
        path = ROOT / (module.replace(".", "/") + ".py")
        package = ROOT / module.replace(".", "/") / "__init__.py"

        assert path.exists() or package.exists(), f"{module} is excluded but does not exist"


def test_mypy_actually_gates():
    """The step ran with continue-on-error, so it reported and never blocked."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    mypy_step = workflow.split("- name: Run MyPy")[1]

    assert "continue-on-error" not in mypy_step.split("- name:")[0]
