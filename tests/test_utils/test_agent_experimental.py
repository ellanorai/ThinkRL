"""thinkrl/utils/agent.py implements an agent executor that nothing calls (#130).

432 lines referenced only by its own test file. The README's design section is built on
this idea, so the framing paragraph of the library described a module with no runtime path
into it. Marked experimental rather than deleted, because the design is the intended one.
"""

import inspect
import pathlib

from thinkrl.utils import agent


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_module_says_it_is_experimental():
    assert "experimental" in inspect.getdoc(agent).lower()


def test_the_docstring_names_the_reason_it_is_not_wired():
    """Multi-turn training needs a loss mask over model-generated tokens only, which the
    RL trainers do not do, so this is not a small wiring job."""
    doc = inspect.getdoc(agent)

    assert "loss mask" in doc
    assert "#130" in doc


def test_nothing_in_training_or_cli_constructs_an_executor():
    """Pins the claim. If this starts failing, the module got wired and the warning should
    come off rather than stay as stale text."""
    hits = []
    for area in ("thinkrl/training", "thinkrl/cli"):
        for path in (ROOT / area).rglob("*.py"):
            if "AgentExecutorBase" in path.read_text():
                hits.append(str(path))

    assert hits == [], f"agent executor is used in {hits}; update the experimental note"


def test_the_readme_does_not_present_it_as_a_shipped_abstraction():
    readme = (ROOT / "README.md").read_text()
    section = readme.split("### Token-in-Token-out Agents")[1].split("---")[0]

    assert "experimental" in section
    assert "direction rather" in section
