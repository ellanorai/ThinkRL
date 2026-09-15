"""The Sphinx configuration has to actually document the package (#133), and the README
must not announce artifacts that are not in the repository (#134).

`extensions = []` meant the published site carried two pages and no reference for any of
the package, while the README linked it as the primary documentation entry.
"""

import pathlib
import re

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "docs" / "source" / "conf.py"


@pytest.mark.parametrize(
    "extension",
    ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon", "myst_parser"],
)
def test_the_extension_is_enabled(extension):
    assert extension in CONF.read_text(), f"{extension} missing, so the site documents less than it could"


def test_autosummary_actually_generates():
    assert "autosummary_generate = True" in CONF.read_text()


def test_the_package_is_importable_from_the_docs_build():
    """autodoc imports the package to read docstrings, so sys.path has to reach it."""
    assert 'sys.path.insert(0, os.path.abspath("../.."))' in CONF.read_text()


def test_optional_extras_are_mocked():
    """Building docs must not require deepspeed, vllm and friends to be installed."""
    conf = CONF.read_text()

    for package in ("deepspeed", "vllm", "fastapi"):
        assert f'"{package}"' in conf


def test_the_api_page_covers_the_top_level_packages():
    api = (ROOT / "docs" / "source" / "api.rst").read_text()

    for module in ("thinkrl.algorithms", "thinkrl.training", "thinkrl.rewards", "thinkrl.evaluation"):
        assert module in api


def test_the_markdown_guides_are_in_the_sphinx_sources():
    """docs/reinforce_pp.md is the most substantial writing in the repository and did not
    appear on the site at all, because it was not reachable from any toctree."""
    guides = ROOT / "docs" / "source" / "guides"

    for name in ("reinforce_pp", "grpo", "vllm_worker"):
        stub = guides / f"{name}.md"
        assert stub.exists(), f"{name} is not in the Sphinx sources"
        # Included rather than copied, so the file keeps the path the README links to.
        assert f"../../{name}.md" in stub.read_text()


def test_every_guide_stub_points_at_a_real_file():
    guides = ROOT / "docs" / "source" / "guides"

    for stub in guides.glob("*.md"):
        target = re.search(r"\{include\}\s+(\S+)", stub.read_text()).group(1)
        assert (stub.parent / target).resolve().exists(), f"{stub.name} includes a missing file"


def test_the_readme_does_not_announce_absent_benchmarks():
    """There are no benchmark results, scripts or configs in the repository. For an RL
    library the benchmark table is what tells a reader the implementation is correct, so
    announcing one that does not exist is the costliest claim to get wrong."""
    readme = (ROOT / "README.md").read_text()

    assert "Released benchmarks" not in readme


def test_dr_grpo_is_named_correctly():
    """Dr. GRPO is "GRPO Done Right", not "Distributionally Robust"."""
    readme = (ROOT / "README.md").read_text()

    assert "Distributionally Robust" not in readme
