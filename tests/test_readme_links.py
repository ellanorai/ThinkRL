"""Relative links in the README must point at files that exist.

The training-guides section linked four scripts under examples/scripts/, a directory that
has never existed in the repository.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Markdown links whose target is a relative path, ignoring URLs and in-page anchors.
RELATIVE_LINKS = sorted(
    {
        target
        for target in re.findall(r"\]\(([^)]+)\)", README.read_text())
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    }
)


def test_readme_has_relative_links_to_check():
    assert RELATIVE_LINKS


@pytest.mark.parametrize("target", RELATIVE_LINKS)
def test_relative_link_resolves(target):
    path = (ROOT / target.split("#")[0]).resolve()
    assert path.exists(), f"README links {target}, which does not exist"


def test_linked_examples_are_not_empty():
    for target in RELATIVE_LINKS:
        path = ROOT / target.split("#")[0]
        if path.suffix == ".py" and path.exists():
            assert path.stat().st_size > 0, f"README links {target}, which is an empty file"
