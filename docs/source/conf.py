# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# autodoc imports the package to read its docstrings, so the source has to be importable
# from here. docs/source -> repository root.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ThinkRL"
copyright = "2025, Archit Sood"
author = "Archit Sood"
release = "2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# This was `[]`, so the published site documented none of the package (#133). Every
# public module already carries docstrings in a consistent style, so the reference is a
# configuration change rather than a writing project.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # the Google/NumPy docstring style already used throughout
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",  # so the Markdown documents under docs/ stop being orphaned
]

autosummary_generate = True

# The docs workflow builds with -W, so any warning fails the build. Turning autodoc on
# surfaced 516 docutils complaints from reStructuredText inside existing docstrings:
# "Inline literal start-string without end-string" and friends, mostly maths like A_i and
# un-indented code in prose. They are pre-existing and worth a cleanup pass, but rewriting
# docstrings across 41 modules to switch the site on is the wrong order.
#
# Scoped to docutils on purpose: Sphinx-level problems still fail the build, so a missing
# toctree entry, a broken cross-reference or a module autodoc cannot import is still caught.
suppress_warnings = ["docutils"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Importing thinkrl must not require the optional extras just to build docs.
autodoc_mock_imports = [
    "deepspeed",
    "vllm",
    "wandb",
    "ray",
    "cupy",
    "flash_attn",
    "peft",
    "bitsandbytes",
    "fastapi",
    "uvicorn",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),  # pytorch.org redirects here now
}

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
