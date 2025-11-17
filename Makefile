# Makefile for ThinkRL
#
# Author: Archit Sood @ EllanorAI
#
# This Makefile provides convenience targets for common development tasks.
# It assumes a Python 3.8+ environment.

# ============================================================================
# Configuration
# ============================================================================

# Define the default Python interpreter
PYTHON3 := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

# Virtual environment directory
VENV_NAME ?= .venv
VENV_PATH := $(CURDIR)/$(VENV_NAME)

# Python interpreter within the virtual environment
PYTHON := $(VENV_PATH)/bin/python

# Source directories
SRC_DIRS := thinkrl tests examples

# Phony targets (targets that aren't files)
.PHONY: all install install-dev format lint test test-cov clean build docs help

# Default target
all: install-dev test lint

# ============================================================================
# Core Development Targets
# ============================================================================

## install: Install production dependencies
install:
	@echo "--- Installing production dependencies ---"
	$(PYTHON3) -m pip install -e .

## install-dev: Create a virtual environment and install all dev dependencies
install-dev: $(VENV_PATH)/bin/activate
$(VENV_PATH)/bin/activate: requirements.txt setup.py
	@echo "--- Creating virtual environment at $(VENV_PATH) ---"
	$(PYTHON3) -m venv $(VENV_NAME)
	@echo "--- Upgrading pip ---"
	$(PYTHON) -m pip install --upgrade pip
	@echo "--- Installing 'complete' development dependencies ---"
	$(PYTHON) -m pip install -e .[complete]
	@echo "--- Installing pre-commit hooks ---"
	$(PYTHON) -m pre-commit install
	@echo
	@echo "Virtual environment created and dependencies installed."
	@echo "Run 'source $(VENV_NAME)/bin/activate' to activate."
	@touch $(VENV_PATH)/bin/activate

## format: Run auto-formatters (black, isort)
format:
	@echo "--- Running formatters (black, isort) ---"
	$(PYTHON) -m black $(SRC_DIRS)
	$(PYTHON) -m isort $(SRC_DIRS)

## lint: Run linters (flake8, mypy)
lint:
	@echo "--- Running linter (flake8) ---"
	$(PYTHON) -m flake8 $(SRC_DIRS) --max-line-length=88 --extend-ignore=E203,W503,E501 --exclude=__pycache__,*.pyc,.git,build,dist,.tox,.pytest_cache
	@echo "--- Running type checker (mypy) ---"
	$(PYTHON) -m mypy thinkrl --ignore-missing-imports --no-strict-optional

## pre-commit: Run all pre-commit hooks on all files
pre-commit:
	@echo "--- Running all pre-commit hooks ---"
	$(PYTHON) -m pre-commit run --all-files

## test: Run pytest
test:
	@echo "--- Running tests ---"
	$(PYTHON) -m pytest tests/

## test-cov: Run pytest with coverage report
test-cov:
	@echo "--- Running tests with coverage ---"
	$(PYTHON) -m pytest tests/ --cov=thinkrl --cov-report=term-missing --cov-report=xml

# ============================================================================
# Build & Cleanup Targets
# ============================================================================

## build: Build the package (sdist and wheel)
build:
	@echo "--- Building package ---"
	rm -rf build dist
	$(PYTHON) -m build

## docs: Build the documentation (placeholder)
docs:
	@echo "--- Building documentation (placeholder) ---"
	# Example: $(PYTHON) -m sphinx -b html docs/ docs/_build/html
	@echo "Docs build target not yet implemented."

## clean: Remove build artifacts, caches, and venv
clean:
	@echo "--- Cleaning up project ---"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf $(VENV_NAME) build dist *.egg-info .coverage coverage.xml bandit-report.json
	@echo "Cleanup complete."

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'