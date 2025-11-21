# ------------------------------------------------------------------
# ThinkRL Makefile
# ------------------------------------------------------------------
.DEFAULT_GOAL := help
PYTHON := python3
PIP := pip
PYTEST := pytest

# PyTorch CUDA 12.1 Index URL (Matches requirements.txt)
TORCH_INDEX := https://download.pytorch.org/whl/cu121

.PHONY: install dev-install clean check test

install: ## Install production dependencies (CUDA enabled)
	$(PIP) install -e . --extra-index-url $(TORCH_INDEX)

dev-install: ## Install with Dev, Docs, and GPU support
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-docs.txt
	$(PIP) install -e ".[complete]" --extra-index-url $(TORCH_INDEX)
	pre-commit install

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage coverage_html
	find . -type d -name "__pycache__" -exec rm -rf {} +

# ------------------------------------------------------------------
# Quality Control
# ------------------------------------------------------------------
format: ## Auto-format code
	black thinkrl tests
	isort thinkrl tests

check: ## Run full strict quality suite (Black, Isort, Flake8, Mypy)
	black --check thinkrl tests
	isort --check-only thinkrl tests
	flake8 thinkrl tests
	mypy thinkrl

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
test: ## Run all tests
	$(PYTEST)

test-fast: ## Run CPU-only unit tests (skip slow integration)
	$(PYTEST) -m "not slow and not gpu"

test-gpu: ## Run only GPU/CuPy tests
	$(PYTEST) -m "gpu"

test-algos: ## Run specific algorithm checks (VAPO/DAPO)
	$(PYTEST) tests/test_algorithms