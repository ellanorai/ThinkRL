# ------------------------------------------------------------------
# ThinkRL Makefile
# ------------------------------------------------------------------
.DEFAULT_GOAL := help
PYTHON := python3
PIP := pip
PYTEST := pytest

# PyTorch CUDA 12.1 Index URL (Matches requirements.txt)
TORCH_INDEX := https://download.pytorch.org/whl/cu121

.PHONY: install dev-install clean check test format lint

install: ## Install production dependencies (CUDA enabled)
	$(PIP) install -e . --extra-index-url $(TORCH_INDEX)

dev-install: ## Install with Dev, Docs, and GPU support
	$(PIP) install -e ".[complete]" --extra-index-url $(TORCH_INDEX)
	pre-commit install

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage coverage_html
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ------------------------------------------------------------------
# Quality Control
# ------------------------------------------------------------------
format: ## Auto-format code
	black thinkrl tests
	isort thinkrl tests

format-check: ## Check code formatting without modifying
	black --check thinkrl tests
	isort --check-only thinkrl tests

lint: ## Run all linters
	@echo "Running flake8..."
	flake8 thinkrl tests --max-line-length=88 --extend-ignore=E203,W503 --statistics
	@echo "Running mypy..."
	mypy thinkrl || true
	@echo "Running bandit..."
	bandit -r thinkrl -ll -ii

check: format-check lint ## Run full strict quality suite

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
test: ## Run all tests
	$(PYTEST) tests/ -v

test-fast: ## Run CPU-only unit tests (skip slow integration)
	$(PYTEST) tests/ -m "not slow and not gpu" -v

test-gpu: ## Run only GPU/CuPy tests
	$(PYTEST) tests/ -m "gpu" -v

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v \
		--cov=thinkrl \
		--cov-report=term-missing \
		--cov-report=html:coverage_html \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=58

test-algos: ## Run specific algorithm checks (VAPO/DAPO)
	$(PYTEST) tests/test_algorithms -v

# ------------------------------------------------------------------
# CI Helpers
# ------------------------------------------------------------------
ci-lint: format-check lint ## CI linting (fails on issues)

ci-test: ## CI testing with coverage
	$(PYTEST) tests/ -v \
		--cov=thinkrl \
		--cov-report=xml \
		--cov-fail-under=58 \
		--tb=short

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
help: ## Show this help message
	@echo "ThinkRL Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
```

## Branch Protection Rules (GitHub Settings)

Add these required status checks in GitHub Settings > Branches > Branch protection rules for `main`:
```
Required status checks:
- pre-commit / Pre-commit Checks
- lint / Code Quality Checks
- test / test (3.9)
- test / test (3.10)
- test / test (3.11)
- compatibility / compatibility (3.11)