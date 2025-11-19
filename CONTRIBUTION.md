-----

# 🤝 Contributing to ThinkRL

**Welcome to the ThinkRL community\!**

We are on a mission to democratize Reinforcement Learning from Human Feedback (RLHF) and Reasoning models. Whether you are fixing a bug, optimizing a CUDA kernel, improving documentation, or proposing the next SOTA algorithm like VAPO or DAPO, your contributions are essential to this goal.

## 📋 Table of Contents

  - [Code of Conduct](https://www.google.com/search?q=%23-code-of-conduct)
  - [Development Workflow](https://www.google.com/search?q=%23-development-workflow)
      - [1. Environment Setup](https://www.google.com/search?q=%231-environment-setup)
      - [2. The Development Loop](https://www.google.com/search?q=%232-the-development-loop)
      - [3. Running Tests](https://www.google.com/search?q=%233-running-tests)
  - [Coding Standards](https://www.google.com/search?q=%23-coding-standards)
  - [Submitting a Pull Request](https://www.google.com/search?q=%23-submitting-a-pull-request)
  - [Community & Support](https://www.google.com/search?q=%23-community--support)

-----

## 👮 Code of Conduct

We are committed to fostering an inclusive and respectful environment. Please read and follow our **[Code of Conduct](https://www.google.com/search?q=CODE_OF_CONDUCT.md)** in all project interactions, including GitHub issues and discussions.

-----

## 🛠 Development Workflow

### 1\. Environment Setup

We recommend using a virtual environment to keep dependencies isolated. You can use standard `venv` or the provided `Makefile` shortcuts.

**Prerequisites:**

  * Python 3.8+
  * CUDA 12.x (Recommended for GPU acceleration)

**Quick Start via Makefile:**
If you have `make` installed (Linux/MacOS), you can simply run:

```bash
git clone https://github.com/Archit03/ThinkRL.git
cd ThinkRL
make install-dev
```

**Manual Setup:**

```bash
# 1. Clone the repository
git clone https://github.com/Archit03/ThinkRL.git
cd ThinkRL

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  
# On Windows: .venv\Scripts\activate

# 3. Install the complete development environment (includes dev, test, and docs tools)
pip install -e .[complete]

# 4. Install pre-commit hooks (Crucial for linting)
pre-commit install
```

### 2\. The Development Loop

1.  **Find an Issue:** Check [Issues](https://github.com/Archit03/ThinkRL/issues) for tasks labeled `good first issue` or `help wanted`.
2.  **Fork & Branch:**
    ```bash
    git checkout -b feat/my-new-feature
    # or
    git checkout -b fix/memory-leak-patch
    ```
3.  **Hack Away:** Make your changes.
4.  **Format & Lint:** Before committing, run the formatters to ensure CI passes.
    ```bash
    make format  # Runs black and isort
    make lint    # Runs flake8 and mypy
    ```

### 3\. Running Tests

We use `pytest` for testing. Please ensure all tests pass before submitting your PR.

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov
```

> **Note on GPU Tests:** Some tests require a GPU. If you are developing on a CPU-only machine, use `pytest -m "not gpu"` to skip hardware-specific tests.

-----

## 📏 Coding Standards

To maintain a high-quality codebase, we strictly adhere to the following standards:

  * **Style:** We use [Black](https://github.com/psf/black) for formatting and [Isort](https://pycqa.github.io/isort/) for import sorting.
  * **Type Hinting:** All new functions and classes must have Python type hints. We check this with `mypy`.
  * **Docstrings:** Use **Google Style** docstrings.
      * *Why?* This allows us to auto-generate API documentation using Sphinx.
  * **Performance:**
      * Use **CuPy** (`cp`) instead of NumPy (`np`) for heavy numerical computations to leverage GPU acceleration.
      * Always provide a CPU fallback if a GPU is not detected.

**Example:**

```python
def compute_advantage(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE).

    Args:
        rewards (torch.Tensor): The rewards per timestep.
        values (torch.Tensor): The value function estimates.

    Returns:
        torch.Tensor: The computed advantages.
    """
    # Implementation...
```

-----

## 🚀 Submitting a Pull Request

1.  **Commit Messages:** Use [Conventional Commits](https://www.conventionalcommits.org/).
      * `feat: add VAPO algorithm support`
      * `fix: resolve memory leak in multimodal trainer`
      * `docs: update README with installation steps`
2.  **Push:** Push your branch to your fork (`git push origin feat/my-feature`).
3.  **Open PR:** Open a Pull Request against the `main` branch of `Archit03/ThinkRL`.
4.  **Description:** Clearly describe *what* you changed and *why*. Link any relevant issues (e.g., `Closes #123`).
5.  **CI Checks:** Ensure the GitHub Actions (Tests, Linting, TruffleHog) pass.

-----

## 📚 Documentation Contributions

Documentation is located in the `docs/` directory. We use **Sphinx** and **reStructuredText/Markdown**.

To build documentation locally:

```bash
# Ensure you installed with [docs] extra
pip install -e .[docs]

cd docs
make html
# Open docs/build/html/index.html in your browser
```

-----

## 💬 Community & Support

  * **GitHub Discussions:** Best for feature requests, ideas, and Q\&A.
  * **Issues:** Best for bug reports and specific tasks.
  * **Email:** Contact maintainers at `architsood@ellanorai.org` for private matters.

### Project Structure

  * `thinkrl/`: The core package.
      * `algorithms/`: RL implementations (PPO, DAPO, GRPO).
      * `models/`: Model definitions.
      * `utils/`: Logging, metrics, and helpers.
  * `tests/`: Unit and integration tests.
  * `examples/`: Runnable scripts demonstrating library usage.

-----

**Built with ❤️ by EllanorAI in India 🇮🇳**