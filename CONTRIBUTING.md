Contributing to ThinkRL
Thank you for your interest in contributing to ThinkRL! We’re thrilled to grow a vibrant community around this open-source Reinforcement Learning from Human Feedback (RLHF) library. Whether you’re fixing bugs, adding features, enhancing documentation, or sharing ideas, your contributions make ThinkRL better for everyone.

Getting Started
1. Code of Conduct
We are committed to fostering an inclusive and respectful environment. Please read and follow our Code of Conduct in all project interactions.
2. Finding Issues

Explore the Issues page for tasks labeled good first issue or help wanted.
Comment on an issue to express interest or discuss your approach before starting work.
For new feature proposals, open an issue to discuss with maintainers before coding.

3. Development Setup
To set up ThinkRL for development:
# Clone the repository
git clone https://github.com/Archit03/ThinkRL.git
cd ThinkRL

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the complete development environment
pip install -e .[complete]

# Install pre-commit hooks
pre-commit install

The [complete] extra includes all dependencies (e.g., cupy-cuda12x, transformers, deepspeed, pytest) needed for development and testing.

Contribution Workflow
1. Fork and Branch

Fork the repository to your GitHub account.
Create a feature branch:git checkout -b feature/your-feature-name



2. Code Standards

Adhere to PEP 8 for Python code style.
Write clear, concise docstrings in Google style.
Ensure compatibility with Python 3.8+.
Use CuPy (instead of NumPy) for numerical computations to leverage GPU acceleration.

3. Testing

Add unit tests for new features or bug fixes in the tests/ directory.
Run tests locally:pytest


Strive for 100% test coverage for new code.
Ensure all tests pass before submitting a pull request.

4. Commit Messages

Write clear, descriptive commit messages using the Conventional Commits format. Examples:feat: add VAPO algorithm support
fix: resolve memory leak in multimodal trainer
docs: update README with installation steps



5. Pull Request (PR)

Push your branch to your fork:git push origin feature/your-feature-name


Open a pull request against the main branch of Archit03/ThinkRL.
Provide a clear description of your changes, linking relevant issues (e.g., Closes #123).
Ensure your PR passes all CI checks (tests, linting).
Respond to maintainer feedback promptly.


Documentation Contributions

Enhance documentation in the docs/ directory or update README.md.
Use clear, consistent markdown formatting.
Contribute tutorials or examples to docs/tutorials/ or examples/ to aid users.

Reporting Bugs

Use the bug report template when opening an issue.
Include:
ThinkRL version
Python version
Operating system
Steps to reproduce
Expected vs. actual behavior



Feature Requests

Submit ideas using the feature request template.
Describe the use case, proposed solution, and potential impact.

Community

Engage in discussions on our GitHub Discussions page.
Contact maintainers at archit@ellanorai.org for questions or feedback.

### Project Structure
- `thinkrl/`: The main package directory, containing `__init__.py` and submodules like `scripts/`.
- `tests/`: Test files for the project.
- `setup.py`: Package installation script.

Acknowledgments
All contributors will be credited in our CONTRIBUTORS.md file and release notes. Thank you for advancing AI innovation with ThinkRL!


  ⭐ Ready to contribute? Star us on GitHub and get started!
  Built with ❤️ by EllanorAI in India 🇮🇳
