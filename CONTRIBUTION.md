# 🤝 Contributing to ThinkRL

**Welcome to the ThinkRL community!**

We're building the next-generation RLHF library with state-of-the-art algorithms, reasoning capabilities, and production-ready infrastructure. Whether you're fixing a bug, implementing a new algorithm, optimizing performance, or improving documentation, your contributions are essential to democratizing reinforcement learning from human feedback.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
  - [Project Structure](#project-structure)
- [Core Technologies & Libraries](#core-technologies--libraries)
  - [PyTorch Ecosystem](#pytorch-ecosystem)
  - [GPU Acceleration with CuPy](#gpu-acceleration-with-cupy)
  - [vLLM Integration](#vllm-integration)
  - [HuggingFace Transformers](#huggingface-transformers)
  - [Other Key Libraries](#other-key-libraries)
- [Development Workflow](#development-workflow)
  - [Branch Strategy](#branch-strategy)
  - [Coding Standards](#coding-standards)
  - [Testing Requirements](#testing-requirements)
  - [Documentation Standards](#documentation-standards)
- [Implementation Guides](#implementation-guides)
  - [Adding New Algorithms](#adding-new-algorithms)
  - [Adding New Models](#adding-new-models)
  - [Adding Utilities](#adding-utilities)
  - [Integrating with vLLM](#integrating-with-vllm)
- [Performance Guidelines](#performance-guidelines)
- [Submitting Changes](#submitting-changes)
- [Community & Support](#community--support)
- [Learning Resources](#learning-resources)

---

## 👮 Code of Conduct

We are committed to fostering an inclusive, respectful, and collaborative environment. All contributors must adhere to our Code of Conduct. Be kind, be constructive, and help us build something amazing together.

**Key Principles:**
- Treat everyone with respect and professionalism
- Provide constructive feedback, not criticism
- Be patient with newcomers and help them learn
- Focus on the problem, not the person
- Celebrate diverse perspectives and approaches

---

## 🚀 Getting Started

### Prerequisites

**System Requirements:**
- **Python**: Version 3.10, 3.11, or 3.12
- **CUDA**: Version 12.1 or higher for GPU acceleration
- **RAM**: Minimum 16GB, recommended 32GB+ for training
- **GPU**: 16GB+ VRAM for full training pipelines
- **Storage**: At least 50GB free space for models and datasets

**Development Tools:**
- **Git**: For version control
- **IDE**: VS Code, PyCharm, or similar with Python support
- **Terminal**: Bash, Zsh, or PowerShell
- **Docker** (optional): For containerized development

**Recommended Operating Systems:**
- Linux (Ubuntu 22.04 or later preferred)
- macOS (with Homebrew for dependencies)
- Windows 11 (with WSL2 for best compatibility)

### Development Setup

**Step 1: Clone the Repository**
- Fork the ThinkRL repository on GitHub to your account
- Clone your fork locally using Git
- Add the upstream repository as a remote for syncing future updates

**Step 2: Set Up Virtual Environment**
- Create an isolated Python environment using venv or conda
- Activate the environment before any development work
- This prevents dependency conflicts with your system packages

**Step 3: Install Dependencies**
- Install the package in editable mode with the "complete" extra flag
- This includes all development, testing, and documentation tools
- Install pre-commit hooks immediately after setup (this is mandatory)
- Pre-commit hooks will automatically run code quality checks before each commit

**Key Dependencies Installed:**
- **Core ML**: PyTorch (with CUDA support), CuPy, transformers, accelerate
- **RLHF Libraries**: datasets, tokenizers, PEFT, vLLM
- **Testing**: pytest, pytest-cov, pytest-xdist, pytest-mock
- **Code Quality**: ruff, mypy, bandit, pre-commit
- **Documentation**: Sphinx, myst-parser, nbsphinx

**Step 4: Verify Your Setup**
- Run a simple test to ensure pytest is working correctly
- Check that code quality tools (ruff, mypy) are accessible from command line
- Verify GPU availability if you have CUDA installed by checking PyTorch CUDA detection
- Confirm pre-commit hooks are active by attempting a test commit

### Project Structure

Understanding the codebase organization is crucial for effective contributions:

**Main Package (`thinkrl/`):**
- **algorithms/**: RL algorithm implementations (PPO, VAPO, DAPO, GRPO, DPO, REINFORCE)
- **models/**: Model architectures (GPT, LLaMA, Qwen, multimodal models)
- **training/**: Training loops, distributed training coordination, trainer classes
- **data/**: Dataset loaders, data processors, custom data formats
- **utils/**: Core utilities (logging, metrics, checkpointing, tokenization)
- **reasoning/**: Chain-of-Thought and Tree-of-Thought implementations
- **evaluation/**: Evaluation metrics, benchmarking tools
- **peft/**: Parameter-efficient fine-tuning (LoRA, QLoRA)
- **integration/**: External integrations (vLLM client, API connections)
- **scripts/**: Command-line interface scripts for training and inference

**Supporting Directories:**
- **tests/**: Comprehensive test suite mirroring main package structure
- **configs/**: YAML configuration templates for algorithms, models, training
- **examples/**: Runnable example scripts and Jupyter notebooks
- **docs/**: Documentation source files for Sphinx
- **.github/workflows/**: CI/CD pipeline definitions (testing, linting, deployment)

---

## 🔧 Core Technologies & Libraries

ThinkRL is built on cutting-edge libraries optimized for performance, scalability, and ease of use. Understanding these technologies is essential for meaningful contributions.

### PyTorch Ecosystem

**Role**: Primary deep learning framework for all model operations and training.

**Version**: PyTorch 2.0+ with CUDA 12.1 support

**Key Capabilities:**
- Automatic differentiation for gradient computation
- GPU acceleration via CUDA
- Distributed training via torch.distributed
- JIT compilation with torch.compile for performance optimization
- Mixed-precision training with torch.autocast
- Gradient checkpointing for memory efficiency
- Model parallelism and data parallelism support

**Usage Guidelines:**
- Always check GPU availability before moving tensors to CUDA
- Use torch.cuda.is_available() for device detection
- Detach tensors from computation graph before converting to NumPy arrays
- Leverage torch.compile() for production code to enable JIT optimization
- Use DistributedDataParallel for multi-GPU training
- Apply gradient checkpointing for large models to reduce memory usage
- Set seeds for reproducibility across runs

**Integration Points:**
- All model definitions must inherit from torch.nn.Module
- All loss computations should use PyTorch operations for autograd compatibility
- Optimizers should be torch.optim classes (AdamW, SGD, etc.)
- Data loading must go through torch.utils.data.DataLoader

**Common Patterns:**
- Initialize models and move to device
- Set training/evaluation modes appropriately
- Clear gradients before backward pass
- Clip gradients to prevent exploding gradients
- Use torch.no_grad() for inference to save memory

### GPU Acceleration with CuPy

**Role**: Zero-copy GPU operations for metrics and statistical computations.

**Purpose**: Accelerate numerical operations on GPU tensors without costly CPU transfers.

**Key Benefits:**
- Direct GPU array operations without PyTorch overhead
- Zero-copy conversion from PyTorch tensors via DLPack protocol
- Full NumPy-compatible API for familiar usage
- Advanced statistical functions via cupyx.scipy library

**When to Use CuPy:**
- Computing statistics on large GPU tensors (mean, standard deviation, percentiles)
- Batch metric calculations during training loops
- Custom mathematical operations not available in PyTorch
- Post-processing of model outputs for logging and analysis
- Advanced statistical measures (skewness, kurtosis, entropy)

**Implementation Requirements:**
- Always provide NumPy fallback for CPU compatibility
- Handle ImportError gracefully (CuPy may not be installed in all environments)
- Handle OSError gracefully (CUDA libraries may be missing)
- Use DLPack protocol for zero-copy tensor conversion
- Document GPU acceleration behavior in function docstrings
- Test both GPU and CPU code paths

**Critical Pattern:**
- Check if CuPy is available in the environment
- Check if tensor is located on GPU device
- If both conditions are true, convert via DLPack and use CuPy operations
- Otherwise, convert to NumPy on CPU and use standard NumPy operations
- Always return results in consistent format regardless of code path taken

**Performance Considerations:**
- Avoid unnecessary data transfers between CPU and GPU
- Batch operations when possible to amortize kernel launch overhead
- Use CuPy for operations that will remain on GPU
- Fall back to NumPy for operations that require CPU data anyway

**See Reference**: Check `thinkrl/utils/metrics.py` for production-quality implementation examples

### vLLM Integration

**Role**: High-throughput inference engine for RLAIF data generation and online RL.

**Architecture Overview:**

ThinkRL uses a "bridge" architecture to connect training and inference processes:

**Component 1 - Training Process (Rank 0):**
- Manages model training and weight updates
- Coordinates rollout generation from vLLM
- Broadcasts updated weights to vLLM server via NCCL
- Handles training data collection and processing

**Component 2 - vLLM Server (Rank 1):**
- Runs as separate inference process on dedicated GPU(s)
- Handles batch generation requests with high throughput
- Receives weight updates from training process via NCCL bridge
- Optimizes inference with PagedAttention and continuous batching

**Component 3 - NCCL Bridge:**
- Provides low-latency weight synchronization between processes
- Uses StatelessProcessGroup for isolation from training communications
- Operates independently of training distributed data parallel communications
- Enables efficient parameter broadcasts without serialization overhead

**Integration Workflow:**

**Step 1: Server Initialization**
- Start vLLM server on dedicated GPU(s) with specified model
- Configure tensor parallelism based on available GPUs
- Set server port for HTTP API access

**Step 2: Client Setup**
- Initialize vLLM client in training code (rank 0 only)
- Configure NCCL bridge with unique port (must not conflict with training)
- Initialize weight synchronization with device specification

**Step 3: Generation Cycle**
- Sample prompts from training dataset
- Send generation requests to vLLM server via HTTP API
- Collect generated responses for training

**Step 4: Weight Update**
- Train model on collected rollouts
- Broadcast updated weights from training process to vLLM
- vLLM updates its internal model without restart

**Step 5: Repeat**
- Continue generation-training-update cycle iteratively

**Critical Considerations:**
- Only rank 0 in training process should communicate with vLLM server
- Use separate NCCL bridge port to avoid conflicts with training communications
- Handle FSDP/DDP model contexts appropriately during weight broadcasts
- Ensure vLLM and trainer use compatible model architectures and tokenizers
- Test integration with small models before scaling to large models
- Monitor memory usage on both training and inference GPUs

**Configuration Requirements:**
- vLLM server URL and port for HTTP API
- NCCL bridge port for weight synchronization (default: 51216)
- Tensor parallel size for vLLM inference
- Generation parameters (temperature, top_p, max_tokens, etc.)
- Model architecture compatibility between trainer and vLLM

**Troubleshooting Tips:**
- Verify NCCL libraries are installed correctly
- Check port availability before starting server
- Ensure model architectures match exactly
- Monitor GPU memory on inference nodes
- Test with synchronous generation before enabling asynchronous

**See Reference**: Check `thinkrl/integration/vllm_client.py` for complete implementation details

### HuggingFace Transformers

**Role**: Model loading, tokenization, and integration with pre-trained models.

**Version**: 4.35.0 or higher recommended

**Primary Uses:**
- Loading pre-trained models from HuggingFace Hub
- Tokenizer initialization and configuration
- Model configuration management
- Fine-tuning pre-trained checkpoints
- Accessing state-of-the-art model architectures

**Critical Integration Points:**

**Model Loading Best Practices:**
- Use AutoModelForCausalLM for language model architectures
- Specify torch_dtype (float16, bfloat16) for memory efficiency
- Use device_map="auto" for automatic multi-GPU device placement
- Set trust_remote_code appropriately (security consideration for custom code)
- Cache models locally to avoid repeated downloads from Hub

**Tokenizer Configuration Guidelines:**
- Always verify and set pad_token if not present in tokenizer
- Set padding_side to "left" for generation tasks
- Set padding_side to "right" for training tasks
- Handle special tokens (bos, eos, unk, pad) consistently
- Verify tokenizer max_length matches model's position embedding size
- Test tokenization output before starting training

**Model Configuration:**
- Review model config for architecture-specific parameters
- Adjust max_position_embeddings if needed for longer sequences
- Configure attention mechanisms (flash attention, etc.)
- Set appropriate dropout rates for fine-tuning

**Best Practices:**
- Use fast tokenizers (Rust-based) when available for better performance
- Align tokenizer vocabulary with model embedding layer
- Test tokenization/detokenization round-trip for correctness
- Handle unknown tokens appropriately in your application
- Document any custom tokenizer modifications

**Common Patterns:**
- Load model and tokenizer together
- Ensure model and tokenizer use same vocabulary
- Resize token embeddings if adding new special tokens
- Save model and tokenizer together for reproducibility

**See Reference**: Check `thinkrl/utils/tokenizer.py` for tokenizer utility functions

### Other Key Libraries

**Accelerate (HuggingFace):**
- **Purpose**: Simplifies distributed training across multiple GPUs and nodes
- **Features**: Automatic mixed precision, gradient accumulation, device placement automation
- **Usage**: Wrap training loop with Accelerator class for transparent multi-GPU scaling
- **Benefits**: Same code works on single GPU, multi-GPU, and multi-node setups

**PEFT (Parameter-Efficient Fine-Tuning):**
- **Purpose**: Efficient fine-tuning with LoRA, QLoRA, and other adapter methods
- **Benefits**: Dramatically reduced memory usage, faster training, much smaller checkpoints
- **Integration**: Wrap base models with PEFT adapters before training
- **Supported Methods**: LoRA, AdaLoRA, IA3, Prefix Tuning, P-Tuning

**Datasets (HuggingFace):**
- **Purpose**: Efficient data loading and processing pipelines
- **Features**: Streaming for large datasets, disk caching, memory-mapped files
- **Usage**: Load datasets from HuggingFace Hub or custom local files
- **Benefits**: Consistent API across different data sources

**DeepSpeed (Optional):**
- **Purpose**: Large-scale distributed training with ZeRO optimization stages
- **Features**: Model parallelism, optimizer state sharding, gradient checkpointing
- **Usage**: Requires separate JSON configuration file with ZeRO settings
- **When to Use**: Models with 10B+ parameters or limited GPU memory

**SafeTensors:**
- **Purpose**: Safe model serialization format without pickle
- **Benefits**: Fast loading, prevents arbitrary code execution vulnerabilities
- **Usage**: Preferred format for saving and loading model weights
- **Security**: Eliminates risks from malicious pickle files

**wandb (Weights & Biases):**
- **Purpose**: Experiment tracking and visualization platform
- **Features**: Real-time metrics logging, hyperparameter tracking, artifact storage
- **Usage**: Initialize at training start, log metrics throughout training
- **Integration**: Seamless with PyTorch training loops

**SciPy:**
- **Purpose**: Advanced statistical computations (CPU fallback for CuPy)
- **Usage**: Skewness, kurtosis, and other higher-order statistics
- **Note**: CPU-only library, use CuPy for GPU-accelerated equivalents

**Performance Comparison Matrix:**

| Library | GPU Support | Primary Use Case | When to Use |
|---------|-------------|------------------|-------------|
| PyTorch | Full | Model training | Always (core framework) |
| CuPy |  Full | Statistical operations | GPU tensors only |
| vLLM |  Full | High-throughput inference | RLAIF/online RL pipelines |
| Transformers | Full | Model loading | Pre-trained model usage |
| Accelerate |  Full | Distributed training | Multi-GPU setups |
| PEFT |  Full | Efficient fine-tuning | Large model training |
| DeepSpeed | Full | Large-scale training | 10B+ parameter models |
| Datasets | Partial | Data loading | Large dataset handling |
| SafeTensors | Full | Model serialization | All checkpointing |
| wandb |  No | Experiment tracking | All experiments |
| SciPy |  No | Statistics (CPU) | CPU fallback only |

---

## 📝 Development Workflow

### Branch Strategy

**Repository Structure:**
- **main**: Production-ready code (protected branch, no direct commits allowed)
- **dev**: Active development branch (default branch for all pull requests)
- **feature/[name]**: New features and enhancements (branch from dev)
- **fix/[name]**: Bug fixes (branch from dev)
- **docs/[name]**: Documentation updates (branch from dev)

**Standard Workflow Steps:**

**1. Fork and Branch:**
- Fork the ThinkRL repository to your GitHub account
- Clone your fork locally to your development machine
- Create a new feature branch from the dev branch
- Use descriptive branch names (e.g., feature/add-vapo-algorithm, fix/memory-leak-in-training)

**2. Development:**
- Make your changes in the feature branch
- Commit frequently with clear, descriptive commit messages
- Follow conventional commit format (feat:, fix:, docs:, etc.)
- Keep commits focused on single logical changes

**3. Code Quality:**
- Run code formatting and linting tools before committing
- Ensure all tests pass locally
- Add new tests for new functionality
- Update documentation for changed behavior

**4. Submit Pull Request:**
- Push your feature branch to your fork
- Create pull request targeting the dev branch (not main)
- Fill out pull request template completely
- Link related issues using GitHub keywords
- Request review from maintainers

**5. Code Review:**
- Address feedback from reviewers
- Push additional commits to same branch
- Respond to comments and questions
- Be open to suggestions and improvements

**6. Merge:**
- After approval, maintainers will merge to dev
- dev branch is periodically merged to main for releases
- Delete your feature branch after successful merge

**Branch Naming Conventions:**
- Use lowercase letters with hyphens for separation
- Start with category prefix (feature/, fix/, docs/)
- Keep names concise but descriptive
- Examples: feature/add-grpo, fix/gpu-memory-leak, docs/update-api-reference

### Coding Standards

#### 1. Code Formatting with Ruff

**Tool**: Ruff v0.1.14 - a fast, unified replacement for Black, isort, and Flake8

**What Ruff Does:**
- Lints code for style violations and potential bugs
- Automatically fixes many issues (unused imports, formatting, etc.)
- Formats code to consistent style following PEP 8
- Runs automatically via pre-commit hooks on every commit

**How to Use Ruff:**

**Check for Issues:**
- Run Ruff check command on source directories
- Review reported issues and warnings
- Note which issues can be auto-fixed

**Auto-Fix Issues:**
- Run Ruff with fix flag to automatically correct fixable issues
- Review changes before committing
- Manually fix any remaining issues that can't be auto-fixed

**Format Code:**
- Run Ruff format command to apply consistent formatting
- This reformats code according to configured style
- Similar to Black formatting but faster

**Pre-commit Integration:**
- Ruff runs automatically when you attempt to commit changes
- Commits are blocked if Ruff finds unfixable issues
- Fix reported issues and retry the commit
- This ensures all committed code meets quality standards

**Configuration:**
- Ruff settings are in pyproject.toml file
- Line length set to 100 characters
- Follows Google Python style guide
- Excludes generated files and migrations

#### 2. Type Hints (Mandatory)

**Requirements:**
- All function parameters must have type annotations
- All function return values must have type annotations
- Use typing module for complex types
- Add type hints to class attributes
- Use TypeVar for generic types when appropriate

**Type Checking:**
- Run mypy static type checker on your code before submitting
- Fix all type errors and warnings reported
- Use "type: ignore" comments sparingly with justification
- Aim for zero mypy errors in new code

**Common Type Patterns:**
- Use torch.Tensor for PyTorch tensors
- Use Dict[str, Any] for flexible dictionaries with string keys
- Use Optional[T] for values that can be None
- Use Union[T1, T2] for parameters accepting multiple types
- Use List[T], Tuple[T, ...] for collections
- Use Callable[[Args], Return] for function types

**Benefits of Type Hints:**
- Catch bugs early before runtime
- Improve code readability and documentation
- Enable better IDE autocomplete and refactoring
- Make function contracts explicit
- Facilitate code review and maintenance

#### 3. Docstring Requirements (Google Style)

**Mandatory Elements:**

**Summary Line:**
- Brief one-line summary of function purpose
- Keep under 80 characters
- Use imperative mood ("Compute" not "Computes")
- End without period

**Detailed Description (if needed):**
- Provide additional context and explanation
- Explain algorithms or mathematical operations
- Note important side effects or assumptions
- Reference related functions or papers

**Args Section:**
- List all parameters with types and descriptions
- Describe expected shapes for tensors
- Note default values and their meanings
- Explain valid ranges or constraints

**Returns Section:**
- Describe return value and its structure
- Specify tensor shapes and dtypes
- Explain dictionary keys if returning dict
- Note special return values (None, empty list, etc.)

**Raises Section:**
- List all exceptions that can be raised
- Explain conditions that trigger each exception
- Help users handle errors appropriately

**Example Section:**
- Show typical usage with realistic inputs
- Include expected outputs
- Demonstrate edge cases if relevant
- Keep examples runnable and tested

**Docstring Standards:**
- Write in imperative mood for consistency
- Be specific about tensor shapes (e.g., [batch_size, seq_len, hidden_dim])
- Document side effects like file I/O or state modifications
- Include computational complexity for algorithms (O(n), O(n²), etc.)
- Reference academic papers for algorithm implementations
- Use consistent terminology across codebase

**What to Document:**
- All public classes, methods, and functions
- All public modules with module-level docstrings
- Module-level constants and variables
- Complex private functions that aren't immediately obvious
- Any code that future contributors need to understand

**Example Structure:**
- Start with concise summary
- Add detailed description paragraph
- Document all parameters with types
- Describe return value structure
- List possible exceptions
- Provide usage example

#### 4. Security Checks

**Tool**: Bandit for static security analysis of Python code

**What Bandit Checks:**
- SQL injection vulnerabilities in database queries
- Command injection risks in subprocess calls
- Unsafe deserialization (pickle, yaml.load)
- Hardcoded credentials, passwords, API keys
- Weak cryptographic algorithms (MD5, DES)
- Insecure random number generation
- Path traversal vulnerabilities
- Use of eval() or exec()

**How to Run Bandit:**
- Run on entire source directory recursively
- Use low severity level to catch all potential issues
- Review all reported issues in output
- Exclude false positives in configuration if needed

**Handling Bandit Reports:**
- Fix high and medium severity issues before submitting PR
- Document any intentional security exceptions with comments
- Consult with maintainers if unsure about security implications
- Add security-related tests for fixed vulnerabilities

**Security Best Practices:**
- Never commit API keys, tokens, or passwords
- Use environment variables for sensitive configuration
- Validate all user inputs before use
- Use parameterized queries for databases
- Prefer secrets management solutions
- Keep dependencies updated for security patches

### Testing Requirements

#### Test Organization

**Directory Structure:**
- Mirror the main package structure in tests directory
- test_utils/ for testing utility functions
- test_algorithms/ for testing algorithm implementations
- test_models/ for testing model architectures
- test_data/ for testing data loading and processing
- test_integration/ for end-to-end integration tests

**Test File Naming:**
- Prefix all test files with "test_"
- Name files after the module being tested
- Example: tests/test_utils/test_metrics.py tests thinkrl/utils/metrics.py
- Keep one test file per source module generally

**Test Class Organization:**
- Group related tests in test classes
- Use descriptive class names (e.g., TestAdvantageComputation)
- One class per major functionality or class being tested
- Use setup and teardown methods for common initialization

#### Running Tests

**Basic Test Execution:**

**Run All Tests:**
- Execute pytest on entire test directory
- Use verbose flag for detailed output
- Monitor test execution time and memory usage

**Run Specific Tests:**
- Execute pytest on specific test files for targeted testing
- Run specific test classes using :: separator
- Run individual test methods for focused debugging

**Generate Coverage Reports:**
- Run pytest with coverage plugin enabled
- Generate HTML coverage reports for visualization
- Review coverage reports to identify untested code
- Aim to meet or exceed coverage thresholds

**Filter Tests with Markers:**
- Run quick tests by excluding slow marker
- Run GPU tests only when CUDA is available
- Run integration tests separately from unit tests
- Use custom markers for specific test categories

**Common Testing Commands:**

**Quick Development Testing:**
- Run without slow tests for rapid feedback
- Run specific modules during active development
- Use parallel execution for faster results

**Full Test Suite:**
- Run all tests including slow ones before submitting PR
- Generate coverage report to verify threshold
- Check both unit and integration tests pass

**GPU-Specific Testing:**
- Run GPU tests on machines with CUDA
- Skip GPU tests on CPU-only machines
- Verify both CPU and GPU code paths work

**Parallel Execution:**
- Use pytest-xdist for parallel test execution
- Significantly faster on multi-core machines
- Number of workers should match CPU cores

#### Test Markers

**Available Markers:**

**@pytest.mark.slow:**
- Use for tests taking over 10 seconds
- Typically long training loops or large dataset processing
- Excluded from quick development test runs
- Always run in CI pipeline before merge

**@pytest.mark.gpu:**
- Use for tests requiring CUDA-enabled GPU
- Include skip condition for CPU-only systems
- Test GPU-specific functionality and performance
- Verify numerical correctness against CPU baseline

**@pytest.mark.integration:**
- Use for end-to-end integration tests
- Test multiple components working together
- May require external services (vLLM, databases)
- Typically slower than unit tests

**@pytest.mark.requires_vllm:**
- Use for tests needing vLLM server running
- Include skip condition if vLLM unavailable
- Test vLLM client integration
- Verify weight synchronization works correctly

**When to Use Markers:**

**GPU Marker Usage:**
- Mark all tests that call .cuda() or create GPU tensors
- Include conditional skip if CUDA unavailable
- Test GPU-accelerated functionality
- Verify GPU memory management

**Slow Marker Usage:**
- Mark tests exceeding 10 second threshold
- Training loops with multiple epochs
- Large model inference tests
- Extensive data processing tests

**Integration Marker Usage:**
- Tests spanning multiple modules
- Tests requiring external services
- End-to-end pipeline tests
- Tests with file I/O or network operations

**Combining Markers:**
- Tests can have multiple markers
- Example: GPU + slow for large model training test
- Example: integration + requires_vllm for vLLM pipeline test

#### Coverage Requirements

**Thresholds:**
- Overall project coverage: minimum 58% (enforced by CI)
- New code contributions: aim for 80% or higher coverage
- Core algorithm implementations: target 100% coverage
- Utility functions: target 90% or higher coverage
- Documentation and examples: not included in coverage

**Coverage Best Practices:**

**Test Both Success and Failure Paths:**
- Test expected behavior with valid inputs
- Test error handling with invalid inputs
- Test boundary conditions and edge cases
- Test exception raising and error messages

**Include Edge Cases:**
- Empty inputs (empty lists, zero-length tensors)
- Single element inputs
- Maximum size inputs
- Negative values where applicable
- None values for optional parameters

**Test Error Handling:**
- Verify appropriate exceptions are raised
- Check error messages are informative
- Test exception handling in calling code
- Verify cleanup happens on errors

**Verify Type Checking:**
- Test type validation catches invalid types
- Test type conversions work correctly
- Verify type hints match runtime behavior

**Improving Coverage:**
- Identify uncovered lines in coverage report
- Write tests specifically for uncovered code
- Refactor untestable code to be more testable
- Remove dead code that can't be covered

**Coverage Report Interpretation:**
- Green lines are covered by tests
- Red lines are not covered
- Yellow lines are partially covered (branches)
- Focus on red and yellow lines first

### Documentation Standards

#### Building Documentation

**Setup Process:**

**Install Documentation Tools:**
- Install package with documentation extra dependencies
- This includes Sphinx, theme, and extensions
- Required for building documentation locally

**Build HTML Documentation:**
- Navigate to docs subdirectory
- Run Sphinx make command for HTML output
- Generated HTML appears in build directory
- Open index.html in web browser to view

**Live Rebuild During Development:**
- Use sphinx-autobuild for live preview
- Documentation rebuilds automatically on file changes
- Useful during documentation writing

**Clean Build:**
- Clean build directory to remove old artifacts
- Useful when reorganizing documentation structure
- Ensures fresh build without stale files

#### Documentation Structure

**API Reference:**
- Automatically generated from docstrings
- Uses Sphinx autodoc and autosummary directives
- Organized by module and functionality
- Includes all public APIs

**Tutorials and Guides:**
- Manually written instructional content
- Step-by-step learning materials
- Conceptual explanations
- Best practice recommendations

**Example Notebooks:**
- Executable Jupyter notebooks
- Demonstrate real-world usage
- Include visualizations and outputs
- Must be runnable without errors

**Architecture Documentation:**
- System design explanations
- Component interaction diagrams
- Performance characteristics
- Design decision rationale

#### Documentation Requirements

**API Reference Standards:**

**Auto-Generation Setup:**
- Sphinx autodoc extracts docstrings automatically
- Configure autodoc in Sphinx conf.py
- Use autosummary for module overviews
- Include type hints in API documentation

**Coverage:**
- All public modules must be documented
- All public classes and methods included
- All public functions documented
- Private APIs excluded unless exceptionally important

**Consistency:**
- Follow Google docstring format strictly
- Use consistent terminology throughout
- Cross-reference related functions
- Link to tutorials for complex topics

**Tutorial Writing:**

**Structure Requirements:**
- Clear learning objectives at start
- Prerequisite knowledge listed
- Step-by-step instructions
- Expected outputs shown
- Common pitfalls highlighted

**Content Guidelines:**
- Start with simplest case
- Build complexity gradually
- Explain "why" not just "how"
- Link to relevant API documentation
- Include troubleshooting section

**Format Options:**
- Markdown for text-heavy tutorials
- reStructuredText for advanced formatting
- Jupyter notebooks for interactive tutorials
- Include both prose and code

**Example Requirements:**

**Executable:**
- Examples must run without modification
- Include all necessary imports
- Show data preparation steps
- Handle dataset downloads

**Realistic:**
- Use real-world dataset sizes
- Show practical hyperparameters
- Include realistic training times
- Demonstrate actual use cases

**Documented:**
- Document hardware requirements clearly
- Show expected resource usage
- Include timing benchmarks
- Note GPU memory requirements

**Complete:**
- Show full pipeline from data to results
- Include evaluation and metrics
- Demonstrate checkpoint saving/loading
- Show how to use trained models

**Architecture Documentation:**

**System Diagrams:**
- Create diagrams for complex architectures
- Use Mermaid or PlantUML syntax
- Show component relationships
- Illustrate data flow

**Design Explanations:**
- Explain architectural decisions
- Document trade-offs considered
- Justify technology choices
- Note future improvement areas

**Performance Documentation:**
- Document performance characteristics
- Show scaling behavior with data size
- Note computational complexity
- Include memory usage patterns

**Integration Documentation:**
- Explain how components interact
- Document API contracts between modules
- Show sequence diagrams for complex flows
- Note dependencies and requirements

---

## 🛠️ Implementation Guides

### Adding New Algorithms

**Overview:**
RL algorithms are the core of ThinkRL. Adding new algorithms requires careful implementation of loss computation, advantage calculation, and optimization steps.

**File Location:**
- Create new file in thinkrl/algorithms/ directory
- Name file descriptively (e.g., vapo.py, grpo.py)
- Follow existing naming conventions

**Required Components:**

**1. Algorithm Class:**
- Main class containing algorithm logic
- Initialize with model, reference model, and hyperparameters
- Store all configuration as class attributes
- Implement standard training interface

**2. Constructor:**
- Accept model and reference model as parameters
- Accept all algorithm-specific hyperparameters
- Initialize optimizer for model parameters
- Store learning rate and other optimization settings
- Set up any additional components (KL controller, schedulers)

**3. Loss Computation Method:**
- Take batch dictionary as input (input_ids, attention_mask, labels, etc.)
- Perform forward pass through policy model
- Compute advantages using GAE or other method
- Calculate policy loss (typically with clipping or importance sampling)
- Compute KL divergence penalty from reference model
- Optionally compute value function loss if using critic
- Return dictionary with total loss and component losses

**4. Training Step Method:**
- Take batch as input
- Zero gradients in optimizer
- Call loss computation method
- Perform backward pass on total loss
- Clip gradients to prevent exploding gradients
- Update parameters via optimizer step
- Return metrics dictionary for logging

**5. Utility Methods:**
- Implement advantage computation (GAE, TD-lambda)
- Implement policy evaluation if needed
- Add methods for learning rate scheduling
- Include any algorithm-specific computations

**Implementation Checklist:**
- [ ] Create algorithm class with proper inheritance
- [ ] Implement constructor with all hyperparameters
- [ ] Write loss computation method
- [ ] Implement training step with gradient clipping
- [ ] Add advantage calculation method
- [ ] Include KL divergence computation
- [ ] Write comprehensive docstrings
- [ ] Add type hints to all methods
- [ ] Handle edge cases (empty batches, NaN losses)
- [ ] Support both CPU and GPU execution

**Registration:**- Add your algorithm to the ALGORITHM_REGISTRY in thinkrl/registry/algorithms.py
- Register with a descriptive name string
- This enables loading via configuration files
- Allows easy swapping between algorithms

**Testing:**
- Create test file in tests/test_algorithms/ directory
- Write unit tests for loss computation
- Test gradient flow through entire pipeline
- Verify convergence on simple tasks
- Test both CPU and GPU execution
- Include numerical stability tests
- Test edge cases (empty batches, single samples)
- Verify metric logging works correctly

**Configuration:**
- Create YAML config file in configs/algos/ directory
- Include all hyperparameters with sensible defaults
- Document each hyperparameter's purpose
- Provide example configurations for different scales

**Documentation:**
- Add algorithm to documentation in docs/algorithms/
- Explain algorithm theory and motivation
- Document hyperparameter effects
- Include performance benchmarks
- Reference original papers
- Show example usage

**Best Practices:**
- Study existing algorithm implementations first
- Follow established patterns for consistency
- Use helper functions from utils module
- Log all relevant metrics during training
- Handle numerical instability (NaN, Inf)
- Support gradient accumulation if needed
- Make code modular and reusable

### Adding New Models

**Overview:**
Model architectures define the neural networks used in training. Adding new models requires proper integration with HuggingFace and PyTorch.

**File Location:**
- Create new file in thinkrl/models/ directory
- Name after model architecture (e.g., gpt.py, llama.py)
- Keep related models in same file

**Required Components:**

**1. Model Configuration Class:**
- Inherit from PretrainedConfig
- Define model_type string attribute
- Include all architectural hyperparameters
- Set sensible default values
- Support serialization to/from JSON

**Configuration Parameters:**
- Vocabulary size
- Hidden dimension size
- Number of layers
- Number of attention heads
- Intermediate size for feedforward
- Activation function type
- Dropout rates
- Maximum sequence length
- Any architecture-specific parameters

**2. Model Class:**
- Inherit from PreTrainedModel
- Set config_class attribute
- Implement proper initialization
- Define forward pass logic
- Support different output modes

**Model Structure:**
- Token embedding layer
- Position embedding layer (if needed)
- Transformer encoder/decoder layers
- Layer normalization
- Language modeling head (for generation)
- Value head (for RL algorithms)
- Optional: multiple task-specific heads

**Forward Pass:**
- Accept input_ids and attention_mask
- Optionally accept labels for training
- Compute embeddings and apply transformations
- Pass through transformer layers
- Generate logits from language modeling head
- Optionally compute values from value head
- Return outputs in standard format

**Output Format:**
- Return dictionary or dataclass with structured outputs
- Include logits for next token prediction
- Include values for RL (if applicable)
- Include hidden states if requested
- Include attention weights if requested
- Support both training and inference modes

**3. Integration Requirements:**

**HuggingFace Compatibility:**
- Support standard HuggingFace interfaces
- Implement from_pretrained class method
- Implement save_pretrained method
- Register configuration properly
- Support auto loading mechanisms

**Device Handling:**
- Support moving to CUDA devices
- Handle device placement automatically
- Support device_map for large models
- Work with Accelerate for distributed training

**Memory Optimization:**
- Support gradient checkpointing
- Implement efficient attention mechanisms
- Support Flash Attention if applicable
- Allow mixed precision training

**Implementation Checklist:**
- [ ] Create config class with all parameters
- [ ] Implement model class with proper inheritance
- [ ] Write forward method with clear logic
- [ ] Add value head for RL if needed
- [ ] Support gradient checkpointing
- [ ] Implement proper initialization
- [ ] Handle device placement correctly
- [ ] Write comprehensive docstrings
- [ ] Add type hints throughout
- [ ] Test on various input shapes

**Testing:**
- Create test file in tests/test_models/
- Test model initialization
- Test forward pass with various inputs
- Test gradient computation
- Test saving and loading
- Test device movement (CPU ↔ GPU)
- Test with different batch sizes
- Test with different sequence lengths
- Verify output shapes are correct
- Test integration with training loop

**Configuration:**
- Create model config in configs/models/
- Include all architectural parameters
- Provide configs for different model sizes
- Document parameter effects on performance

**Documentation:**
- Document model architecture clearly
- Explain design decisions
- Show parameter counts for different configs
- Include memory requirements
- Reference original architecture papers
- Provide usage examples

### Adding Utilities

**Overview:**
Utility functions provide reusable functionality across the codebase. They should be general-purpose, well-tested, and documented.

**File Location:**
- Add to existing files in thinkrl/utils/ if related
- Create new file if functionality is distinct
- Keep utilities focused and cohesive

**GPU-Aware Implementation Pattern:**

When implementing utilities that process tensors, follow this pattern for optimal performance:

**1. Check Environment:**
- Verify if CuPy is available
- Check if input tensor is on GPU
- Determine which code path to use

**2. GPU Path (if available):**
- Convert PyTorch tensor to CuPy array via DLPack
- Perform operations using CuPy functions
- Convert result back to Python types or PyTorch tensors
- This avoids CPU transfers and is much faster

**3. CPU Fallback Path:**
- Convert tensor to NumPy array
- Perform operations using NumPy or SciPy
- Convert result back to appropriate format
- Always provide this fallback for compatibility

**4. Consistent Returns:**
- Return results in same format regardless of path
- Document return types clearly
- Ensure numerical consistency between paths

**Implementation Considerations:**

**Error Handling:**
- Handle ImportError if CuPy not installed
- Handle OSError if CUDA libraries missing
- Handle device mismatch errors
- Provide informative error messages

**Testing:**
- Test both GPU and CPU paths
- Verify numerical consistency
- Test with various input shapes
- Test edge cases (empty, single element)
- Test with different data types

**Performance:**
- Avoid unnecessary data transfers
- Batch operations when possible
- Use vectorized operations
- Profile both code paths

**Documentation:**
- Document GPU acceleration behavior
- Note performance differences
- Explain when to use GPU vs CPU
- Include complexity analysis

**Common Utility Categories:**

**Logging Utilities:**
- Experiment tracking setup
- Metric logging functions
- Progress bar integration
- Log formatting utilities

**Checkpoint Utilities:**
- Model saving functions
- Checkpoint loading functions
- State dictionary handling
- Checkpoint cleanup and management

**Metric Utilities:**
- Statistical computation functions
- Reward processing functions
- Advantage calculation helpers
- KL divergence computations

**Data Utilities:**
- Tokenization helpers
- Batch collation functions
- Data preprocessing utilities
- Dataset format converters

**Best Practices:**
- Keep functions focused and single-purpose
- Use descriptive function names
- Provide sensible default parameters
- Handle edge cases gracefully
- Write defensive code with validation
- Log important operations
- Cache expensive computations when appropriate

### Integrating with vLLM

**Overview:**
vLLM integration enables high-throughput inference for online RL and RLAIF. Integration requires careful coordination between training and inference processes.

**When to Use vLLM:**
- Online RL algorithms requiring rollout generation
- RLAIF pipelines with human feedback simulation
- High-throughput inference during training
- When generation speed is critical
- Multi-turn conversation generation

**Setup Process:**

**1. Start vLLM Server:**
- Launch vLLM server on dedicated GPU(s)
- Specify model path or HuggingFace model ID
- Configure tensor parallelism based on GPUs available
- Set appropriate port for HTTP API
- Configure generation parameters

**Server Configuration:**
- Model name or path
- Tensor parallel size (number of GPUs for inference)
- Port number for API server
- Trust remote code setting
- Quantization options if needed
- GPU memory utilization setting

**2. Initialize Client in Training Code:**
- Import VLLMClient from integration module
- Create client with server URL and NCCL port
- Initialize on rank 0 process only
- Configure NCCL bridge for weight synchronization

**Client Configuration:**
- Server URL (e.g., http://localhost:8000)
- NCCL bridge port (must not conflict with training)
- Device for weight synchronization
- Timeout settings for generation requests

**3. Weight Synchronization:**
- Initialize weight sync at training start
- Specify device (typically cuda:0)
- Handle FSDP or DDP model contexts
- Ensure model architectures match exactly

**Weight Sync Considerations:**
- Only rank 0 communicates with vLLM
- Model must have identical architecture
- Tokenizer must be compatible
- Handle model partitioning properly

**Integration in Training Loop:**

**Generation Phase:**
- Sample prompts from training dataset
- Prepare prompts in correct format
- Send generation request to vLLM server
- Wait for completed responses
- Parse and validate responses

**Generation Parameters:**
- Maximum tokens to generate
- Temperature for sampling
- Top-p for nucleus sampling
- Top-k for top-k sampling
- Repetition penalty
- Stop sequences

**Training Phase:**
- Process generated responses
- Compute rewards or feedback signals
- Prepare training batches
- Run training steps
- Update model parameters

**Weight Update Phase:**
- Broadcast updated weights from trainer to vLLM
- Ensure synchronization completes
- Verify vLLM has latest weights
- Continue to next generation cycle

**Critical Implementation Notes:**

**Process Isolation:**
- vLLM runs as separate process from training
- Communication via HTTP API and NCCL
- Training and inference on different GPUs preferred
- Manage GPU memory carefully on shared devices

**Rank Management:**
- Only rank 0 interacts with vLLM client
- Other ranks skip vLLM operations
- Broadcast generated data to all ranks if needed
- Synchronize properly in distributed setting

**Error Handling:**
- Handle connection errors to vLLM server
- Retry failed generation requests
- Validate generated text format
- Handle timeout errors gracefully
- Monitor GPU memory on inference nodes

**Performance Optimization:**
- Batch generation requests when possible
- Use asynchronous generation if supported
- Pipeline generation and training
- Monitor latency and throughput
- Adjust batch sizes based on GPU memory

**Testing vLLM Integration:**
- Test with small models first
- Verify weight synchronization works
- Check generation quality
- Monitor GPU memory usage
- Test error recovery
- Benchmark throughput
- Test with different batch sizes

**Troubleshooting:**
- Verify NCCL installation and version
- Check port availability before starting
- Ensure model architecture matches exactly
- Verify tokenizer compatibility
- Monitor GPU memory on both sides
- Check NCCL bridge initialization
- Validate HTTP API connectivity

**Best Practices:**
- Start with single-GPU setup before scaling
- Test weight sync with dummy data
- Monitor generation quality throughout training
- Log vLLM performance metrics
- Handle vLLM server restarts gracefully
- Cache generation results when appropriate
- Validate generated text before using

---

## ⚡ Performance Guidelines

### GPU Memory Management

**Optimization Strategies:**

**Gradient Checkpointing:**
- Apply to transformer layers in large models
- Trades compute for memory
- Enable via model configuration
- Essential for models over 1B parameters

**Memory Cleanup:**
- Clear CUDA cache periodically during training
- Especially after evaluation or large batch operations
- Clear every 50-100 training steps
- Monitor memory usage with nvidia-smi

**Data Type Selection:**
- Use bfloat16 on Ampere or newer GPUs
- Use float16 on older GPU architectures
- Keep some operations in float32 for stability
- Use automatic mixed precision training

**Batch Size Tuning:**
- Start with small batch size
- Increase gradually while monitoring memory
- Use gradient accumulation for effective larger batches
- Balance batch size with throughput

**Model Parallelism:**
- Use for models that don't fit in single GPU
- Consider tensor parallelism for large layers
- Consider pipeline parallelism for deep models
- Use FSDP for parameter sharding

### Efficient Data Loading

**DataLoader Configuration:**

**Num Workers:**
- Set to 4-8 for good CPU utilization
- More workers for I/O-bound datasets
- Fewer workers if preprocessing is CPU-intensive
- Monitor CPU usage to tune

**Pin Memory:**
- Enable for faster CPU-to-GPU transfers
- Allocates page-locked memory
- Small CPU memory overhead
- Significant speedup for large batches

**Prefetch Factor:**
- Prefetch 2-4 batches ahead
- Overlaps data loading with GPU computation
- Reduces GPU idle time
- Tune based on batch processing time

**Dataset Optimization:**
- Use memory-mapped files for large datasets
- Cache preprocessed data when possible
- Use streaming for datasets too large for memory
- Shard datasets in distributed training

**Batching Strategies:**
- Group similar-length sequences
- Reduces padding overhead
- Improves training efficiency
- Use dynamic batching when possible

### Profiling and Debugging

**PyTorch Profiler:**

**Basic Profiling:**
- Profile CPU and CUDA activities
- Identify performance bottlenecks
- Analyze kernel execution times
- Find memory allocation patterns

**Profiling Workflow:**
- Wrap code section to profile
- Run for limited iterations to avoid overhead
- Generate trace or table output
- Analyze results to identify bottlenecks

**What to Look For:**
- Long-running operations
- GPU idle time
- CPU-GPU transfer overhead
- Memory allocation patterns
- Kernel launch overhead

**Advanced Profiling:**
- Use Chrome trace viewer for visualization
- Analyze tensor operations in detail
- Profile with different batch sizes
- Compare different implementations

**Performance Optimization Process:**

**1. Measure Baseline:**
- Profile existing implementation
- Record key metrics (throughput, latency, memory)
- Identify top bottlenecks

**2. Optimize Hotspots:**
- Focus on operations taking most time
- Try different implementations
- Enable compiler optimizations
- Use fused operations when possible

**3. Verify Improvement:**
- Profile after changes
- Compare against baseline
- Ensure correctness maintained
- Document improvements

**4. Iterate:**
- Move to next bottleneck
- Continue until goals met
- Balance optimization effort with gains

### Training Speed Optimization

**Mixed Precision Training:**
- Use automatic mixed precision (AMP)
- Reduces memory usage by ~50%
- Increases speed by 2-3x on modern GPUs
- Requires careful loss scaling

**torch.compile():**
- Enable JIT compilation with PyTorch 2.0+
- Significantly faster inference
- Some speedup for training
- Test compatibility with your models

**Efficient Operations:**
- Use fused optimizers (FusedAdam)
- Use efficient attention implementations
- Minimize tensor copying
- Batch operations when possible

**Distributed Training:**
- Use DistributedDataParallel for multi-GPU
- Enable NCCL backend for best performance
- Use gradient compression if network-limited
- Overlap communication with computation

**Checkpoint Strategy:**
- Save checkpoints asynchronously
- Use SafeTensors format
- Checkpoint less frequently
- Keep only recent checkpoints

---

## 🚀 Submitting Changes

### Pull Request Process

**Step 1: Prepare Your Changes**
- Ensure all tests pass locally
- Run code quality tools (ruff, mypy, bandit)
- Update relevant documentation
- Add or update tests for new functionality
- Update CHANGELOG.md with your changes

**Step 2: Create Pull Request**
- Push feature branch to your fork
- Create PR against dev branch (not main)
- Use descriptive PR title
- Fill out PR template completely
- Link related issues with keywords (Fixes #123, Closes #456)

**PR Title Format:**
- Use conventional commit format
- Examples: "feat: Add VAPO algorithm implementation"
- Examples: "fix: Resolve memory leak in metrics computation"
- Examples: "docs: Update API reference for vLLM integration"

**Step 3: PR Description**
- Explain what changes were made and why
- Describe testing performed
- Note any breaking changes
- Include relevant benchmarks or measurements
- Add screenshots for UI changes

**Step 4: Code Review**
- Respond to reviewer feedback promptly
- Push additional commits to same branch
- Explain your reasoning when appropriate
- Be open to suggestions and improvements
- Request re-review after addressing comments

**Step 5: Final Checks**
- Ensure CI pipeline passes completely
- Verify documentation builds successfully
- Check test coverage meets threshold
- Confirm no merge conflicts with dev branch

**Step 6: Merge**
- Maintainer will merge after approval
- PR is squash-merged or merge-committed
- Your branch can be deleted after merge
- Changes will be included in next release

### Pull Request Checklist

Before submitting your PR, verify:

**Code Quality:**
- [ ] Ruff check passes without errors
- [ ] Ruff format applied to all files
- [ ] Mypy type checking passes
- [ ] Bandit security checks pass
- [ ] No debug print statements or commented code
- [ ] Code follows established patterns

**Testing:**
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Test coverage meets 58% minimum
- [ ] GPU tests pass if applicable
- [ ] Integration tests pass if applicable

**Documentation:**
- [ ] Docstrings added/updated (Google style)
- [ ] API documentation updated
- [ ] README updated if needed
- [ ] CHANGELOG.md updated with changes
- [ ] Examples updated if API changed

**Git Hygiene:**
- [ ] Commits follow conventional commit format
- [ ] Commit messages are clear and descriptive
- [ ] No merge commits in feature branch
- [ ] Branch is up to date with dev

**CI/CD:**
- [ ] Pre-commit hooks pass
- [ ] GitHub Actions workflow passes
- [ ] Documentation builds successfully
- [ ] No test failures in CI

### Commit Message Guidelines

**Conventional Commits Format:**

Use conventional commit format for all commits:

**Type:**
- **feat**: New feature for users
- **fix**: Bug fix for users
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring (no feature change)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system changes
- **ci**: CI/CD pipeline changes
- **chore**: Other changes (dependencies, config)

**Scope (optional):**
- Specify component affected
- Examples: algorithms, models, utils, docs
- Use when change is localized

**Description:**
- Use imperative mood ("Add" not "Added" or "Adds")
- Don't capitalize first letter
- No period at end
- Keep under 72 characters

**Body (optional):**
- Provide additional context
- Explain "why" not "what"
- Reference issues and tickets
- Include breaking change notes

**Examples:**
- feat(algorithms): add VAPO algorithm implementation
- fix(metrics): resolve GPU memory leak in advantage computation
- docs(api): update vLLM integration guide
- test(utils): add tests for checkpoint utilities
- refactor(models): simplify attention mechanism
- perf(data): optimize batch collation speed

### Review Process

**What Reviewers Look For:**

**Correctness:**
- Code works as intended
- Edge cases handled properly
- No obvious bugs or errors
- Tests verify behavior

**Quality:**
- Code is readable and maintainable
- Follows project style and conventions
- Appropriate abstractions used
- No unnecessary complexity

**Documentation:**
- Clear docstrings present
- API changes documented
- Examples provided when helpful
- Comments explain non-obvious code

**Testing:**
- Adequate test coverage
- Tests are meaningful
- Edge cases tested
- No flaky tests

**Performance:**
- No obvious performance regressions
- Efficient algorithms used
- Resource usage reasonable
- Benchmarks provided for optimizations

**How to Respond to Feedback:**

**Be Receptive:**
- Thank reviewers for their time
- Consider suggestions seriously
- Ask questions if unclear
- Don't take criticism personally

**Be Responsive:**
- Respond to comments promptly
- Make requested changes
- Explain if you disagree respectfully
- Mark conversations resolved when addressed

**Be Collaborative:**
- Discuss trade-offs openly
- Propose alternative solutions
- Learn from reviewer expertise
- Help improve the codebase together

---

## 🤝 Community & Support

### Communication Channels

**GitHub Issues:**
- **Purpose**: Bug reports, feature requests, task tracking
- **Use When**: You found a bug, want to propose a feature, or need to track work
- **Response Time**: Usually within 1-2 business days

**GitHub Discussions:**
- **Purpose**: Questions, ideas, general discussion
- **Use When**: You have questions, want feedback on ideas, or need help
- **Response Time**: Community-driven, varies

**Email:**
- **Contact**: archit@ellanorai.org
- **Use When**: Private concerns, security issues, collaboration proposals
- **Response Time**: Usually within 2-3 business days

### Reporting Bugs

**Before Reporting:**
- Search existing issues to avoid duplicates
- Verify bug exists in latest version
- Try to isolate minimum reproducible case
- Gather relevant system information

**Bug Report Contents:**

**Clear Description:**
- What you were trying to do
- What you expected to happen
- What actually happened
- Impact of the bug

**Reproduction Steps:**
- Numbered list of exact steps
- Include commands run
- Note any configuration used
- Specify data or models used

**Environment Information:**
- Operating system and version
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Relevant library versions
- GPU model (if applicable)

**Error Information:**
- Full error traceback
- Relevant log output
- Screenshots if applicable
- Any error codes or messages

**Additional Context:**
- When did this start happening?
- Does it happen consistently?
- Any recent changes to your setup?
- Attempted workarounds?

### Requesting Features

**Feature Request Contents:**

**Use Case:**
- Describe the problem you're trying to solve
- Explain why current functionality is insufficient
- Provide concrete examples of use cases
- Explain how this benefits other users

**Proposed Solution:**
- Describe your suggested approach
- Explain API design if applicable
- Consider backward compatibility
- Note implementation complexity if known

**Alternatives Considered:**
- What other approaches did you consider?
- Why is your proposed solution better?
- Any workarounds currently possible?
- Trade-offs of different approaches?

**Additional Context:**
- Related features or issues
- Examples from other libraries
- Academic papers if applicable
- Implementation resources available?

### Contributing to Discussions

**Best Practices:**
- Search before asking questions
- Provide context and examples
- Be specific about your problem
- Share what you've already tried
- Thank those who help you

**When Asking Questions:**
- Clear title describing the question
- Provide relevant code and configuration
- Specify your environment
- Explain what you're trying to achieve
- Note any error messages

**When Answering Questions:**
- Be patient and respectful
- Provide clear explanations
- Include examples when helpful
- Link to relevant documentation
- Follow up to ensure resolution

---

## 📚 Learning Resources

### RLHF Fundamentals

**Key Papers:**
- "Learning to Summarize from Human Feedback" (OpenAI, 2020) - arXiv:2009.01325
- "Training Language Models to Follow Instructions" (InstructGPT) - arXiv:2203.02155
- "Constitutional AI" (Anthropic, 2022) - arXiv:2212.08073
- "Direct Preference Optimization" - arXiv:2305.18290
- "RLAIF: Scaling Reinforcement Learning from Human Feedback" - arXiv:2309.00267

**Algorithm-Specific Papers:**
- VAPO: [Link to be added when available]
- DAPO: [Link to be added when available]
- GRPO: [Link to be added when available]

### Technical Documentation

**PyTorch:**
- Official PyTorch tutorials and documentation
- PyTorch Performance Tuning Guide
- Distributed Training documentation
- PyTorch 2.0 Compiler guide

**CuPy:**
- CuPy User Guide
- CuPy-NumPy API compatibility reference
- CuPy performance optimization guide
- cupyx.scipy documentation

**vLLM:**
- vLLM documentation and quickstart
- vLLM architecture overview
- vLLM performance benchmarks
- NCCL collective operations guide

**HuggingFace:**
- Transformers documentation
- Model Hub documentation
- PEFT library guide
- Accelerate for distributed training

### Development Resources

**Python Best Practices:**
- Google Python Style Guide
- PEP 8 - Style Guide for Python Code
- Type Hints (PEP 484, 585, 586)
- Python documentation best practices

**Testing:**
- Pytest documentation
- pytest-cov for coverage
- Testing best practices guide
- Test-driven development resources

**Git and GitHub:**
- Git branching strategies
- Conventional Commits specification
- GitHub Flow documentation
- Code review best practices

### Machine Learning

**Deep Learning Fundamentals:**
- Neural networks and backpropagation
- Optimization algorithms (Adam, SGD)
- Regularization techniques
- Transformer architecture

**Reinforcement Learning:**
- Policy Gradient methods
- Actor-Critic algorithms
- Proximal Policy Optimization (PPO)
- Advantage estimation

**Large Language Models:**
- Attention mechanisms
- Positional encodings
- Layer normalization
- Token embeddings

---

## 🙏 Acknowledgments

Thank you for contributing to ThinkRL! Your efforts help democratize access to state-of-the-art RLHF technology and advance the field of AI alignment.

**Special Thanks:**
- All contributors who have submitted code, documentation, and feedback
- The open-source community for building the amazing tools we depend on
- Researchers advancing the field of RLHF and AI safety

**Questions?**
If you have any questions about contributing, please open a GitHub Discussion or contact us at archit@ellanorai.org

---

**Happy Contributing! 🎉**