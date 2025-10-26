"""
ThinkRL: Universal RLHF Training Library
========================================

A powerful, open-source library for Reinforcement Learning from Human Feedback (RLHF)
with state-of-the-art algorithms, reasoning capabilities, and multimodal support.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return __doc__

# Read version from __init__.py
def read_version():
    """Read version from thinkrl/__init__.py."""
    try:
        with open("thinkrl/__init__.py", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "0.1.0"

# Core dependencies - minimal, stable, no GPU requirements
CORE_REQUIREMENTS = [
    "torch>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "pyyaml>=6.0,<7.0",
    "tqdm>=4.65.0",
    "accelerate>=0.21.0,<1.0.0",
]

# Algorithm-specific dependencies
ALGORITHM_REQUIREMENTS = {
    # RLHF Algorithms
    "algorithms": [
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    
    # GPU acceleration (optional but recommended)
    "cuda": [
        "cupy-cuda12x>=12.0.0,<13.0.0",
    ],
    
    # CUDA 11.x support
    "cuda11": [
        "cupy-cuda11x>=11.0.0,<12.0.0",
    ],
}

# ML Framework integrations
FRAMEWORK_REQUIREMENTS = {
    # HuggingFace ecosystem
    "transformers": [
        "transformers>=4.30.0,<5.0.0",
        "tokenizers>=0.15.0,<1.0.0",
        "datasets>=2.14.0,<3.0.0",
        "safetensors>=0.3.0",
    ],
    
    # Parameter-efficient fine-tuning
    "peft": [
        "peft>=0.4.0,<1.0.0",
        "bitsandbytes>=0.41.0",
    ],
    
    # Distributed training
    "deepspeed": [
        "deepspeed>=0.9.0,<1.0.0",
    ],
    
    # Alternative distributed training
    "fsdp": [
        "torch>=2.0.0",
        "accelerate>=0.21.0",
    ],
}

# Modality-specific requirements
MODALITY_REQUIREMENTS = {
    # Vision-language models
    "vision": [
        "pillow>=9.0.0,<11.0.0",
        "torchvision>=0.15.0,<1.0.0",
        "opencv-python>=4.5.0",
    ],
    
    # Audio processing
    "audio": [
        "torchaudio>=2.0.0,<3.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
    ],
    
    # Multimodal (vision + audio)
    "multimodal": [
        "transformers>=4.30.0,<5.0.0",
        "pillow>=9.0.0,<11.0.0",
        "torchvision>=0.15.0,<1.0.0",
        "torchaudio>=2.0.0,<3.0.0",
        "opencv-python>=4.5.0",
    ],
}

# Reasoning capabilities
REASONING_REQUIREMENTS = {
    # Chain-of-Thought and Tree-of-Thought
    "reasoning": [
        "networkx>=3.1,<4.0",
        "graphviz>=0.20.0",
        "matplotlib>=3.5.0",
    ],
    
    # Mathematical reasoning
    "math": [
        "sympy>=1.12.0",
        "scipy>=1.10.0",
    ],
    
    # Code reasoning
    "code": [
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
    ],
}

# Experiment tracking and logging
LOGGING_REQUIREMENTS = {
    # Weights & Biases
    "wandb": [
        "wandb>=0.15.0,<1.0.0",
    ],
    
    # TensorBoard
    "tensorboard": [
        "tensorboard>=2.13.0",
        "tensorboardX>=2.6.0",
    ],
    
    # MLflow
    "mlflow": [
        "mlflow>=2.5.0,<3.0.0",
    ],
}

# Development and testing
DEV_REQUIREMENTS = {
    # Core development tools
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-cov>=4.1.0,<5.0.0",
        "pytest-xdist>=3.3.0",
        "pytest-mock>=3.11.0",
    ],
    
    # Code formatting and linting
    "format": [
        "black>=23.9.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<7.0.0",
        "mypy>=1.5.0,<2.0.0",
    ],
    
    # Security and quality
    "quality": [
        "bandit>=1.7.0,<2.0.0",
        "safety>=2.3.0,<3.0.0",
        "pre-commit>=3.0.0,<4.0.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        "nbsphinx>=0.9.0",
    ],
}

# Benchmark and evaluation
EVAL_REQUIREMENTS = {
    # Evaluation frameworks
    "eval": [
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "bleurt>=0.0.2",
    ],
    
    # Benchmarks
    "benchmarks": [
        "lm-eval>=0.4.0",
        "human-eval>=1.0.0",
    ],
}

# Convenience bundles
CONVENIENCE_BUNDLES = {
    # State-of-the-art algorithms with GPU support
    "sota": [
        *ALGORITHM_REQUIREMENTS["algorithms"],
        *ALGORITHM_REQUIREMENTS["cuda"],
        *FRAMEWORK_REQUIREMENTS["transformers"],
        *FRAMEWORK_REQUIREMENTS["peft"],
        *REASONING_REQUIREMENTS["reasoning"],
    ],
    
    # Distributed training setup
    "distributed": [
        *FRAMEWORK_REQUIREMENTS["deepspeed"],
        *FRAMEWORK_REQUIREMENTS["fsdp"],
        *ALGORITHM_REQUIREMENTS["cuda"],
    ],
    
    # Research setup
    "research": [
        *FRAMEWORK_REQUIREMENTS["transformers"],
        *FRAMEWORK_REQUIREMENTS["peft"],
        *REASONING_REQUIREMENTS["reasoning"],
        *LOGGING_REQUIREMENTS["wandb"],
        *EVAL_REQUIREMENTS["eval"],
    ],
    
    # Production setup
    "production": [
        *ALGORITHM_REQUIREMENTS["cuda"],
        *FRAMEWORK_REQUIREMENTS["transformers"],
        *FRAMEWORK_REQUIREMENTS["deepspeed"],
        *LOGGING_REQUIREMENTS["tensorboard"],
    ],
    
    # Complete development environment
    "complete": [
        *ALGORITHM_REQUIREMENTS["cuda"],
        *FRAMEWORK_REQUIREMENTS["transformers"],
        *FRAMEWORK_REQUIREMENTS["peft"],
        *FRAMEWORK_REQUIREMENTS["deepspeed"],
        *MODALITY_REQUIREMENTS["multimodal"],
        *REASONING_REQUIREMENTS["reasoning"],
        *REASONING_REQUIREMENTS["math"],
        *LOGGING_REQUIREMENTS["wandb"],
        *LOGGING_REQUIREMENTS["tensorboard"],
        *DEV_REQUIREMENTS["dev"],
        *DEV_REQUIREMENTS["format"],
        *DEV_REQUIREMENTS["quality"],
        *EVAL_REQUIREMENTS["eval"],
    ],
    
    # Everything (for CI/testing)
    "all": [
        *ALGORITHM_REQUIREMENTS["cuda"],
        *FRAMEWORK_REQUIREMENTS["transformers"],
        *FRAMEWORK_REQUIREMENTS["peft"],
        *FRAMEWORK_REQUIREMENTS["deepspeed"],
        *MODALITY_REQUIREMENTS["multimodal"],
        *REASONING_REQUIREMENTS["reasoning"],
        *REASONING_REQUIREMENTS["math"],
        *REASONING_REQUIREMENTS["code"],
        *LOGGING_REQUIREMENTS["wandb"],
        *LOGGING_REQUIREMENTS["tensorboard"],
        *LOGGING_REQUIREMENTS["mlflow"],
        *DEV_REQUIREMENTS["dev"],
        *DEV_REQUIREMENTS["format"],
        *DEV_REQUIREMENTS["quality"],
        *DEV_REQUIREMENTS["docs"],
        *EVAL_REQUIREMENTS["eval"],
        *EVAL_REQUIREMENTS["benchmarks"],
    ],
}

# Combine all extras
EXTRAS_REQUIRE = {
    **ALGORITHM_REQUIREMENTS,
    **FRAMEWORK_REQUIREMENTS,
    **MODALITY_REQUIREMENTS,
    **REASONING_REQUIREMENTS,
    **LOGGING_REQUIREMENTS,
    **DEV_REQUIREMENTS,
    **EVAL_REQUIREMENTS,
    **CONVENIENCE_BUNDLES,
}

# Console scripts for CLI tools
CONSOLE_SCRIPTS = {
    # Core training commands
    "thinkrl": "thinkrl.scripts.train:main",
    "thinkrl-train": "thinkrl.scripts.train:main",
    
    # Evaluation commands
    "thinkrl-eval": "thinkrl.scripts.evaluate:main",
    "thinkrl-evaluate": "thinkrl.scripts.evaluate:main",
    
    # Reasoning commands
    "thinkrl-cot": "thinkrl.scripts.chain_of_thought:main",
    "thinkrl-tot": "thinkrl.scripts.tree_of_thought:main",
    "thinkrl-reason": "thinkrl.scripts.reasoning:main",
    
    # Specialized training
    "thinkrl-multimodal": "thinkrl.scripts.multimodal_train:main",
    "thinkrl-distributed": "thinkrl.scripts.distributed_train:main",
    
    # Utilities
    "thinkrl-convert": "thinkrl.scripts.convert:main",
    "thinkrl-benchmark": "thinkrl.scripts.benchmark:main",
}

# Project classifiers
CLASSIFIERS = [
    # Development status
    "Development Status :: 4 - Beta",
    
    # Intended audience
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    
    # License
    "License :: OSI Approved :: Apache Software License",
    
    # OS support
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    
    # Python versions

    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    
    # Topics
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Text Processing :: Linguistic",
    
    # Framework
    "Framework :: Pytest",
    
    # Natural language
    "Natural Language :: English",
    
    # Environment
    "Environment :: Console",
    "Environment :: GPU :: NVIDIA CUDA",
]

# Keywords for PyPI search
KEYWORDS = [
    "machine-learning",
    "deep-learning",
    "reinforcement-learning",
    "rlhf",
    "human-feedback",
    "transformers",
    "large-language-models",
    "llm",
    "nlp",
    "pytorch",
    "artificial-intelligence",
    "ppo",
    "dapo",
    "grpo",
    "chain-of-thought",
    "tree-of-thought",
    "reasoning",
    "multimodal",
    "vision-language",
    "alignment",
    "fine-tuning",
    "distributed-training",
]

# Project URLs
PROJECT_URLS = {
    "Homepage": "https://github.com/Archit03/ThinkRL",
    "Documentation": "https://thinkrl.readthedocs.io/",
    "Repository": "https://github.com/Archit03/ThinkRL",
    "Bug Reports": "https://github.com/Archit03/ThinkRL/issues",
    "Feature Requests": "https://github.com/Archit03/ThinkRL/discussions",
    "Changelog": "https://github.com/Archit03/ThinkRL/blob/main/CHANGELOG.md",
    "Company": "https://ellanorai.org",
    "Examples": "https://github.com/Archit03/ThinkRL/tree/main/examples",
    "Tutorials": "https://github.com/Archit03/ThinkRL/tree/main/docs/tutorials",
}

# Main setup configuration
setup(
    # Basic package information
    name="thinkrl",
    version=read_version(),
    author="Archit Sood",
    author_email="archit@ellanorai.org",
    maintainer="EllanorAI Team",
    maintainer_email="team@ellanorai.org",
    
    # Package description
    description="Universal RLHF Training Library with state-of-the-art algorithms, reasoning, and multimodal support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # URLs and links
    url="https://github.com/Archit03/ThinkRL",
    project_urls=PROJECT_URLS,
    
    # Package discovery
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "docs",
            "docs.*",
            "examples",
            "examples.*",
            "scripts",
            "scripts.*",
        ]
    ),
    
    # Include additional files
    include_package_data=True,
    package_data={
        "thinkrl": [
            "configs/*.yaml",
            "configs/*.json",
            "data/*.json",
            "py.typed",
        ],
    },
    
    # Dependencies
    python_requires=">=3.8,<4.0",
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points
    entry_points={
        "console_scripts": [
            f"{name}={entry_point}"
            for name, entry_point in CONSOLE_SCRIPTS.items()
        ],
    },
    
    # Package metadata
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    license="Apache-2.0",
    
    # Package options
    zip_safe=False,
    platforms=["any"],
    
    # Additional metadata
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal due to platform-specific dependencies
        }
    },
)