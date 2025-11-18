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
        # Assuming thinkrl/__init__.py exists and contains __version__
        version_file = os.path.join("thinkrl", "__init__.py")
        if os.path.exists(version_file):
             with open(version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"').strip("'")
        else:
             # Fallback if thinkrl/__init__.py doesn't exist yet
             with open("tests/__init__.py", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    # Default version if not found
    return "0.1.0"


# Core dependencies - minimal, stable, no GPU requirements
CORE_REQUIREMENTS = [
    "torch>=2.0.0,<3.0.0",
    # REPLACED: "numpy>=1.24.0,<2.0.0",
    "cupy-cuda12x>=12.0.0,<13.0.0", 
    "scipy>=1.10.0", # Added from requirements.txt
    "pyyaml>=6.0,<7.0",
    "tqdm>=4.65.0",
    "accelerate>=0.21.0,<1.0.0",
]

# --- Dependencies based on requirements.txt ---

# GPU Acceleration
GPU_REQUIREMENTS = {
    "cuda": [
        "cupy-cuda12x>=12.0.0,<13.0.0", # From requirements.txt
    ],
    # Assuming cuda11 might still be relevant for some users
    "cuda11": [
        "cupy-cuda11x>=11.0.0,<12.0.0",
    ],
}

# HuggingFace Ecosystem & Core ML Framework Dependencies
FRAMEWORK_REQUIREMENTS = {
    "transformers": [
        "transformers>=4.30.0,<5.0.0",
        "tokenizers>=0.15.0,<1.0.0",
        "datasets>=2.14.0,<3.0.0",
        "safetensors>=0.3.0",
        "evaluate>=0.4.0", # Added from requirements.txt
        "huggingface-hub>=0.16.0,<1.0.0", # Added from requirements.txt
    ],
    "peft": [
        "peft>=0.4.0,<1.0.0",
        "bitsandbytes>=0.41.0", # From requirements.txt
    ],
    "deepspeed": [
        "deepspeed>=0.9.0,<1.0.0", # From requirements.txt
    ],
    "fsdp": [ # Kept from original, might be relevant
        "torch>=2.0.0",
        "accelerate>=0.21.0",
    ],
}

# Multimodal Support
MODALITY_REQUIREMENTS = {
    "vision": [
        "pillow>=9.0.0,<11.0.0", # From requirements.txt
        "torchvision>=0.15.0,<1.0.0", # From requirements.txt
        "opencv-python>=4.5.0", # From requirements.txt
    ],
    "audio": [
        "torchaudio>=2.0.0,<3.0.0", # From requirements.txt
        "librosa>=0.10.0", # From requirements.txt
        "soundfile>=0.12.0", # From requirements.txt
    ],
    "multimodal": [ # Combined vision and audio, plus transformers
        "transformers>=4.30.0,<5.0.0",
        "pillow>=9.0.0,<11.0.0",
        "torchvision>=0.15.0,<1.0.0",
        "torchaudio>=2.0.0,<3.0.0",
        "opencv-python>=4.5.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
    ],
}

# Reasoning Capabilities
REASONING_REQUIREMENTS = {
    "reasoning": [ # General reasoning (CoT/ToT support)
        "networkx>=3.1,<4.0", # From requirements.txt
        "graphviz>=0.20.0", # From requirements.txt
        "matplotlib>=3.5.0", # From requirements.txt
        "seaborn>=0.12.0", # Added from requirements.txt
    ],
    "math": [ # Mathematical reasoning
        "sympy>=1.12.0", # From requirements.txt
        "scipy>=1.10.0", # Already in core, but listed here for clarity
    ],
     # Code reasoning requirements from original setup.py
    "code": [
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
    ],
}

# Logging and Monitoring
LOGGING_REQUIREMENTS = {
    "wandb": [
        "wandb>=0.15.0,<1.0.0", # From requirements.txt
    ],
    "tensorboard": [
        "tensorboard>=2.13.0", # From requirements.txt
        "tensorboardX>=2.6.0", # From requirements.txt
    ],
    "mlflow": [ # Kept from original
        "mlflow>=2.5.0,<3.0.0",
    ],
}

# Data Processing and Utilities
UTILITIES_REQUIREMENTS = {
    "data_processing": [
        "pandas>=2.0.0", # From requirements.txt
        "pyarrow>=12.0.0", # From requirements.txt
        "xxhash>=3.3.0", # From requirements.txt
        "dill>=0.3.7", # From requirements.txt
        "multiprocess>=0.70.0", # From requirements.txt
    ],
    "general_utils": [
        "jsonlines>=3.1.0", # From requirements.txt
        "regex>=2023.0.0", # From requirements.txt
        "filelock>=3.12.0", # From requirements.txt
        "packaging>=23.0", # From requirements.txt
        "psutil>=5.9.0", # From requirements.txt
    ],
    "serialization": [
        "sentencepiece>=0.1.99", # From requirements.txt
        "protobuf>=3.20.0,<5.0.0", # From requirements.txt
    ],
     "api": [ # HTTP/API libs from requirements.txt
        "requests>=2.31.0",
        "aiohttp>=3.8.0",
        "urllib3>=2.0.0",
    ],
}

# Development and Testing
DEV_REQUIREMENTS = {
    "dev": [ # Core testing tools from requirements.txt
        "pytest>=7.0.0,<8.0.0",
        "pytest-cov>=4.1.0,<5.0.0",
        "pytest-xdist>=3.3.0",
        "pytest-mock>=3.11.0",
    ],
    "format": [ # Formatting/Linting tools from requirements.txt
        "black>=23.9.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<7.0.0",
        "mypy>=1.5.0,<2.0.0",
    ],
    "quality": [ # Quality/Security tools from requirements.txt
        "bandit>=1.7.0,<2.0.0",
        "safety>=2.3.0,<3.0.0",
        "pre-commit>=3.0.0,<4.0.0",
    ],
    "docs": [ # Kept from original, potentially relevant
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        "nbsphinx>=0.9.0",
    ],
}

# Benchmark and Evaluation
EVAL_REQUIREMENTS = {
    "eval": [ # Core evaluation tools
        "evaluate>=0.4.0", # Already in transformers extra
        "rouge-score>=0.1.2", # From requirements.txt
        "bert-score>=0.3.13", # From requirements.txt
        "nltk>=3.8.0", # Added from requirements.txt
        "scikit-learn>=1.3.0", # Added from requirements.txt
    ],
    "benchmarks": [ # Kept from original, potentially relevant
        "lm-eval>=0.4.0",
        "human-eval>=1.0.0",
    ],
}

# Inference and Serving (New for RLAIF/vLLM)
INFERENCE_REQUIREMENTS = {
    "inference": [
        "vllm>=0.2.0",  # High-throughput inference for RLAIF gen
        "ray",          # Distributed inference support
    ],
}


# Convenience bundles
CONVENIENCE_BUNDLES = {
    # State-of-the-art algorithms with GPU support
    # Includes inference for RLAIF generation
    "sota": list(set(
        # CORE_REQUIREMENTS + # Core should be installed by default
        GPU_REQUIREMENTS["cuda"] +
        FRAMEWORK_REQUIREMENTS["transformers"] +
        FRAMEWORK_REQUIREMENTS["peft"] +
        REASONING_REQUIREMENTS["reasoning"] +
        REASONING_REQUIREMENTS["math"] +
        INFERENCE_REQUIREMENTS["inference"]
    )),

    # Distributed training setup
    "distributed": list(set(
        # CORE_REQUIREMENTS +
        GPU_REQUIREMENTS["cuda"] +
        FRAMEWORK_REQUIREMENTS["deepspeed"] +
        FRAMEWORK_REQUIREMENTS["fsdp"]
    )),

    # Research setup
    "research": list(set(
        # CORE_REQUIREMENTS +
        GPU_REQUIREMENTS["cuda"] +
        FRAMEWORK_REQUIREMENTS["transformers"] +
        FRAMEWORK_REQUIREMENTS["peft"] +
        REASONING_REQUIREMENTS["reasoning"] +
        REASONING_REQUIREMENTS["math"] +
        LOGGING_REQUIREMENTS["wandb"] +
        EVAL_REQUIREMENTS["eval"] +
        UTILITIES_REQUIREMENTS["data_processing"]
    )),

    # Production setup (example, adjust as needed)
    "production": list(set(
        # CORE_REQUIREMENTS +
        GPU_REQUIREMENTS["cuda"] +
        FRAMEWORK_REQUIREMENTS["transformers"] +
        FRAMEWORK_REQUIREMENTS["deepspeed"] +
        LOGGING_REQUIREMENTS["tensorboard"] +
        UTILITIES_REQUIREMENTS["api"] +
        INFERENCE_REQUIREMENTS["inference"]
    )),

    # Complete development environment - includes almost everything from requirements.txt
    "complete": list(set(
        CORE_REQUIREMENTS +
        GPU_REQUIREMENTS["cuda"] +
        FRAMEWORK_REQUIREMENTS["transformers"] +
        FRAMEWORK_REQUIREMENTS["peft"] +
        FRAMEWORK_REQUIREMENTS["deepspeed"] +
        MODALITY_REQUIREMENTS["multimodal"] +
        REASONING_REQUIREMENTS["reasoning"] +
        REASONING_REQUIREMENTS["math"] +
        REASONING_REQUIREMENTS["code"] + 
        LOGGING_REQUIREMENTS["wandb"] +
        LOGGING_REQUIREMENTS["tensorboard"] +
        LOGGING_REQUIREMENTS["mlflow"] + 
        UTILITIES_REQUIREMENTS["data_processing"] +
        UTILITIES_REQUIREMENTS["general_utils"] +
        UTILITIES_REQUIREMENTS["serialization"] +
        UTILITIES_REQUIREMENTS["api"] +
        DEV_REQUIREMENTS["dev"] +
        DEV_REQUIREMENTS["format"] +
        DEV_REQUIREMENTS["quality"] +
        EVAL_REQUIREMENTS["eval"] +
        EVAL_REQUIREMENTS["benchmarks"] +
        INFERENCE_REQUIREMENTS["inference"] # Added vLLM support
    )),

    # 'all' extra - made identical to 'complete' but includes legacy/compat versions
    "all": list(set(
        CORE_REQUIREMENTS +
        GPU_REQUIREMENTS["cuda"] +
        GPU_REQUIREMENTS["cuda11"] + 
        FRAMEWORK_REQUIREMENTS["transformers"] +
        FRAMEWORK_REQUIREMENTS["peft"] +
        FRAMEWORK_REQUIREMENTS["deepspeed"] +
        FRAMEWORK_REQUIREMENTS["fsdp"] + 
        MODALITY_REQUIREMENTS["multimodal"] +
        REASONING_REQUIREMENTS["reasoning"] +
        REASONING_REQUIREMENTS["math"] +
        REASONING_REQUIREMENTS["code"] +
        LOGGING_REQUIREMENTS["wandb"] +
        LOGGING_REQUIREMENTS["tensorboard"] +
        LOGGING_REQUIREMENTS["mlflow"] +
        UTILITIES_REQUIREMENTS["data_processing"] +
        UTILITIES_REQUIREMENTS["general_utils"] +
        UTILITIES_REQUIREMENTS["serialization"] +
        UTILITIES_REQUIREMENTS["api"] +
        DEV_REQUIREMENTS["dev"] +
        DEV_REQUIREMENTS["format"] +
        DEV_REQUIREMENTS["quality"] +
        DEV_REQUIREMENTS["docs"] + 
        EVAL_REQUIREMENTS["eval"] +
        EVAL_REQUIREMENTS["benchmarks"] +
        INFERENCE_REQUIREMENTS["inference"]
    )),
}

# Combine all extras
EXTRAS_REQUIRE = {
    **GPU_REQUIREMENTS,
    **FRAMEWORK_REQUIREMENTS,
    **MODALITY_REQUIREMENTS,
    **REASONING_REQUIREMENTS,
    **LOGGING_REQUIREMENTS,
    **UTILITIES_REQUIREMENTS,
    **DEV_REQUIREMENTS,
    **EVAL_REQUIREMENTS,
    **INFERENCE_REQUIREMENTS,
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
    
    # RLAIF generation
    "thinkrl-rlaif": "thinkrl.scripts.generate_rlaif_data:main",

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
    "rlaif", # Added
    "human-feedback",
    "ai-feedback", # Added
    "transformers",
    "large-language-models",
    "llm",
    "nlp",
    "pytorch",
    "artificial-intelligence",
    "ppo",
    "dapo",
    "grpo",
    "vapo",
    "dpo", # Added
    "vllm", # Added
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
    description="Universal RLHF/RLAIF Training Library with state-of-the-art algorithms, reasoning, and multimodal support",
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
            "universal": False, 
        }
    },
)