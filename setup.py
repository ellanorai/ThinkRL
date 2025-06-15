from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies (minimal, no CUDA requirements)
install_requires = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "accelerate>=0.21.0",  # Hugging Face Accelerate for distributed training
]

# Optional dependencies
extras_require = {
    "cuda": [
        "cupy-cuda12x>=12.0.0",  # Only install CuPy when explicitly requested
    ],
    "transformers": [
        "transformers>=4.30.0",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
    ],
    "multimodal": [
        "transformers>=4.30.0", 
        "pillow>=9.0.0",
        "torchvision>=0.15.0",
    ],
    "peft": [
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
    ],
    "deepspeed": [
        "deepspeed>=0.9.0",
    ],
    "reasoning": [
        "torch>=2.0.0", 
        "transformers>=4.30.0",
        "networkx>=3.1",
    ],
    "wandb": [
        "wandb>=0.15.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "black>=23.9.0",
        "isort>=5.12.0",
        "pre-commit>=3.0.0",
    ],
}

# Convenience extras
extras_require.update({
    "all": [
        "cupy-cuda12x>=12.0.0",  # Include CUDA support in 'all'
        "transformers>=4.30.0",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
        "pillow>=9.0.0",
        "torchvision>=0.15.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "deepspeed>=0.9.0",
        "wandb>=0.15.0",
        "networkx>=3.1",
    ],
    "sota": [
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "cupy-cuda12x>=12.0.0",  # SOTA algorithms benefit from GPU
    ],
    "distributed": [
        "deepspeed>=0.9.0",
        "torch>=2.0.0",
        "accelerate>=0.21.0",
    ],
    "complete": [
        "cupy-cuda12x>=12.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.15.0",
        "datasets>=2.14.0",
        "pillow>=9.0.0",
        "torchvision>=0.15.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "deepspeed>=0.9.0",
        "wandb>=0.15.0",
        "networkx>=3.1",
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "black>=23.9.0",
        "isort>=5.12.0",
        "pre-commit>=3.0.0",
    ],
})

setup(
    name="thinkrl",
    version="0.1.0",
    author="Archit Sood",
    author_email="archit@ellanorai.org",
    description="Universal RLHF Training Library for advanced algorithms, reasoning, and multimodal models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Archit03/ThinkRL",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "thinkrl=thinkrl.scripts.train:main",
            "thinkrl-eval=thinkrl.scripts.evaluate:main",
            "thinkrl-cot=thinkrl.scripts.chain_of_thought:main",
            "thinkrl-tot=thinkrl.scripts.tree_of_thought:main",
            "thinkrl-multimodal=thinkrl.scripts.multimodal_train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "deep-learning", 
        "reinforcement-learning",
        "rlhf",
        "transformers",
        "llm",
        "pytorch",
        "artificial-intelligence",
    ],
    project_urls={
        "Homepage": "https://github.com/Archit03/ThinkRL",
        "Documentation": "https://github.com/Archit03/ThinkRL/docs",
        "Bug Reports": "https://github.com/Archit03/ThinkRL/issues",
        "Source": "https://github.com/Archit03/ThinkRL",
        "Company": "https://ellanorai.org",
    },
)