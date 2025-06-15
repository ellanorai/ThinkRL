from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
install_requires = [
    "cupy-cuda12x>=12.0.0",  # CuPy for CUDA 12.x
    "torch>=2.0.0",
    "pyyaml>=6.0",
    "accelerate>=0.21.0",  # Hugging Face Accelerate for distributed training
]

# Optional dependencies
extras_require = {
    "transformers": ["transformers>=4.30.0"],
    "multimodal": ["transformers>=4.30.0", "pillow>=9.0.0"],
    "peft": ["peft>=0.4.0"],
    "deepspeed": ["deepspeed>=0.9.0"],
    "reasoning": ["torch>=2.0.0", "transformers>=4.30.0"],
    "wandb": ["wandb>=0.15.0"],
    "all": [
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "peft>=0.4.0",
        "deepspeed>=0.9.0",
        "wandb>=0.15.0",
    ],
    "sota": ["transformers>=4.30.0"],
    "distributed": [
        "deepspeed>=0.9.0",
        "torch>=2.0.0",
        "accelerate>=0.21.0",
    ],
    "complete": [
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "peft>=0.4.0",
        "deepspeed>=0.9.0",
        "wandb>=0.15.0",
        "pytest>=7.0.0",
        "pre-commit>=3.0.0",
        "accelerate>=0.21.0",
    ],
}

setup(
    name="thinkrl",
    version="0.1.0",
    author="Archit Sood",
    author_email="archit@ellanorai.org",
    description="Universal RLHF Training Library for advanced algorithms, reasoning, and multimodal models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Archit03/ThinkRL",
    packages=find_namespace_packages(include=["thinkrl*"]),  # Improved package discovery
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
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
    include_package_data=True,  # Include non-Python files
    exclude_package_data={
        "": ["*.cu", "*.cpp"],  # Exclude C++ and CUDA source files
    },
)