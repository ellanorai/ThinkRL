---
# ThinkRL: Universal RLHF Training Library

<div align="center">
  <img src="assets/logo.png" alt="ThinkRL Logo" width="200"/>
  <h3>Innovate. Optimize. Scale.</h3>
  <p>A powerful, open-source library for Reinforcement Learning from Human Feedback (RLHF)</p>
  <p>By <a href="https://github.com/Archit03">Archit Sood</a> @ <a href="https://ellanorai.org">EllanorAI</a></p>
  <a href="https://github.com/ellanorai/ThinkRL"><img src="https://img.shields.io/github/stars/ellanorai/ThinkRL?style=social" alt="GitHub Stars"></a>
  <a href="https://pypi.org/project/thinkrl/"><img src="https://img.shields.io/pypi/v/thinkrl" alt="PyPI Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</div>

---

## 🎉 Latest Updates
- **July 2025**: ThinkRL will launch with cutting-edge algorithms such as VAPO, DAPO, GRPO, PPO, and REINFORCE, aiming to set new standards in reasoning performance.

ThinkRL will emerge as an open-source, modular platform for large-scale RLHF, delivering robust algorithm implementations, efficient training infrastructure, and curated datasets. Rooted in modern machine learning principles, it will enable researchers and developers to explore the frontiers of artificial intelligence.

---

## 🚀 Features

### 🧠 State-of-the-Art Algorithms
- **VAPO**: Value-model-based Augmented PPO with Length-adaptive GAE.
- **DAPO**: Decoupled Clip and Dynamic Sampling Policy Optimization.
- **GRPO**: Group Relative Policy Optimization.
- **PPO**: Enhanced Proximal Policy Optimization.
- **REINFORCE**: Policy gradient with variance reduction.

### 🤔 Reasoning Capabilities
- **Chain-of-Thought (CoT)**: Step-by-step reasoning for tackling complex tasks.
- **Tree-of-Thought (ToT)**: Multi-path exploration for generating robust solutions.
- **Long-CoT**: Extended reasoning tailored to intricate problem-solving.
- **Self-Verification**: Automated checks to validate reasoning outputs.

### 🌐 Model Compatibility
- **GPT**: Autoregressive GPT-style models.
- **LLaMA**: LLaMA, LLaMA-2, and Code Llama variants.
- **Qwen**: Qwen-2.5 and future iterations.
- **T5/BART**: Encoder-decoder architectures.
- **Multimodal**: Vision-language models like CLIP and BLIP.

### ⚡ Performance Optimization
- **Zero-Dependency Core**: Minimal setup for basic operations.
- **PEFT**: LoRA and QLoRA for efficient fine-tuning.
- **DeepSpeed**: Distributed training with ZeRO optimization.
- **HuggingFace**: Seamless integration with transformers.
- **Mixed Precision**: FP16/BF16 for faster, resource-efficient training.

### 📊 Dataset Support
- **HuggingFace Datasets**: Effortless data integration.
- **Multimodal Datasets**: Tools for vision-language data processing.
- **Custom Formats**: Flexible preprocessing pipelines.
- **Quality Filtering**: Automated tools for dataset refinement.

---

## 📦 Installation

### Core Installation
```bash
pip install thinkrl
```

### Optional Features
```bash
# HuggingFace transformers
pip install thinkrl[transformers]

# Multimodal models
pip install thinkrl[multimodal]

# Parameter-efficient fine-tuning
pip install thinkrl[peft]

# Distributed training with DeepSpeed
pip install thinkrl[deepspeed]

# Advanced reasoning (CoT/ToT)
pip install thinkrl[reasoning]

# Experiment tracking with Weights & Biases
pip install thinkrl[wandb]

# All features
pip install thinkrl[all]
```

### Specialized Setups
```bash
# State-of-the-art algorithms (VAPO/DAPO)
pip install thinkrl[sota]

# Large-scale distributed training
pip install thinkrl[distributed]

# Full development environment
pip install thinkrl[complete]
```

---

## 🎯 Quick Start

### Basic RLHF Training
```python
from thinkrl import RLHFTrainer, ModelConfig

config = ModelConfig(
    model_name_or_path="microsoft/DialoGPT-small",
    model_type="gpt",
    algorithm="vapo"
)

trainer = RLHFTrainer(config)
trainer.train()
```

### Chain-of-Thought Reasoning
```python
from thinkrl import CoTTrainer, ReasoningConfig

config = ReasoningConfig(
    model_name_or_path="Qwen/Qwen2.5-8B",
    reasoning_type="cot",
    max_reasoning_steps=10
)

trainer = CoTTrainer(config)
trainer.train()
```

### Multimodal Training
```python
from thinkrl import MultimodalTrainer, MultimodalDataset

dataset = MultimodalDataset.from_huggingface(
    "EllanorAI/multimodal-reasoning-dataset"
)

config = ModelConfig(
    model_name_or_path="Salesforce/blip2-opt-2.7b",
    model_type="multimodal",
    vision_encoder="clip"
)

trainer = MultimodalTrainer(config, dataset=dataset)
trainer.train()
```

---

## 🛠️ Command-Line Interface
```bash
# Train with a configuration file
thinkrl train --config configs/vapo_qwen.yaml

# Evaluate a model
thinkrl eval --model-path checkpoints/best --dataset AIFE-2025

# Chain-of-Thought reasoning
thinkrl cot --model Qwen/Qwen2.5-8B --problem "Solve: 2x + 5 = 2025"

# Tree-of-Thought reasoning
thinkrl tot --model Qwen/Qwen2.5-8B --problem "Plan a 7-day Japan itinerary"

# Multimodal training
thinkrl multimodal --config configs/multimodal_training.yaml
```

---

## 🏗️ Project Structure
```plaintext
ThinkRL/
├── algorithms/           # RL algorithms (VAPO, DAPO, GRPO, PPO, etc.)
├── models/               # Model architectures (GPT, LLaMA, multimodal)
├── reasoning/            # CoT and ToT implementations
├── training/             # Training pipelines and distributed support
├── data/                 # Data loading and dataset utilities
├── peft/                 # Parameter-efficient fine-tuning
├── utils/                # Logging, metrics, and helpers
├── configs/              # Training configuration templates
└── logs/                 # Training logs and checkpoints
```

---

## 🤝 Contributing
Contributions will be warmly welcomed! Detailed guidelines will be available in our [Contributing Guide](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/ellanorai/ThinkRL.git
cd ThinkRL
pip install -e .[complete]
pre-commit install
```

---

## 📜 License
ThinkRL will be licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgments
- ByteDance Seed Team will be recognized for their DAPO contributions.
- The research community will be thanked for advancements like VAPO.
- [HuggingFace](https://huggingface.co) will be credited for its transformers ecosystem.
- The open-source ML community will be acknowledged for inspiration and tools.

---

## 📞 Contact
- **Archit Sood**: [@Archit03](https://github.com/Archit03) - [archit@ellanorai.org](mailto:archit@ellanorai.org)
- **EllanorAI**: [https://ellanorai.org](https://ellanorai.org)
- **Project**: [https://github.com/ellanorai/ThinkRL](https://github.com/ellanorai/ThinkRL)

<div align="center">
  ⭐ <strong>Star us on <a href="https://github.com/ellanorai/ThinkRL">GitHub</a> to support ThinkRL!</strong>
  <p>Crafted with ❤️ for AI innovation in India 🇮🇳 by <a href="https://ellanorai.org">EllanorAI</a></p>
</div>

---