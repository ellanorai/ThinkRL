ThinkRL: Universal RLHF Training Library
<div align="center"> <img src="assets/logo.png" alt="ThinkRL Logo" width="200"/>
Think Different. Train Smarter. Scale Further.

A comprehensive reinforcement learning library for human feedback training

By Archit Sood @ EllanorAI

Show Image
Show Image
Show Image
Show Image

</div>
🔥 Important
🎉 News!!!
[2025/06] We release ThinkRL with support for VAPO, DAPO, GRPO, PPO, and REINFORCE algorithms, achieving state-of-the-art results on reasoning benchmarks.
[2025/06] Full multimodal support for vision-language models with HuggingFace integration and Chain-of-Thought (CoT) + Tree-of-Thought (ToT) reasoning capabilities.
[2025/06] Complete zero-dependency philosophy - minimal core requirements with optional advanced features.
We release a fully open-sourced system for large-scale LLM RL, including algorithm implementations, training infrastructure, and curated datasets. The system achieves state-of-the-art large-scale LLM RL performance across multiple benchmarks. We propose implementations of the latest algorithms including VAPO (Value-model-based Augmented Proximal Policy Optimization) and DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization).

Our system is built on solid engineering principles and modern ML infrastructure. Thanks to the amazing open-source community for making this possible!

🚀 Features
🧠 State-of-the-Art Algorithms
VAPO: Value-model-based training with Length-adaptive GAE
DAPO: Decoupled clipping with dynamic sampling
GRPO: Group Relative Policy Optimization
PPO: Proximal Policy Optimization with modern improvements
REINFORCE: Classic policy gradient with variance reduction
🎯 Reasoning Capabilities
Chain-of-Thought (CoT): Step-by-step reasoning training
Tree-of-Thought (ToT): Multi-path reasoning exploration
Long-CoT: Extended reasoning for complex problems
Self-verification: Automated reasoning validation
🌐 Universal Model Support
GPT: All GPT-style autoregressive models
LLaMA: LLaMA, LLaMA-2, Code Llama variants
Qwen: Qwen-2.5 and latest versions
T5/BART: Encoder-decoder architectures
Multimodal: Vision-language models (CLIP, BLIP, etc.)
⚡ High-Performance Training
Minimal Dependencies: Core functionality with minimal requirements
PEFT Integration: LoRA, QLoRA, and parameter-efficient methods
DeepSpeed Support: Distributed training with ZeRO optimization
HuggingFace Native: Seamless integration with transformers ecosystem
Mixed Precision: FP16/BF16 training for efficiency
📊 Rich Dataset Support
HuggingFace Datasets: Direct integration with datasets library
Multimodal Datasets: Vision-language dataset support
Custom Formats: Flexible data loading and preprocessing
Automatic Curation: Built-in dataset quality filtering
📦 Installation
Core Installation
bash
# Minimal installation - just the essentials 
pip install thinkrl
Feature-Specific Installations
bash
# For HuggingFace transformer models 
pip install thinkrl[transformers] 
 
# For multimodal (vision-language) models 
pip install thinkrl[multimodal] 
 
# For parameter-efficient fine-tuning 
pip install thinkrl[peft] 
 
# For distributed training with DeepSpeed 
pip install thinkrl[deepspeed] 
 
# For advanced reasoning (CoT/ToT) 
pip install thinkrl[reasoning] 
 
# For experiment tracking with Weights & Biases 
pip install thinkrl[wandb] 
 
# Everything included 
pip install thinkrl[all]
Algorithm-Specific Shortcuts
bash
# For VAPO/DAPO state-of-the-art algorithms 
pip install thinkrl[sota] 
 
# For large-scale distributed training 
pip install thinkrl[distributed] 
 
# Complete development setup 
pip install thinkrl[complete]
🎯 Quick Start
Basic RLHF Training
python
from thinkrl import RLHFTrainer, ModelConfig 
 
# Configure your model 
config = ModelConfig( 
    model_name_or_path="microsoft/DialoGPT-small", 
    model_type="gpt", 
    algorithm="vapo"  # or "dapo", "grpo", "ppo" 
) 
 
# Create trainer and start training 
trainer = RLHFTrainer(config) 
trainer.train()
Advanced Reasoning Training
python
from thinkrl import CoTTrainer, ToTTrainer 
from thinkrl.reasoning import ReasoningConfig 
 
# Chain-of-Thought training 
cot_config = ReasoningConfig( 
    model_name_or_path="Qwen/Qwen2.5-32B", 
    reasoning_type="cot", 
    max_reasoning_steps=10 
) 
 
cot_trainer = CoTTrainer(cot_config) 
cot_trainer.train()
Multimodal Training
python
from thinkrl import MultimodalTrainer 
from thinkrl.data import MultimodalDataset 
 
# Load multimodal dataset from HuggingFace 
dataset = MultimodalDataset.from_huggingface( 
    "EllanorAI/multimodal-reasoning-dataset" 
) 
 
# Configure multimodal model 
config = ModelConfig( 
    model_name_or_path="Salesforce/blip2-opt-2.7b", 
    model_type="multimodal", 
    vision_encoder="clip" 
) 
 
trainer = MultimodalTrainer(config, dataset=dataset) 
trainer.train()
🛠️ Command Line Interface
bash
# Train with configuration file 
thinkrl-train --config configs/vapo_qwen.yaml 
 
# Evaluate trained model 
thinkrl-eval --model-path ./checkpoints/best --dataset AIME-2024 
 
# Chain-of-Thought reasoning 
thinkrl-cot --model Qwen/Qwen2.5-7B --problem "Solve: 2x + 5 = 13" 
 
# Tree-of-Thought reasoning 
thinkrl-tot --model Qwen/Qwen2.5-7B --problem "Plan a 7-day trip to Japan" 
 
# Multimodal training 
thinkrl-multimodal --config configs/multimodal_training.yaml
🏗️ Architecture
ThinkRL/ 
├── algorithms/           # RL algorithms (VAPO, DAPO, GRPO, PPO) 
├── models/              # Model architectures & multimodal support 
├── reasoning/           # CoT, ToT reasoning implementations 
├── training/            # Training infrastructure & distributed support 
├── data/                # Data loading & multimodal datasets 
├── peft/                # Parameter-efficient fine-tuning 
├── utils/               # Utilities, metrics, and logging 
└── configs/             # Configuration templates
🤝 Contributing
We welcome contributions! Please see our Contributing Guide for details.

Development Setup
bash
git clone https://github.com/Archit03/ThinkRL.git 
cd ThinkRL 
pip install -e .[complete] 
pre-commit install
📜 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

🙏 Acknowledgments
Thanks to the ByteDance Seed team for open-sourcing DAPO
Thanks to the research community for VAPO and other algorithms
Thanks to HuggingFace for the transformers ecosystem
Thanks to the open-source ML community
📞 Contact
Archit Sood - @Archit03 - archit@ellanorai.org

EllanorAI - https://ellanorai.org

Project Link: https://github.com/Archit03/ThinkRL

<div align="center">
⭐ Star us on GitHub if this project helped you! ⭐

Made with ❤️ and passion for AI & Science in India 🇮🇳 by EllanorAI

</div>
