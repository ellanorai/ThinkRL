 Priority: Get basic training loop working

Files to implement:
├── thinkrl/__init__.py                    # Package initialization, version
├── thinkrl/utils/logging.py               # Setup logger, get_logger
├── thinkrl/utils/metrics.py               # Basic metrics (rewards, loss)
├── thinkrl/utils/checkpoint.py            # Save/load checkpoints
├── thinkrl/utils/data.py                  # Data utilities
└── tests/test_sample.py                   # Basic import tests

Deliverable: Can import thinkrl, log messages, save/load files
```

### **Week 3-4: Base Classes & Models**
```
 Priority: Define interfaces and one working model

Files to implement:
├── thinkrl/models/base.py                 # BaseModel, ModelConfig, ModelProtocol
├── thinkrl/models/critics.py              # ValueHead, TransformerCritic, SharedBackboneCritic
├── thinkrl/models/gpt.py                  # GPTModel (using HuggingFace)
├── thinkrl/algorithms/base.py             # AlgorithmConfig, BaseAlgorithm, AlgorithmOutput
└── tests/test_models/test_base.py         # Model tests

Deliverable: Can load GPT-2, create critic, run forward pass
```

### **Week 5-6: First Algorithm (PPO)**
```
 Priority: ONE complete algorithm working end-to-end

Files to implement:
├── thinkrl/algorithms/ppo.py              # PPO, PPOConfig, PPOLoss, GAE
│   ├── PPO.__init__(actor, critic)
│   ├── compute_values()
│   ├── compute_advantages()
│   └── compute_loss()
├── tests/test_algorithms/test_ppo.py      # PPO unit tests
└── tests/test_algorithms/base.py          # Test utilities (MockModel)

Deliverable: PPO algorithm passes all tests
```

### **Week 7-8: Basic Training Loop**
```
 Priority: Train GPT-2 with PPO on simple dataset

Files to implement:
├── thinkrl/data/datasets.py               # RLHFDataset, PreferenceDataset
├── thinkrl/data/loaders.py                # RLHFDataLoader
├── thinkrl/training/trainer.py            # RLHFTrainer (basic version)
│   ├── __init__(config, actor, critic)
│   ├── train()
│   ├── evaluate()
│   └── save_checkpoint()
├── thinkrl/scripts/train.py               # CLI training script
└── examples/basic/train_simple.py         # Simple training example

Deliverable: `python examples/basic/train_simple.py` trains GPT-2 with PPO
```

---

##  **Phase 2: Core Algorithms (Weeks 9-16)**

### **Week 9-10: REINFORCE**
```
Files to implement:
├── thinkrl/algorithms/reinforce.py        # REINFORCE (simplest, no critic)
│   ├── REINFORCEConfig
│   ├── compute_returns()
│   └── compute_policy_loss()
├── tests/test_algorithms/test_reinforce.py
└── examples/basic/train_reinforce.py

Deliverable: REINFORCE working, compare with PPO
```

### **Week 11-13: GRPO (with optional critic)**
```
Files to implement:
├── thinkrl/algorithms/grpo.py             # GRPO + optional critic
│   ├── GRPOConfig (use_critic=False)
│   ├── compute_group_rewards()
│   ├── compute_critic_baseline()         # NEW: Optional critic logic
│   ├── compute_advantages()              # Hybrid mode
│   └── GRPORewardNormalizer
├── tests/test_algorithms/test_grpo.py
└── examples/advanced/grpo_with_critic.py

Deliverable: 
- Pure GRPO working
- GRPO + critic working
- Benchmark comparison document
```

### **Week 14-16: Algorithm Registry & Integration**
```
Files to implement:
├── thinkrl/algorithms/__init__.py         # get_algorithm(), list_algorithms()
├── thinkrl/registry/algorithms.py         # Algorithm registry system
├── thinkrl/scripts/evaluate.py            # Evaluation script
└── thinkrl/evaluation/evaluators.py       # RLHFEvaluator

Deliverable: Can switch algorithms via config, evaluate trained models
```

---

##  **Phase 3: State-of-the-Art Algorithms (Weeks 17-24)**

### **Week 17-19: VAPO (Value-Augmented PPO)**
```
Files to implement:
├── thinkrl/algorithms/vapo.py             # VAPO (needs critic!)
│   ├── VAPOConfig
│   ├── Length-adaptive GAE
│   ├── Value model training
│   └── compute_value_augmented_advantages()
├── tests/test_algorithms/test_vapo.py
└── configs/algos/vapo_config.yaml

Deliverable: VAPO working, benchmark vs PPO on math/reasoning tasks
```

### **Week 20-22: DAPO (Decoupled Advantage PPO)**
```
Files to implement:
├── thinkrl/algorithms/dapo.py             # DAPO (needs critic!)
│   ├── DAPOConfig
│   ├── Decoupled clipping
│   ├── Dynamic sampling
│   ├── DAPOAdvantageEstimator
│   └── DAPOSampler
├── tests/test_algorithms/test_dapo.py
└── configs/algos/dapo_config.yaml

Deliverable: DAPO working, benchmark vs VAPO
```

### **Week 23-24: Multi-Model Support**
```
Files to implement:
├── thinkrl/models/llama.py                # LLaMA support
├── thinkrl/models/qwen.py                 # Qwen support
├── thinkrl/models/__init__.py             # get_model() factory
└── thinkrl/registry/models.py             # Model registry

Deliverable: Can train LLaMA-2, Qwen-2.5 with any algorithm
```

---

##  **Phase 4: Reasoning Capabilities (Weeks 25-32)**

### **Week 25-27: Chain-of-Thought (CoT)**
```
Files to implement:
├── thinkrl/reasoning/config.py            # ReasoningConfig
├── thinkrl/reasoning/cot/prompts.py       # COT_PROMPTS, templates
├── thinkrl/reasoning/cot/cot.py           # ChainOfThought, CoTConfig
├── thinkrl/training/cot_trainer.py        # CoTTrainer
├── thinkrl/scripts/chain_of_thought.py    # CLI for CoT
├── tests/test_reasoning/test_cot.py
└── examples/reasoning/chain_of_thought.py

Deliverable: Train models with CoT reasoning, CLI works
```

### **Week 28-30: Tree-of-Thought (ToT)**
```
Files to implement:
├── thinkrl/reasoning/tot/tree.py          # ThoughtTree, TreeNode
├── thinkrl/reasoning/tot/evaluator.py     # ThoughtEvaluator
├── thinkrl/reasoning/tot/tot.py           # TreeOfThought, ToTConfig
├── thinkrl/training/tot_trainer.py        # ToTTrainer
├── thinkrl/scripts/tree_of_thought.py     # CLI for ToT
├── tests/test_reasoning/test_tot.py
└── examples/reasoning/tree_of_thought.py

Deliverable: ToT reasoning working, visual tree outputs
```

### **Week 31-32: Reasoning Integration**
```
Files to implement:
├── thinkrl/reasoning/__init__.py          # Unified reasoning API
├── configs/reasoning/cot_config.yaml
├── configs/reasoning/tot_config.yaml
└── examples/reasoning/multi_step_reasoning.py

Deliverable: Seamless switching between CoT/ToT modes
```

---

##  **Phase 5: Multimodal & Advanced Features (Weeks 33-40)**

### **Week 33-35: Multimodal Foundation**
```
Files to implement:
├── thinkrl/models/multimodal.py           # MultimodalModel, MultimodalConfig
├── thinkrl/data/processors.py             # process_image, process_audio, process_text
├── thinkrl/training/multimodal_trainer.py # MultimodalTrainer
├── tests/test_models/test_multimodal.py
└── examples/multimodal/vision_language.py

Deliverable: Train vision-language models with RLHF
```

### **Week 36-37: Distributed Training**
```
Files to implement:
├── thinkrl/training/distributed.py        # DistributedTrainer (DeepSpeed/FSDP)
├── thinkrl/scripts/multimodal_train.py    # Multimodal CLI
└── examples/advanced/distributed_training.py

Deliverable: Multi-GPU training working
```

### **Week 38-40: PEFT Integration**
```
Files to implement:
├── thinkrl/peft/config.py                 # PEFTConfig (LoRA/QLoRA)
├── thinkrl/peft/peft_model.py             # PEFT wrapper
├── thinkrl/peft/optimizer.py              # PEFT-aware optimizers
├── configs/peft/lora_config.yaml
└── examples/advanced/lora_training.py

Deliverable: LoRA/QLoRA fine-tuning with any algorithm
```

---

##  **Phase 6: Evaluation & Benchmarks (Weeks 41-44)**

### **Week 41-42: Evaluation Suite**
```
Files to implement:
├── thinkrl/evaluation/metrics.py          # compute_reward, compute_kl, accuracy
├── thinkrl/evaluation/benchmarks.py       # BenchmarkSuite, AIFEBenchmark
├── configs/eval_config.yaml
└── examples/basic/evaluate_model.py

Deliverable: Comprehensive evaluation metrics
```

### **Week 43-44: Documentation & Polish**
```
Files to complete:
├── README.md                              # Update with real examples
├── CONTRIBUTING.md                        # Complete contribution guide
├── CHANGELOG.md                           # Document all changes
├── docs/                                  # Full documentation
│   ├── installation.md
│   ├── quickstart.md
│   ├── algorithms.md
│   ├── reasoning.md
│   └── api_reference.md
└── All config files                       # Complete all YAML configs

Deliverable: Production-ready documentation