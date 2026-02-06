"""
ThinkRL CLI Main
================

Command-line interface for ThinkRL RLHF training.

Commands:
    train     - Run RLHF training with config file
    sft       - Supervised Fine-Tuning (like trl sft)
    dpo       - Direct Preference Optimization
    ppo       - Proximal Policy Optimization
    grpo      - Group Relative Policy Optimization
    reward    - Train reward model
    generate  - Generate rollouts
    merge     - Merge LoRA adapters
    info      - Show configuration info

Author: EllanorAI
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Annotated


# Optional typer support
try:
    import typer
    from typer import Argument, Option

    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    typer = None
    Argument = None
    Option = None


logger = logging.getLogger(__name__)


def _check_typer():
    """Check if typer is available."""
    if not TYPER_AVAILABLE:
        logger.error("Error: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)


# Create app if typer is available
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="thinkrl",
        help="ThinkRL: RLHF Training Framework for Reasoning Models",
        add_completion=False,
        rich_markup_mode="rich",
    )

    @app.command()
    def train(
        config: Annotated[Path, Option("--config", "-c", help="Path to config YAML/JSON file")],
        overrides: Annotated[list[str] | None, Option("--override", "-o", help="Config overrides (key=value)")] = None,
        resume: Annotated[Path | None, Option("--resume", "-r", help="Path to checkpoint to resume from")] = None,
        dry_run: Annotated[bool, Option("--dry-run", help="Validate config without running")] = False,
    ):
        """
        Run RLHF training with the specified configuration.

        Example:
            thinkrl train --config configs/ppo.yaml
            thinkrl train -c configs/ppo.yaml -o algorithm.learning_rate=1e-5
        """
        from thinkrl.config import load_config, merge_configs

        # Load config
        cfg = load_config(config)

        # Apply overrides
        if overrides:
            override_dict = {}
            for override in overrides:
                if "=" not in override:
                    typer.echo(f"Invalid override format: {override} (expected key=value)")
                    raise typer.Exit(1)
                key, value = override.split("=", 1)
                # Try to parse as JSON for proper typing
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
                override_dict[key] = value
            cfg = merge_configs(cfg, override_dict)

        # Validate
        errors = cfg.validate()
        if errors:
            typer.echo("Configuration errors:")
            for error in errors:
                typer.echo(f"  - {error}")
            raise typer.Exit(1)

        if dry_run:
            typer.echo("Configuration is valid!")
            typer.echo(json.dumps(cfg.to_dict(), indent=2))
            return

        # Run training
        typer.echo(f"Starting training with algorithm: {cfg.algorithm.name}")
        typer.echo(f"Model: {cfg.model.name_or_path}")
        typer.echo(f"Strategy: {cfg.distributed.strategy}")

        # TODO: Implement actual training loop
        typer.echo("\nNote: Training implementation pending")

    @app.command()
    def generate(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        prompts: Annotated[Path, Option("--prompts", "-p", help="Path to prompts JSONL file")],
        output: Annotated[Path, Option("--output", "-o", help="Output path for generations")] = Path(
            "outputs/generations.jsonl"
        ),
        max_new_tokens: Annotated[int, Option("--max-new-tokens", help="Maximum tokens to generate")] = 512,
        temperature: Annotated[float, Option("--temperature", "-t", help="Sampling temperature")] = 0.7,
        batch_size: Annotated[int, Option("--batch-size", "-b", help="Batch size")] = 8,
    ):
        """
        Generate completions for prompts.

        Example:
            thinkrl generate --model meta-llama/Llama-3.1-8B --prompts data/prompts.jsonl
        """
        typer.echo(f"Generating from model: {model}")
        typer.echo(f"Prompts: {prompts}")
        typer.echo(f"Output: {output}")

        # TODO: Implement generation
        typer.echo("\nNote: Generation implementation pending")

    @app.command()
    def merge(
        base_model: Annotated[str, Option("--base-model", help="Base model name or path")],
        adapter: Annotated[Path, Option("--adapter", "-a", help="Path to LoRA adapter")],
        output: Annotated[Path, Option("--output", "-o", help="Output path for merged model")],
        push_to_hub: Annotated[
            str | None, Option("--push-to-hub", help="Push merged model to HuggingFace Hub")
        ] = None,
    ):
        """
        Merge LoRA adapter weights into base model.

        Example:
            thinkrl merge --base-model meta-llama/Llama-3.1-8B --adapter ./adapter --output ./merged
        """
        from thinkrl.peft import is_peft_available, merge_lora_weights

        if not is_peft_available():
            typer.echo("Error: peft library required. Install with: pip install peft")
            raise typer.Exit(1)

        typer.echo(f"Merging adapter: {adapter}")
        typer.echo(f"Base model: {base_model}")
        typer.echo(f"Output: {output}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from peft import PeftModel

            # Load base model
            typer.echo("Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            # Load adapter
            typer.echo("Loading adapter...")
            model = PeftModel.from_pretrained(model, adapter)

            # Merge
            typer.echo("Merging weights...")
            merged_model = merge_lora_weights(model)

            # Save
            typer.echo(f"Saving to {output}...")
            merged_model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)

            # Push to hub if requested
            if push_to_hub:
                typer.echo(f"Pushing to hub: {push_to_hub}")
                merged_model.push_to_hub(push_to_hub)
                tokenizer.push_to_hub(push_to_hub)

            typer.echo("Done!")

        except Exception as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(1) from e

    @app.command()
    def info(
        config: Annotated[Path | None, Option("--config", "-c", help="Show config file info")] = None,
    ):
        """
        Show ThinkRL information.

        Example:
            thinkrl info
            thinkrl info --config configs/ppo.yaml
        """
        import thinkrl

        typer.echo(f"ThinkRL v{thinkrl.__version__}")
        typer.echo()

        # Show available algorithms
        from thinkrl.algorithms import ALGORITHMS

        typer.echo("Available algorithms:")
        for name in ALGORITHMS:
            typer.echo(f"  - {name}")
        typer.echo()

        # Show config if provided
        if config:
            from thinkrl.config import load_config

            cfg = load_config(config)
            typer.echo(f"Configuration from: {config}")
            typer.echo(json.dumps(cfg.to_dict(), indent=2))

        # Check dependencies
        typer.echo("Dependencies:")

        try:
            import torch

            typer.echo(f"  - PyTorch: {torch.__version__}")
            typer.echo(f"    CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                typer.echo(f"    CUDA version: {torch.version.cuda}")
        except ImportError:
            typer.echo("  - PyTorch: NOT INSTALLED")

        try:
            import transformers

            typer.echo(f"  - Transformers: {transformers.__version__}")
        except ImportError:
            typer.echo("  - Transformers: NOT INSTALLED")

        try:
            import peft

            typer.echo(f"  - PEFT: {peft.__version__}")
        except ImportError:
            typer.echo("  - PEFT: NOT INSTALLED")

        try:
            import deepspeed

            typer.echo(f"  - DeepSpeed: {deepspeed.__version__}")
        except ImportError:
            typer.echo("  - DeepSpeed: NOT INSTALLED")

    # =========================================================================
    # TRL-style single-command training interfaces
    # =========================================================================

    @app.command()
    def sft(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Dataset name or path")],
        output_dir: Annotated[str, Option("--output-dir", "-o", help="Output directory")] = "./sft_output",
        max_seq_length: Annotated[int, Option("--max-seq-length", help="Maximum sequence length")] = 2048,
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 2e-5,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 3,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        gradient_accumulation_steps: Annotated[
            int, Option("--gradient-accumulation-steps", "--gas", help="Gradient accumulation steps")
        ] = 1,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
        packing: Annotated[bool, Option("--packing/--no-packing", help="Use sequence packing")] = False,
        push_to_hub: Annotated[str | None, Option("--push-to-hub", help="Push to HuggingFace Hub repo")] = None,
    ):
        """
        Supervised Fine-Tuning (SFT) - Similar to `trl sft`.

        Train a model on instruction-response pairs.

        Example:
            thinkrl sft --model meta-llama/Llama-3.1-8B --dataset tatsu-lab/alpaca
            thinkrl sft -m meta-llama/Llama-3.2-1B -d imdb --epochs 1 --lora-r 8
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Supervised Fine-Tuning (SFT)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Max seq length: {max_seq_length}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"Gradient accumulation: {gradient_accumulation_steps}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo(f"Packing: {packing}")
        typer.echo()

        # TODO: Implement SFT training
        typer.echo("Note: SFT training implementation pending")
        typer.echo("This will use thinkrl.training.SFTTrainer")

    @app.command()
    def dpo(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Preference dataset name or path")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./dpo_output"),
        beta: Annotated[float, Option("--beta", help="DPO beta parameter")] = 0.1,
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        loss_type: Annotated[str, Option("--loss-type", help="Loss type: sigmoid, hinge, ipo")] = "sigmoid",
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    ):
        """
        Direct Preference Optimization (DPO) - Similar to `trl dpo`.

        Train a model on preference pairs (chosen vs rejected).

        Example:
            thinkrl dpo --model meta-llama/Llama-3.1-8B --dataset Anthropic/hh-rlhf
            thinkrl dpo -m meta-llama/Llama-3.2-1B -d argilla/dpo-mix-7k --beta 0.1 --lora-r 8
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Direct Preference Optimization (DPO)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Beta: {beta}")
        typer.echo(f"Loss type: {loss_type}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo()

        # TODO: Implement DPO training
        typer.echo("Note: DPO training implementation pending")
        typer.echo("This will use thinkrl.algorithms.DPOAlgorithm")

    @app.command()
    def ppo(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        reward_model: Annotated[str, Option("--reward-model", "-r", help="Reward model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./ppo_output"),
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
        kl_coeff: Annotated[float, Option("--kl-coeff", help="KL penalty coefficient")] = 0.1,
        clip_range: Annotated[float, Option("--clip-range", "--epsilon", help="PPO clip range")] = 0.2,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    ):
        """
        Proximal Policy Optimization (PPO).

        Train a model with PPO using a reward model.

        Example:
            thinkrl ppo --model meta-llama/Llama-3.1-8B --reward-model OpenAssistant/reward-model-deberta-v3-large-v2 --dataset Anthropic/hh-rlhf
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Proximal Policy Optimization (PPO)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Reward Model: {reward_model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"KL coefficient: {kl_coeff}")
        typer.echo(f"Clip range: {clip_range}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo()

        # TODO: Implement PPO training
        typer.echo("Note: PPO training implementation pending")
        typer.echo("This will use thinkrl.algorithms.PPOAlgorithm")

    @app.command()
    def grpo(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./grpo_output"),
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
        kl_coeff: Annotated[float, Option("--kl-coeff", help="KL penalty coefficient")] = 0.1,
        group_size: Annotated[int, Option("--group-size", "-g", help="Number of samples per prompt")] = 4,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
        reward_fn: Annotated[str | None, Option("--reward-fn", help="Path to reward function module")] = None,
    ):
        """
        Group Relative Policy Optimization (GRPO).

        Critic-free RL algorithm using group-relative advantages.
        Similar to DeepSeek-R1's training approach.

        Example:
            thinkrl grpo --model meta-llama/Llama-3.1-8B --dataset math_dataset --group-size 8
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Group Relative Policy Optimization (GRPO)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"KL coefficient: {kl_coeff}")
        typer.echo(f"Group size: {group_size}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo(f"Reward function: {reward_fn if reward_fn else 'Default'}")
        typer.echo()

        # TODO: Implement GRPO training
        typer.echo("Note: GRPO training implementation pending")
        typer.echo("This will use thinkrl.algorithms.GRPOAlgorithm")

    @app.command()
    def reward(
        model: Annotated[str, Option("--model", "-m", help="Base model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Preference dataset name or path")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./reward_output"),
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-5,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    ):
        """
        Train a Reward Model.

        Train a reward model on preference pairs.

        Example:
            thinkrl reward --model meta-llama/Llama-3.1-8B --dataset Anthropic/hh-rlhf
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Reward Model Training")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo()

        # TODO: Implement reward model training
        typer.echo("Note: Reward model training implementation pending")
        typer.echo("This will use thinkrl.models.RewardModel")

    @app.command()
    def orpo(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Preference dataset name or path")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./orpo_output"),
        beta: Annotated[float, Option("--beta", help="ORPO beta parameter (odds ratio weight)")] = 0.1,
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-5,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    ):
        """
        Odds Ratio Preference Optimization (ORPO).

        No reference model needed - combines SFT and preference optimization.

        Example:
            thinkrl orpo --model meta-llama/Llama-3.1-8B --dataset argilla/dpo-mix-7k
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Odds Ratio Preference Optimization (ORPO)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Beta: {beta}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo()

        # TODO: Implement ORPO training
        typer.echo("Note: ORPO training implementation pending")
        typer.echo("This will use thinkrl.algorithms.ORPOAlgorithm")

    @app.command()
    def kto(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Dataset with binary feedback")],
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./kto_output"),
        beta: Annotated[float, Option("--beta", help="KTO beta parameter")] = 0.1,
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    ):
        """
        Kahneman-Tversky Optimization (KTO).

        Works with binary feedback (desirable/undesirable) using prospect theory.

        Example:
            thinkrl kto --model meta-llama/Llama-3.1-8B --dataset trl-lib/kto-mix
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL Kahneman-Tversky Optimization (KTO)")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"Beta: {beta}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
        typer.echo(f"BF16: {bf16}")
        typer.echo()

        # TODO: Implement KTO training
        typer.echo("Note: KTO training implementation pending")
        typer.echo("This will use thinkrl.algorithms.KTOAlgorithm")

    @app.command()
    def reinforce_pp(
        model: Annotated[str, Option("--model", "-m", help="Model name or path")],
        dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
        dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
        dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name (e.g., 'main' for gsm8k)")] = None,
        prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
        source: Annotated[
            str, Option("--source", "-s", help="Dataset source: 'hf' (HuggingFace), 'local', 'json', 'csv'")
        ] = "hf",
        ref_model: Annotated[
            str | None, Option("--ref-model", "-r", help="Reference model name or path (required)")
        ] = None,
        output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path(
            "./reinforce_pp_output"
        ),
        learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
        mode: Annotated[str, Option("--mode", help="Mode: 'general' (k=1) or 'baseline' (k>1)")] = "baseline",
        group_size: Annotated[int, Option("--group-size", "-g", help="Group size (for baseline mode)")] = 4,
        entropy_coeff: Annotated[float, Option("--entropy-coeff", help="Entropy coefficient")] = 0.01,
        kl_loss_coeff: Annotated[float, Option("--kl-coeff", help="KL loss coefficient (for baseline mode)")] = 0.1,
        num_train_epochs: Annotated[
            int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")
        ] = 1,
        per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
        lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
        lora_init: Annotated[
            str, Option("--lora-init", help="LoRA init type: 'default', 'garbage', 'pissa', 'pissa_niter_[n]'")
        ] = "default",
        grad_accum: Annotated[int, Option("--grad-accum", "-ga", help="Gradient accumulation steps")] = 1,
        bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
        fp16: Annotated[bool, Option("--fp16/--no-fp16", help="Use float16 precision")] = False,
        use_flash_attention: Annotated[
            bool, Option("--flash-attn/--no-flash-attn", help="Use Flash Attention 2")
        ] = False,
        reward_fn: Annotated[
            str | None, Option("--reward-fn", help="Path to reward function (module.py:func_name)")
        ] = None,
        deepspeed: Annotated[str | None, Option("--deepspeed", help="Path to DeepSpeed configuration file")] = None,
        use_vllm: Annotated[bool, Option("--use-vllm", help="Use VLLM for generation")] = False,
        vllm_group_port: Annotated[int, Option("--vllm-group-port", help="NCCL group port for VLLM sync")] = 51216,
        gradient_checkpointing: Annotated[
            bool,
            Option(
                "--gradient-checkpointing/--no-gradient-checkpointing",
                help="Enable gradient checkpointing to save memory",
            ),
        ] = False,
        logging_backend: Annotated[
            str, Option("--logging-backend", help="Logging backend: 'tensorboard', 'wandb', or 'none'")
        ] = "tensorboard",
        wandb_project: Annotated[
            str, Option("--wandb-project", help="WandB project name (if using wandb)")
        ] = "thinkrl-reinforce-pp",
        max_length: Annotated[int, Option("--max-length", help="Maximum sequence length")] = 512,
        max_samples: Annotated[
            int | None, Option("--max-samples", help="Maximum number of samples to load from dataset")
        ] = None,
        target_column: Annotated[str, Option("--target-column", help="Column name for target answers")] = "answer",
    ):
        """
        REINFORCE++ Policy Optimization.

        Critic-free RL algorithm with global advantage normalization.
        Can run in 'general' mode (standard PPO-like reward) or 'baseline' mode (group relative).

        Example:
            thinkrl reinforce_pp --model meta-llama/Llama-3.1-8B --dataset math_dataset --mode baseline --group-size 4
        """
        typer.echo("=" * 60)
        typer.echo("ThinkRL REINFORCE++")
        typer.echo("=" * 60)
        typer.echo(f"Model: {model}")
        typer.echo(f"Ref Model: {ref_model}")
        typer.echo(f"Dataset: {dataset}")
        typer.echo(f"Dataset Source: {source}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"DeepSpeed: {deepspeed}")
        typer.echo(f"Mode: {mode}")
        typer.echo(f"Group Size: {group_size}")
        typer.echo(f"Learning rate: {learning_rate}")
        typer.echo(f"LoRA Rank: {lora_r}")
        typer.echo(f"LoRA Init: {lora_init}")
        typer.echo(f"Entropy Coeff: {entropy_coeff}")
        typer.echo(f"KL Coeff: {kl_loss_coeff}")
        typer.echo(f"Epochs: {num_train_epochs}")
        typer.echo(f"Batch size: {per_device_train_batch_size}")
        typer.echo(f"BF16 Enabled: {bf16}")
        typer.echo(f"FP16 Enabled: {fp16}")
        typer.echo(f"VLLM Enabled: {use_vllm}")
        typer.echo(f"Logging Backend: {logging_backend}")
        typer.echo(f"Max Length: {max_length}")
        typer.echo(f"Max Samples: {max_samples if max_samples else 'All'}")
        typer.echo()

        # Handle precision overlap: if fp16 is requested, disable bf16
        if fp16 and bf16:
            bf16 = False
            typer.echo("Note: Both BF16 and FP16 requested. Prioritizing FP16 (BF16 disabled).")

        # Custom imports inside command to avoid circular deps / startup costs
        import importlib.util
        import sys

        import torch
        from transformers import AutoTokenizer

        from thinkrl.algorithms.reinforce_pp import REINFORCEPPConfig
        from thinkrl.data.datasets import RLHFDataset
        from thinkrl.models.loader import get_model
        from thinkrl.training.reinforce_pp_trainer import ReinforcePPTrainer
        from thinkrl.utils.distributed_util import get_local_rank, init_distributed

        # 0. Initialize Distributed if needed
        init_distributed()
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        # 1. Load Models
        typer.echo("Loading models...")
        policy_model = get_model(
            model,
            model_type="actor",
            bf16=bf16,
            fp16=fp16,
            trust_remote_code=True,
            lora_rank=lora_r if lora_r else 0,
            lora_init_type=lora_init,
            use_flash_attention=use_flash_attention,
            device_map={"": local_rank} if torch.cuda.is_available() else None,
        )
        if gradient_checkpointing:
            policy_model.gradient_checkpointing_enable()
            typer.echo("Gradient checkpointing enabled for policy model.")

        if ref_model:
            ref_model_inst = get_model(
                ref_model,
                model_type="ref",
                bf16=bf16,
                fp16=fp16,
                trust_remote_code=True,
                lora_init_type=lora_init,
                use_flash_attention=use_flash_attention,
                device_map={"": local_rank} if torch.cuda.is_available() else None,
            )
        else:
            raise typer.Exit("Reference model is required (`--ref-model`)")

        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Load Dataset
        typer.echo(f"Loading dataset: {dataset} (source={source}, split={dataset_split}, config={dataset_config})")
        train_dataset = RLHFDataset(
            dataset_name_or_path=dataset,
            tokenizer=tokenizer,
            split=dataset_split,
            prompt_column=prompt_column,
            source=source,
            max_length=max_length,
            max_samples=max_samples,
            target_column=target_column,
            dataset_config=dataset_config,
        )

        # 3. Load Reward Function
        if reward_fn:
            if ":" in reward_fn:
                module_path, func_name = reward_fn.split(":")
            else:
                module_path, func_name = reward_fn, "reward_fn"

            try:
                spec = importlib.util.spec_from_file_location("reward_module", module_path)
                reward_module = importlib.util.module_from_spec(spec)
                sys.modules["reward_module"] = reward_module
                spec.loader.exec_module(reward_module)
                reward_func_callable = getattr(reward_module, func_name)
                typer.echo(f"Loaded reward function '{func_name}' from {module_path}")
            except Exception as e:
                typer.echo(f"Error loading reward function: {e}")
                raise typer.Exit(1) from e
        else:
            typer.echo("Warning: No reward function provided. Using dummy len-based reward.")

            def reward_func_callable(prompts, completions):
                return torch.tensor([float(len(c)) for c in completions])

        # 4. Config & Trainer
        if logging_backend == "wandb" and local_rank == 0:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    config={
                        "model": model,
                        "dataset": dataset,
                        "mode": mode,
                        "learning_rate": learning_rate,
                        "batch_size": per_device_train_batch_size,
                        "grad_accum": grad_accum,
                        "lora_r": lora_r,
                        "epochs": num_train_epochs,
                    },
                )
                typer.echo(f"W&B initialized: project={wandb_project}")
            except ImportError:
                typer.echo("Error: wandb not installed. Run 'pip install wandb'.")

        trainer = ReinforcePPTrainer(
            model=policy_model,
            ref_model=ref_model_inst,
            tokenizer=tokenizer,
            dataset=train_dataset,
            reward_fn=reward_func_callable,
            config=REINFORCEPPConfig(
                learning_rate=learning_rate,
                batch_size=per_device_train_batch_size,
                mode=mode,
                group_size=group_size,
                entropy_coeff=entropy_coeff,
                kl_loss_coeff=kl_loss_coeff,
                n_epochs=num_train_epochs,
                deepspeed=deepspeed,
                gradient_accumulation_steps=grad_accum,
            ),
            n_epochs=num_train_epochs,
            output_dir=output_dir,
            vllm_group_port=vllm_group_port,
            use_vllm=use_vllm,
        )

        # 5. Train
        typer.echo("Starting training...")
        # Calculate steps from epochs
        steps_per_epoch = len(train_dataset) // per_device_train_batch_size
        total_steps = steps_per_epoch * num_train_epochs
        trainer.train(steps=total_steps, batch_size=per_device_train_batch_size)

        # Finish W&B run
        if logging_backend == "wandb" and local_rank == 0:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                typer.echo("Warning: wandb not installed; unable to finish W&B run cleanly.")

        # 6. Save
        if output_dir and local_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            typer.echo(f"Saved model to {output_dir}")

    def _check_typer():
        """Check if typer is available."""
        if not TYPER_AVAILABLE:
            logger.error("Error: typer is required for CLI. Install with: pip install typer")
            sys.exit(1)

    @app.callback()
    def callback():
        """ThinkRL: RLHF Training Framework for Reasoning Models"""
        pass

else:
    # Fallback when typer is not available
    app = None

    def _train_fallback():
        sys.exit("Train command requires typer. Install with: pip install typer")

    def _generate_fallback():
        sys.exit("Generate command requires typer. Install with: pip install typer")


def main():
    """Main entry point."""
    if TYPER_AVAILABLE:
        app()
    else:
        sys.exit("ThinkRL CLI requires typer. Install with: pip install typer")


def reinforce_pp_entry():
    """Entry point for standalone 'reinforce-pp' command."""
    if not TYPER_AVAILABLE:
        sys.exit("ThinkRL CLI requires typer. Install with: pip install typer")

    if len(sys.argv) > 1 and sys.argv[1] == "reinforce-pp":
        pass
    else:
        sys.argv.insert(1, "reinforce-pp")
    app()


if __name__ == "__main__":
    main()
