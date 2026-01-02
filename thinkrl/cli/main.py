"""
ThinkRL CLI Main
================

Command-line interface for ThinkRL RLHF training.

Commands:
    train     - Run RLHF training
    generate  - Generate rollouts
    merge     - Merge LoRA adapters
    info      - Show configuration info

Author: EllanorAI
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

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
        print("Error: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)


# Create app if typer is available
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="thinkrl",
        help="ThinkRL: RLHF Training Framework for Reasoning Models",
        add_completion=False,
    )

    @app.command()
    def train(
        config: Annotated[
            Path,
            Option("--config", "-c", help="Path to config YAML/JSON file"),
        ],
        overrides: Annotated[
            Optional[list[str]],
            Option("--override", "-o", help="Config overrides (key=value)"),
        ] = None,
        resume: Annotated[
            Optional[Path],
            Option("--resume", "-r", help="Path to checkpoint to resume from"),
        ] = None,
        dry_run: Annotated[
            bool,
            Option("--dry-run", help="Validate config without running"),
        ] = False,
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
        model: Annotated[
            str,
            Option("--model", "-m", help="Model name or path"),
        ],
        prompts: Annotated[
            Path,
            Option("--prompts", "-p", help="Path to prompts JSONL file"),
        ],
        output: Annotated[
            Path,
            Option("--output", "-o", help="Output path for generations"),
        ] = Path("outputs/generations.jsonl"),
        max_new_tokens: Annotated[
            int,
            Option("--max-new-tokens", help="Maximum tokens to generate"),
        ] = 512,
        temperature: Annotated[
            float,
            Option("--temperature", "-t", help="Sampling temperature"),
        ] = 0.7,
        batch_size: Annotated[
            int,
            Option("--batch-size", "-b", help="Batch size"),
        ] = 8,
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
        base_model: Annotated[
            str,
            Option("--base-model", help="Base model name or path"),
        ],
        adapter: Annotated[
            Path,
            Option("--adapter", "-a", help="Path to LoRA adapter"),
        ],
        output: Annotated[
            Path,
            Option("--output", "-o", help="Output path for merged model"),
        ],
        push_to_hub: Annotated[
            Optional[str],
            Option("--push-to-hub", help="Push merged model to HuggingFace Hub"),
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
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
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
            raise typer.Exit(1)

    @app.command()
    def info(
        config: Annotated[
            Optional[Path],
            Option("--config", "-c", help="Show config file info"),
        ] = None,
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

    @app.callback()
    def callback():
        """ThinkRL: RLHF Training Framework for Reasoning Models"""
        pass

else:
    # Fallback when typer is not available
    app = None

    def _train_fallback():
        print("Train command requires typer. Install with: pip install typer")

    def _generate_fallback():
        print("Generate command requires typer. Install with: pip install typer")


def main():
    """Main entry point."""
    if TYPER_AVAILABLE:
        app()
    else:
        print("ThinkRL CLI requires typer. Install with: pip install typer")
        sys.exit(1)


if __name__ == "__main__":
    main()
