"""
ThinkRL DPO CLI
===============

`thinkrl dpo` -- Direct Preference Optimization.

Lives in its own module alongside its own trainer (`training/dpo_trainer.py`),
the shape #214 established and #216-#219 and #223 followed, rather than inline
in `cli/main.py` (#124).

Author: EllanorAI
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Annotated


try:
    import typer
    from typer import Option
except ImportError:
    sys.exit("Error: typer is required for CLI. Install with: pip install typer")


app = typer.Typer(
    name="dpo",
    help="ThinkRL DPO (Direct Preference Optimization) Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="dpo")
def dpo(
    model: Annotated[str, Option("--model", "-m", help="Model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Preference dataset name or path")],
    ref_model: Annotated[
        str | None, Option("--ref-model", help="Reference model (defaults to a frozen copy of --model)")
    ] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    chosen_column: Annotated[str, Option("--chosen-column", help="Column name for the preferred response")] = "chosen",
    rejected_column: Annotated[
        str, Option("--rejected-column", help="Column name for the dispreferred response")
    ] = "rejected",
    source: Annotated[str, Option("--source", "-s", help="Dataset source: 'hf', 'local', 'json', 'csv'")] = "hf",
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name")] = None,
    max_samples: Annotated[int | None, Option("--max-samples", help="Max samples to load")] = None,
    max_length: Annotated[int, Option("--max-length", help="Max sequence length")] = 512,
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./dpo_output"),
    beta: Annotated[float, Option("--beta", help="DPO beta (KL temperature) parameter")] = 0.1,
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
    loss_type: Annotated[str, Option("--loss-type", help="Loss type: sigmoid, hinge, ipo")] = "sigmoid",
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    trust_remote_code: Annotated[bool, Option("--trust-remote-code/--no-trust-remote-code")] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
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
    typer.echo(f"Dataset: {dataset} (source={source}, split={dataset_split})")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Beta: {beta}")
    typer.echo(f"Loss type: {loss_type}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"Epochs: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    import copy

    import torch
    from transformers import AutoTokenizer

    from thinkrl.algorithms.dpo import DPOAlgorithm, DPOConfig
    from thinkrl.data.datasets import PreferenceDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.dpo_trainer import DPOTrainer, DPOTrainerConfig
    from thinkrl.utils.distributed_util import get_local_rank

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo(f"Loading dataset: {dataset}")
    train_dataset = PreferenceDataset(
        dataset_name_or_path=dataset,
        tokenizer=tokenizer,
        prompt_column=prompt_column,
        chosen_column=chosen_column,
        rejected_column=rejected_column,
        source=source,
        split=dataset_split,
        dataset_config=dataset_config,
        max_length=max_length,
        max_samples=max_samples,
    )

    typer.echo("Loading policy model...")
    policy_model = get_model(
        model,
        model_type="actor",
        bf16=bf16,
        lora_rank=lora_r or 0,
        trust_remote_code=trust_remote_code,
        device_map={"": get_local_rank()} if torch.cuda.is_available() else None,
    )

    if ref_model is not None:
        typer.echo(f"Loading reference model: {ref_model}")
        ref_model_inst = get_model(
            ref_model,
            model_type="ref",
            bf16=bf16,
            trust_remote_code=trust_remote_code,
            device_map={"": get_local_rank()} if torch.cuda.is_available() else None,
        )
    else:
        typer.echo("No --ref-model given, cloning --model as a frozen reference.")
        ref_model_inst = copy.deepcopy(policy_model)
        ref_model_inst.eval()
        for param in ref_model_inst.parameters():
            param.requires_grad = False

    algorithm = DPOAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model_inst,
        config=DPOConfig(
            learning_rate=learning_rate,
            beta=beta,
            loss_type=loss_type,
        ),
    )

    trainer = DPOTrainer(
        algorithm=algorithm,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=DPOTrainerConfig(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=str(output_dir),
        ),
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting training loop...")
    trainer.train()
    trainer.save_model()
    typer.echo("Training complete.")


def main():
    app()


if __name__ == "__main__":
    main()
