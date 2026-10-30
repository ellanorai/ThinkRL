"""
ThinkRL IPO CLI
===============

`thinkrl ipo` -- Identity Preference Optimization.

IPO had no CLI command at all before this: it was registered in
`ALGORITHMS`, shipped a loss, a config and a `create_ipo` factory, and the only
way to run it was to construct `DPOTrainer(algorithm=IPOAlgorithm(...))` by
hand. Its own module and its own trainer (`training/ipo_trainer.py`) follow the
shape #214 established and #216-#219, #223 and #227 followed (#124).

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


LOSS_TYPES = ("ipo", "ipo_hinge")

app = typer.Typer(
    name="ipo",
    help="ThinkRL IPO (Identity Preference Optimization) Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="ipo")
def ipo(
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
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./ipo_output"),
    tau: Annotated[float, Option("--tau", help="IPO regularization strength; the loss targets 1/(2*tau)")] = 0.05,
    beta: Annotated[float, Option("--beta", help="Scale for the reported chosen/rejected reward metrics")] = 0.1,
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
    loss_type: Annotated[str, Option("--loss-type", help="Loss type: ipo, ipo_hinge")] = "ipo",
    length_normalization: Annotated[
        bool,
        Option(
            "--length-normalization/--no-length-normalization",
            help="Divide sequence log-probabilities by completion length",
        ),
    ] = False,
    gradient_accumulation_steps: Annotated[
        int, Option("--grad-accum", help="Steps to accumulate before an optimizer step")
    ] = 1,
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    trust_remote_code: Annotated[bool, Option("--trust-remote-code/--no-trust-remote-code")] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
):
    """
    Identity Preference Optimization (IPO).

    Offline preference training on chosen/rejected pairs, like DPO, but with a
    squared loss that drives the policy/reference log-ratio margin to 1/(2*tau)
    instead of pushing it as far apart as the data allows. That bounded target
    is what makes IPO less prone to the deterministic-policy collapse DPO shows
    on noisy or near-tied preferences.

    Example:
        thinkrl ipo --model meta-llama/Llama-3.1-8B --dataset Anthropic/hh-rlhf --tau 0.05
        thinkrl ipo -m meta-llama/Llama-3.2-1B -d argilla/dpo-mix-7k --loss-type ipo_hinge --length-normalization
    """
    if loss_type not in LOSS_TYPES:
        typer.echo(f"Error: --loss-type={loss_type!r} is not an IPO loss. Choose one of: {', '.join(LOSS_TYPES)}.")
        typer.echo("DPO's sigmoid and hinge losses live behind `thinkrl dpo --loss-type`.")
        raise typer.Exit(1)
    if tau <= 0:
        typer.echo(f"Error: --tau={tau} must be positive; the IPO target is 1/(2*tau).")
        raise typer.Exit(1)

    typer.echo("=" * 60)
    typer.echo("ThinkRL Identity Preference Optimization (IPO)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Dataset: {dataset} (source={source}, split={dataset_split})")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Tau: {tau} (loss target 1/(2*tau) = {1.0 / (2.0 * tau):.4g})")
    typer.echo(f"Loss type: {loss_type}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"Epochs: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"Length normalization: {length_normalization}")
    typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    if not length_normalization:
        typer.echo(
            "Warning: IPO sums log-probabilities over the completion, so a chosen response "
            "that is simply longer than the rejected one shifts the margin on its own. "
            "Pass --length-normalization unless your pairs are of comparable length."
        )

    import copy

    import torch
    from transformers import AutoTokenizer

    from thinkrl.algorithms.ipo import IPOConfig
    from thinkrl.data.datasets import PreferenceDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.dpo_trainer import DPOTrainerConfig
    from thinkrl.training.ipo_trainer import IPOTrainer
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

    trainer = IPOTrainer(
        model=policy_model,
        ref_model=ref_model_inst,
        config=IPOConfig(
            learning_rate=learning_rate,
            beta=beta,
            tau=tau,
            loss_type=loss_type,  # type: ignore[arg-type]  # validated against LOSS_TYPES above
            length_normalization=length_normalization,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
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
