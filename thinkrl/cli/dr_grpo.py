from pathlib import Path
import sys
from typing import Annotated, cast

import torch
from transformers import AutoTokenizer


try:
    import typer
    from typer import Option
except ImportError:
    sys.exit("Error: typer is required for CLI. Install with: pip install typer")


app = typer.Typer(
    name="dr-grpo",
    help="ThinkRL Dr.GRPO Training Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="dr-grpo")
def dr_grpo(
    model: Annotated[str, Option("--model", "-m", help="Policy model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
    ref_model: Annotated[
        str | None, Option("--ref-model", "-r", help="Reference model name or path. Only used if --kl-coeff > 0")
    ] = None,
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name")] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    source: Annotated[
        str, Option("--source", "-s", help="Dataset source: 'hf' (HuggingFace), 'local', 'json', 'csv'")
    ] = "hf",
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./dr_grpo_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    kl_coeff: Annotated[
        float, Option("--kl-coeff", help="KL penalty coefficient (0 disables it; requires --ref-model if > 0)")
    ] = 0.0,
    epsilon: Annotated[float, Option("--clip-range", "--epsilon", help="Clip range")] = 0.2,
    group_size: Annotated[int, Option("--group-size", "-g", help="Completions sampled per prompt")] = 4,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="Epochs per rollout")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Prompts per rollout")] = 4,
    max_new_tokens: Annotated[int, Option("--max-new-tokens", help="Generation budget per prompt")] = 256,
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    trust_remote_code: Annotated[
        bool, Option("--trust-remote-code/--no-trust-remote-code", help="Allow executing custom model code")
    ] = False,
    max_length: Annotated[int, Option("--max-length", help="Maximum prompt length")] = 512,
    max_samples: Annotated[int | None, Option("--max-samples", help="Maximum number of samples to load")] = None,
    seed: Annotated[int, Option("--seed", help="Random seed for reproducibility")] = 42,
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
):
    """
    Dr.GRPO.

    GRPO with group mean-centered advantages and no std normalization, removing
    the difficulty-level bias GRPO's local normalization introduces.

    Example:
        thinkrl dr-grpo --model gpt2 --dataset Anthropic/hh-rlhf --group-size 4
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL Dr.GRPO")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Ref Model: {ref_model if ref_model else 'none (KL disabled)' if kl_coeff == 0 else 'required'}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"Clip range: {epsilon}")
    typer.echo(f"KL coeff: {kl_coeff}")
    typer.echo(f"Group size: {group_size}")
    typer.echo(f"Epochs per rollout: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    if kl_coeff > 0 and not ref_model:
        typer.echo("Error: --kl-coeff > 0 requires a reference model (`--ref-model`).", err=True)
        raise typer.Exit(code=1)

    from thinkrl.utils import set_seed

    set_seed(seed)

    from transformers import GenerationConfig

    from thinkrl.algorithms.dr_grpo import DrGRPOConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.dr_grpo_trainer import DrGRPOTrainer

    typer.echo("Loading models...")
    policy_model = get_model(
        model, model_type="actor", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False
    )
    ref_model_inst = (
        get_model(ref_model, model_type="ref", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False)
        if ref_model
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo(f"Loading dataset: {dataset} (source={source}, split={dataset_split})")
    train_dataset = RLHFDataset(
        dataset_name_or_path=dataset,
        tokenizer=tokenizer,
        split=dataset_split,
        prompt_column=prompt_column,
        source=source,
        max_length=max_length,
        max_samples=max_samples,
        dataset_config=dataset_config,
    )

    typer.echo("Warning: No reward model support yet for dr-grpo. Using dummy len-based reward.")

    def reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> torch.Tensor:
        return torch.tensor([float(len(c)) for c in completions])

    trainer = DrGRPOTrainer(
        model=policy_model,
        ref_model=ref_model_inst,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_fn=reward_fn,
        config=DrGRPOConfig(
            learning_rate=learning_rate,
            kl_coeff=kl_coeff,
            epsilon=epsilon,
            group_size=group_size,
            n_epochs=num_train_epochs,
        ),
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=group_size,
        ),
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting training loop...")
    dataset_len = len(train_dataset)
    total_steps = dataset_len // max(1, per_device_train_batch_size)
    if total_steps == 0:
        typer.echo(
            f"Error: {dataset_len} prompts at batch size {per_device_train_batch_size} gives 0 training steps.",
            err=True,
        )
        raise typer.Exit(code=1)
    trainer.train(steps=total_steps, batch_size=per_device_train_batch_size)

    typer.echo("Training complete.")

    from thinkrl.models.actor import Actor

    output_dir.mkdir(parents=True, exist_ok=True)
    cast(Actor, policy_model).save_pretrained(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Model saved to {output_dir}")


def main():
    app()


if __name__ == "__main__":
    main()
