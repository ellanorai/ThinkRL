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
    name="prime",
    help="ThinkRL PRIME Training Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="prime")
def prime(
    model: Annotated[str, Option("--model", "-m", help="Policy model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
    ref_model: Annotated[
        str | None, Option("--ref-model", "-r", help="Reference model name or path. Clones --model if omitted")
    ] = None,
    prm_model: Annotated[
        str | None,
        Option("--prm-model", help="Implicit PRM model name or path. Clones --ref-model if omitted"),
    ] = None,
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name")] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    source: Annotated[
        str, Option("--source", "-s", help="Dataset source: 'hf' (HuggingFace), 'local', 'json', 'csv'")
    ] = "hf",
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./prime_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Policy learning rate")] = 1e-6,
    prm_learning_rate: Annotated[float, Option("--prm-lr", help="Implicit PRM learning rate")] = 1e-6,
    beta: Annotated[float, Option("--beta", help="Implicit reward scaling coefficient")] = 0.05,
    clip_epsilon: Annotated[float, Option("--clip-epsilon", "--epsilon", help="PPO clip range")] = 0.2,
    num_generations_per_prompt: Annotated[
        int, Option("--num-generations-per-prompt", "-k", help="K completions sampled per prompt (RLOO baseline)")
    ] = 8,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="PPO epochs per rollout")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Prompts per rollout")] = 8,
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
    Process Reinforcement through Implicit Rewards (PRIME).

    Learns an online Implicit PRM from outcome labels and uses it alongside an
    RLOO baseline to give the PPO policy step dense, token-level feedback.

    Example:
        thinkrl prime --model gpt2 --dataset gsm8k --num-generations-per-prompt 8
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL Process Reinforcement through Implicit Rewards (PRIME)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Ref Model: {ref_model if ref_model else f'clone of {model}'}")
    typer.echo(f"PRM Model: {prm_model if prm_model else 'clone of ref model'}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"PRM learning rate: {prm_learning_rate}")
    typer.echo(f"Beta: {beta}")
    typer.echo(f"Clip epsilon: {clip_epsilon}")
    typer.echo(f"Generations per prompt (K): {num_generations_per_prompt}")
    typer.echo(f"Epochs per rollout: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    from thinkrl.utils import set_seed

    set_seed(seed)

    from transformers import GenerationConfig

    from thinkrl.algorithms.prime import PRIMEConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.prime_trainer import PRIMETrainer

    typer.echo("Loading models...")
    policy_model = get_model(
        model, model_type="actor", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False
    )
    ref_model_inst = (
        get_model(ref_model, model_type="ref", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False)
        if ref_model
        else None
    )
    prm_model_inst = (
        get_model(prm_model, model_type="ref", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False)
        if prm_model
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

    typer.echo(
        "Note: PRIME scores completions with outcome labels from the dataset's reward "
        "column, not a reward model. Using a dummy len-threshold outcome label."
    )

    def reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> torch.Tensor:
        return torch.tensor([1.0 if len(c) > 20 else 0.0 for c in completions])

    trainer = PRIMETrainer(
        model=policy_model,
        ref_model=ref_model_inst,
        prm_model=prm_model_inst,
        learning_rate=learning_rate,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_fn=reward_fn,
        config=PRIMEConfig(
            prm_learning_rate=prm_learning_rate,
            beta=beta,
            clip_epsilon=clip_epsilon,
            num_generations_per_prompt=num_generations_per_prompt,
            n_epochs=num_train_epochs,
            batch_size=per_device_train_batch_size,
        ),
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_generations_per_prompt,
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
