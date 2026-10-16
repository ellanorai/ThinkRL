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
    name="vapo",
    help="ThinkRL VAPO Training Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="vapo")
def vapo(
    model: Annotated[str, Option("--model", "-m", help="Policy model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
    reward_model: Annotated[
        str | None, Option("--reward-model", "-r", help="Reward model name or path, used to score completions")
    ] = None,
    value_model: Annotated[
        str | None,
        Option("--value-model", help="Critic model name or path. Defaults to a critic head over --model"),
    ] = None,
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name")] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    source: Annotated[
        str, Option("--source", "-s", help="Dataset source: 'hf' (HuggingFace), 'local', 'json', 'csv'")
    ] = "hf",
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./vapo_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Policy learning rate")] = 1e-6,
    value_lr: Annotated[float | None, Option("--value-lr", help="Critic learning rate")] = 2e-6,
    epsilon_low: Annotated[float, Option("--epsilon-low", help="Lower clip range")] = 0.2,
    epsilon_high: Annotated[float, Option("--epsilon-high", help="Upper clip range (Clip-Higher)")] = 0.28,
    adaptive_gae_alpha: Annotated[
        float, Option("--adaptive-gae-alpha", help="Length-adaptive GAE lambda: lambda = 1 - 1/(alpha * length)")
    ] = 0.05,
    lm_loss_coeff: Annotated[
        float, Option("--lm-loss-coeff", help="Weight of the auxiliary NLL loss on correct samples")
    ] = 0.1,
    rollouts_per_prompt: Annotated[
        int, Option("--rollouts-per-prompt", "-g", help="Completions sampled per prompt")
    ] = 1,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="VAPO epochs per rollout")] = 2,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="VAPO mini-batch size")] = 64,
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
    Value-model-based Augmented Proximal Policy Optimization (VAPO).

    Actor-critic RL with decoupled GAE and length-adaptive lambda, built for long
    chain-of-thought reasoning tasks, scored by a reward model.

    Example:
        thinkrl vapo --model gpt2 --dataset Anthropic/hh-rlhf --reward-model OpenAssistant/reward-model-deberta-v3-large-v2
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL Value-model-based Augmented PPO (VAPO)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Reward Model: {reward_model if reward_model else 'none (dummy len-based reward)'}")
    typer.echo(f"Value Model: {value_model if value_model else f'critic head over {model}'}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"Clip range: [{epsilon_low}, {epsilon_high}]")
    typer.echo(f"Adaptive GAE alpha: {adaptive_gae_alpha}")
    typer.echo(f"Rollouts per prompt: {rollouts_per_prompt}")
    typer.echo(f"Epochs per rollout: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    from thinkrl.utils import set_seed

    set_seed(seed)

    from transformers import GenerationConfig

    from thinkrl.algorithms.vapo import VAPOConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.vapo_trainer import VAPOTrainer

    typer.echo("Loading models...")
    policy_model = get_model(
        model, model_type="actor", bf16=bf16, trust_remote_code=trust_remote_code, use_flash_attention=False
    )
    value_model_inst = get_model(
        value_model or model,
        model_type="critic",
        bf16=bf16,
        trust_remote_code=trust_remote_code,
        use_flash_attention=False,
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

    if reward_model:
        reward_model_inst = get_model(reward_model, model_type="reward", bf16=bf16, trust_remote_code=trust_remote_code)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model, trust_remote_code=trust_remote_code)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token

        @torch.no_grad()
        def reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> torch.Tensor:
            reward_model_inst.eval()
            texts = [p + c for p, c in zip(prompts, completions, strict=True)]
            encoded = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            return reward_model_inst(**encoded)

    else:
        typer.echo("Warning: No --reward-model provided. Using dummy len-based reward.")

        def reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> torch.Tensor:
            return torch.tensor([float(len(c)) for c in completions])

    trainer = VAPOTrainer(
        model=policy_model,
        value_model=value_model_inst,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_fn=reward_fn,
        config=VAPOConfig(
            learning_rate=learning_rate,
            value_lr=value_lr,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            adaptive_gae_alpha=adaptive_gae_alpha,
            lm_loss_coeff=lm_loss_coeff,
            n_epochs=num_train_epochs,
            batch_size=per_device_train_batch_size,
        ),
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=rollouts_per_prompt,
        ),
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting training loop...")
    dataset_len = len(train_dataset)
    total_steps = dataset_len // max(1, per_device_train_batch_size // max(1, rollouts_per_prompt))
    if total_steps == 0:
        typer.echo(
            f"Error: {dataset_len} prompts at batch size {per_device_train_batch_size} "
            f"(rollouts-per-prompt={rollouts_per_prompt}) gives 0 training steps.",
            err=True,
        )
        raise typer.Exit(code=1)
    trainer.train(steps=total_steps, batch_size=max(1, per_device_train_batch_size // max(1, rollouts_per_prompt)))

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
