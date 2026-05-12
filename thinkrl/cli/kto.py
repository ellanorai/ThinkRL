import importlib.util
import os
from pathlib import Path
import sys
from typing import Annotated

import torch
from transformers import AutoTokenizer


try:
    import typer
    from typer import Option
except ImportError:
    sys.exit("Error: typer is required for CLI. Install with: pip install typer")


app = typer.Typer(
    name="kto",
    help="ThinkRL KTO (Kahneman-Tversky Optimization) Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="kto")
def kto_cmd(
    model: Annotated[str, Option("--model", "-m", help="Model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[
        str | None, Option("--dataset-config", help="Dataset config name (e.g., 'main' for gsm8k)")
    ] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    source: Annotated[
        str, Option("--source", "-s", help="Dataset source: 'hf' (HuggingFace), 'local', 'json', 'csv'")
    ] = "hf",
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./kto_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    beta: Annotated[float, Option("--beta", help="KTO Beta coefficient")] = 0.1,
    lambda_d: Annotated[float, Option("--lambda-d", help="Weight for desirable generations")] = 1.0,
    lambda_u: Annotated[float, Option("--lambda-u", help="Weight for undesirable generations")] = 1.0,
    reward_threshold: Annotated[float, Option("--reward-threshold", help="Threshold to binarize rewards")] = 0.5,
    entropy_coef: Annotated[float, Option("--entropy-coef", help="Entropy regularization coefficient")] = 0.0,
    batch_size: Annotated[int, Option("--batch-size", help="Batch size")] = 4,
    steps: Annotated[int, Option("--steps", help="Number of training steps")] = 1000,
    n_epochs: Annotated[int, Option("--n-epochs", help="Number of epochs per rollout")] = 1,
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    lora_init: Annotated[
        str, Option("--lora-init", help="LoRA init type: 'default', 'garbage', 'pissa', 'pissa_niter_[n]'")
    ] = "default",
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    fp16: Annotated[bool, Option("--fp16/--no-fp16", help="Use float16 precision")] = False,
    flash_attention: Annotated[
        bool, Option("--flash-attention/--no-flash-attention", help="Use Flash Attention 2")
    ] = True,
    use_vllm: Annotated[bool, Option("--vllm/--no-vllm", help="Use vLLM for generation")] = False,
    reward_fn: Annotated[
        str | None, Option("--reward-fn", help="Path to reward function (module.py:func_name)")
    ] = None,
    target_column: Annotated[str, Option("--target-column", help="Column name for target answers")] = "answer",
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
):
    """
    Kahneman-Tversky Optimization (KTO).

    Online version of KTO that uses a reward function to binarize generated
    completions into desirable and undesirable sets, then optimizes using
    the KTO loss function.

    Example:
        thinkrl kto --model meta-llama/Llama-3.1-8B --dataset gsm8k --beta 0.1
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL KTO (Kahneman-Tversky Optimization)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Beta: {beta}")
    typer.echo(f"Steps: {steps}")
    typer.echo()

    from thinkrl.algorithms.kto import KTOConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.kto_trainer import KTOTrainer
    from thinkrl.utils.distributed_util import get_local_rank, init_distributed

    init_distributed()
    local_rank = get_local_rank()

    typer.echo("Loading models...")
    if fp16:
        bf16 = False
        
    policy_model = get_model(
        model,
        model_type="actor",
        bf16=bf16,
        fp16=fp16,
        use_flash_attention=flash_attention,
        trust_remote_code=True,
        lora_rank=lora_r if lora_r else 0,
        lora_init_type=lora_init,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    
    ref_model = get_model(
        model,
        model_type="ref",
        bf16=bf16,
        fp16=fp16,
        use_flash_attention=flash_attention,
        trust_remote_code=True,
        lora_rank=0,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo(f"Loading dataset: {dataset}")
    train_dataset = RLHFDataset(
        dataset_name_or_path=dataset,
        tokenizer=tokenizer,
        split=dataset_split,
        prompt_column=prompt_column,
        source=source,
        target_column=target_column,
        dataset_config=dataset_config,
    )

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
        typer.echo("Warning: No reward function provided. Using dummy equality-based reward.")

        def reward_func_callable(prompts, completions, targets=None, **kwargs):
            if not targets:
                return torch.zeros(len(completions))
            # Rough equality check
            return torch.tensor([1.0 if t.strip() in c else 0.0 for c, t in zip(completions, targets)])

    config = KTOConfig(
        learning_rate=learning_rate,
        beta=beta,
        lambda_d=lambda_d,
        lambda_u=lambda_u,
        n_epochs=n_epochs,
        use_vllm=use_vllm,
        entropy_coef=entropy_coef,
    )

    trainer = KTOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_fn=reward_func_callable,
        config=config,
        reward_threshold=reward_threshold,
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting KTO loop...")
    trainer.train(steps=steps, batch_size=batch_size)

    typer.echo("Training complete.")

    # Save model
    if output_dir and local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        typer.echo(f"Model saved to {output_dir}")


def main():
    app()


if __name__ == "__main__":
    main()
