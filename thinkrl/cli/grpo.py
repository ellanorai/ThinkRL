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
    name="grpo",
    help="ThinkRL GRPO Training Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="grpo")
def grpo(
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
    ref_model: Annotated[
        str | None, Option("--ref-model", "-r", help="Reference model name or path (required)")
    ] = None,
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./grpo_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    kl_coeff: Annotated[float, Option("--kl-coeff", help="KL penalty coefficient")] = 0.04,
    group_size: Annotated[int, Option("--group-size", "-g", help="Group size")] = 64,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    lora_init: Annotated[
        str, Option("--lora-init", help="LoRA init type: 'default', 'garbage', 'pissa', 'pissa_niter_[n]'")
    ] = "default",
    grad_accum: Annotated[int, Option("--grad-accum", "-ga", help="Gradient accumulation steps")] = 1,
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    fp16: Annotated[bool, Option("--fp16/--no-fp16", help="Use float16 precision")] = False,
    use_flash_attention: Annotated[bool, Option("--flash-attn/--no-flash-attn", help="Use Flash Attention 2")] = False,
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
    ] = "thinkrl-grpo",
    max_length: Annotated[int, Option("--max-length", help="Maximum sequence length")] = 512,
    max_samples: Annotated[
        int | None, Option("--max-samples", help="Maximum number of samples to load from dataset")
    ] = None,
    target_column: Annotated[str, Option("--target-column", help="Column name for target answers")] = "answer",
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
):
    """
    Group Relative Policy Optimization (GRPO).

    Critic-free RL algorithm using group-relative advantages.
    Similar to DeepSeek-R1's training approach.

    Example:
        thinkrl grpo --model meta-llama/Llama-3.1-8B --dataset math_dataset --group-size 64
        grpo --model meta-llama/Llama-3.1-8B --dataset math_dataset --group-size 64
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL Group Relative Policy Optimization (GRPO)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Ref Model: {ref_model}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Dataset Source: {source}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"DeepSpeed: {deepspeed}")
    typer.echo(f"Group Size: {group_size}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"LoRA Rank: {lora_r}")
    typer.echo(f"LoRA Init: {lora_init}")
    typer.echo(f"KL Coeff: {kl_coeff}")
    typer.echo(f"Epochs: {num_train_epochs}")
    typer.echo(f"Batch size: {per_device_train_batch_size}")
    typer.echo(f"BF16 Enabled: {bf16}")
    typer.echo(f"FP16 Enabled: {fp16}")
    typer.echo(f"VLLM Enabled: {use_vllm}")
    typer.echo(f"Logging Backend: {logging_backend}")
    typer.echo(f"Max Length: {max_length}")
    typer.echo(f"Max Samples: {max_samples if max_samples else 'All'}")
    typer.echo()

    if fp16 and bf16:
        bf16 = False
        typer.echo("Note: Both BF16 and FP16 requested. Prioritizing FP16 (BF16 disabled).")

    from thinkrl.algorithms.grpo import GRPOConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.grpo_trainer import GRPOTrainer
    from thinkrl.utils.distributed_util import get_local_rank, init_distributed

    init_distributed()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

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

        def reward_func_callable(prompts, completions, **kwargs):
            return torch.tensor([float(len(c)) for c in completions])

    if logging_backend == "wandb" and local_rank == 0:
        try:
            import wandb

            wandb.init(
                project=wandb_project,
                config={
                    "model": model,
                    "dataset": dataset,
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

    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model_inst,
        tokenizer=tokenizer,
        dataset=train_dataset,
        reward_fn=reward_func_callable,
        config=GRPOConfig(
            learning_rate=learning_rate,
            group_size=group_size,
            beta=kl_coeff,
            n_epochs=num_train_epochs,
        ),
        use_vllm=use_vllm,
        vllm_group_port=vllm_group_port,
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting training loop...")
    per_device_train_batch_size = int(per_device_train_batch_size)
    dataset_len = int(len(train_dataset))
    total_steps = num_train_epochs * dataset_len // per_device_train_batch_size
    trainer.train(steps=total_steps, batch_size=per_device_train_batch_size)

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
