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
    local_rank: Annotated[int, Option("--local_rank", "--local-rank", hidden=True)] = -1,
    use_vllm: Annotated[str, Option("--use-vllm", help="Use VLLM for generation (true/false)")] = "false",
    vllm_group_port: Annotated[int, Option("--vllm-group-port", help="NCCL group port for VLLM sync")] = 51216,
    vllm_url: Annotated[
        str, Option("--vllm-url", help="Address of the vLLM worker started with 'thinkrl-vllm-worker'")
    ] = "http://localhost:8000",
    vllm_sync_world_size: Annotated[
        int, Option("--vllm-sync-world-size", help="Processes participating in the vLLM weight sync")
    ] = 2,
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
    system_prompt: Annotated[
        str | None, Option("--system-prompt", help="System prompt to prepend to each input")
    ] = "You are a helpful arithmetic reasoning assistant. Your task is to solve the given math problem. You must think step-by-step and write out your complete train of thought inside <think></think> tags. After you have finished thinking, you must provide your final numerical answer enclosed exactly inside <answer></answer> tags.",
    chat_template: Annotated[
        bool,
        Option(
            "--chat-template/--no-chat-template",
            help="Render prompts with the tokenizer's chat template (instruct models). "
            "Disable for base models or datasets that are already formatted.",
        ),
    ] = True,
    seed: Annotated[int, Option("--seed", help="Random seed for reproducibility")] = 42,
    trust_remote_code: Annotated[
        bool,
        Option(
            "--trust-remote-code/--no-trust-remote-code",
            help="Allow executing custom model code downloaded from the Hub",
        ),
    ] = False,
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
    # These three used to be parsed and then ignored. A flag that is silently dropped is
    # worse than one that does not exist, because the run looks configured: --grad-accum
    # was even recorded into the W&B config, so a run was logged as though accumulation
    # had been applied when nothing read it. See #79.
    if grad_accum != 1:
        typer.echo(
            f"Error: --grad-accum={grad_accum} is not supported. GRPOTrainer.train steps the "
            "optimizer once per rollout and no accumulation is implemented, so the value "
            "would be silently ignored. Use --batch-size to change the effective batch.",
            err=True,
        )
        raise typer.Exit(1)

    if deepspeed is not None:
        typer.echo(
            f"Error: --deepspeed={deepspeed} is not supported. No RL trainer has a distributed "
            "path yet, so the config would be read and dropped; see issue #128. "
            "DeepSpeed currently applies to SFT only.",
            err=True,
        )
        raise typer.Exit(1)

    known_backends = ("tensorboard", "wandb", "none")
    if logging_backend not in known_backends:
        typer.echo(
            f"Error: --logging-backend={logging_backend!r} is not one of {known_backends}.",
            err=True,
        )
        raise typer.Exit(1)

    from thinkrl.utils import set_seed

    set_seed(seed)
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
    typer.echo(f"System Prompt: {system_prompt}")
    typer.echo()

    if fp16 and bf16:
        bf16 = False
        typer.echo("Note: Both BF16 and FP16 requested. Prioritizing FP16 (BF16 disabled).")

    if not ref_model:
        typer.echo("Error: a reference model is required (`--ref-model`).", err=True)
        raise typer.Exit(code=1)

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
        trust_remote_code=trust_remote_code,
        lora_rank=lora_r if lora_r else 0,
        lora_init_type=lora_init,
        use_flash_attention=use_flash_attention,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    if gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        typer.echo("Gradient checkpointing enabled for policy model.")

    ref_model_inst = get_model(
        ref_model,
        model_type="ref",
        bf16=bf16,
        fp16=fp16,
        trust_remote_code=trust_remote_code,
        lora_init_type=lora_init,
        use_flash_attention=use_flash_attention,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

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
        system_prompt=system_prompt,
        apply_chat_template=chat_template,
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

    metric_logger = None
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
                    "lora_r": lora_r,
                    "epochs": num_train_epochs,
                },
            )
            typer.echo(f"W&B initialized: project={wandb_project}")
        except ImportError:
            typer.echo("Error: wandb not installed. Run 'pip install wandb'.")
    elif logging_backend == "tensorboard" and local_rank == 0:
        # The default value, which logged nothing at all until now, despite
        # TensorBoardLogger existing and being covered by its own tests.
        #
        # Tolerated rather than fatal, matching the wandb branch above: tensorboard is an
        # optional dependency and this is the *default* backend, so raising here would
        # break every run on a machine that simply does not have it installed.
        try:
            from thinkrl.logging.tensorboard import TensorBoardLogger

            metric_logger = TensorBoardLogger(log_dir=f"{output_dir}/tensorboard")
            metric_logger.log_hyperparams(
                {
                    "model": model,
                    "dataset": dataset,
                    "learning_rate": learning_rate,
                    "batch_size": per_device_train_batch_size,
                    "lora_r": lora_r,
                    "epochs": num_train_epochs,
                }
            )
            typer.echo(f"TensorBoard logging to {output_dir}/tensorboard")
        except ImportError:
            typer.echo("Warning: tensorboard not installed, metrics will not be logged. pip install tensorboard")

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
        use_vllm=(str(use_vllm).lower() == "true"),
        vllm_group_port=vllm_group_port,
        vllm_url=vllm_url,
        vllm_sync_world_size=vllm_sync_world_size,
    )

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting training loop...")
    per_device_train_batch_size = int(per_device_train_batch_size)
    dataset_len = int(len(train_dataset))
    total_steps = num_train_epochs * dataset_len // per_device_train_batch_size
    if total_steps == 0:
        typer.echo(
            f"Error: {dataset_len} samples over {num_train_epochs} epoch(s) at batch size "
            f"{per_device_train_batch_size} gives 0 training steps. Lower --batch-size or "
            f"raise --max-samples.",
            err=True,
        )
        raise typer.Exit(code=1)
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
