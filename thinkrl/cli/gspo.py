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
    name="gspo",
    help="ThinkRL GSPO Training Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="gspo")
def gspo(
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
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./gspo_output"),
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Learning rate")] = 1e-6,
    beta: Annotated[
        float,
        Option("--beta", help="KL penalty coefficient. GSPO's paper default is 0 (no explicit KL term)."),
    ] = 0.0,
    epsilon_low: Annotated[
        float, Option("--epsilon-low", help="Lower sequence-ratio clip bound (paper default 3e-4)")
    ] = 3e-4,
    epsilon_high: Annotated[
        float, Option("--epsilon-high", help="Upper sequence-ratio clip bound (paper default 4e-4)")
    ] = 4e-4,
    group_size: Annotated[int, Option("--group-size", "-g", help="Group size")] = 64,
    num_train_epochs: Annotated[int, Option("--num-train-epochs", "--epochs", help="Number of training epochs")] = 1,
    per_device_train_batch_size: Annotated[int, Option("--batch-size", "-b", help="Per-device batch size")] = 4,
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    lora_init: Annotated[
        str, Option("--lora-init", help="LoRA init type: 'default', 'garbage', 'pissa', 'pissa_niter_[n]'")
    ] = "default",
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    fp16: Annotated[bool, Option("--fp16/--no-fp16", help="Use float16 precision")] = False,
    use_flash_attention: Annotated[bool, Option("--flash-attn/--no-flash-attn", help="Use Flash Attention 2")] = False,
    remote_rm_url: Annotated[
        str | None,
        Option(
            "--remote-rm-url",
            help="Score completions with a reward model server instead of a local reward function",
        ),
    ] = None,
    reward_fn: Annotated[
        str | None, Option("--reward-fn", help="Path to reward function (module.py:func_name)")
    ] = None,
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
    ] = "thinkrl-gspo",
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
    Group Sequence Policy Optimization (GSPO).

    GRPO with a sequence-level, length-normalized importance ratio in place of
    GRPO's token-level one -- the algorithm behind Qwen3's RL training.

    Example:
        thinkrl gspo --model meta-llama/Llama-3.1-8B --ref-model meta-llama/Llama-3.1-8B --dataset math_dataset --group-size 8
    """
    typer.echo("=" * 60)
    typer.echo("ThinkRL Group Sequence Policy Optimization (GSPO)")
    typer.echo("=" * 60)

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
    typer.echo(f"Group Size: {group_size}")
    typer.echo(f"Learning rate: {learning_rate}")
    typer.echo(f"LoRA Rank: {lora_r}")
    typer.echo(f"LoRA Init: {lora_init}")
    typer.echo(f"Beta: {beta}")
    typer.echo(f"Epsilon low/high: {epsilon_low}/{epsilon_high}")
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

    if not ref_model:
        typer.echo("Error: a reference model is required (`--ref-model`).", err=True)
        raise typer.Exit(code=1)

    from thinkrl.algorithms.gspo import GSPOAlgorithm, GSPOConfig
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
        # get_model() is typed to return nn.Module; mypy resolves attributes it does
        # not statically declare (gradient_checkpointing_enable, save_pretrained)
        # through Module's own __getattr__ stub, which returns Tensor | Module and
        # is therefore "not callable". Every model get_model() can actually return
        # (Actor/Critic/RewardModel) defines this method for real.
        policy_model.gradient_checkpointing_enable()  # type: ignore[operator]
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

    if remote_rm_url:
        from thinkrl.rewards import RemoteRewardScorer

        reward_func_callable = RemoteRewardScorer(remote_urls=remote_rm_url)
        typer.echo(f"Scoring with remote reward model at {remote_rm_url}")
    elif reward_fn:
        if ":" in reward_fn:
            module_path, func_name = reward_fn.split(":")
        else:
            module_path, func_name = reward_fn, "reward_fn"

        try:
            spec = importlib.util.spec_from_file_location("reward_module", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load a module spec from {module_path}")
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

    algorithm = GSPOAlgorithm(
        policy_model=policy_model,
        ref_model=ref_model_inst,
        config=GSPOConfig(
            learning_rate=learning_rate,
            group_size=group_size,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            n_epochs=num_train_epochs,
        ),
    )

    trainer = GRPOTrainer(
        algorithm=algorithm,
        # GRPOTrainer's own type hint (tokenizer: PreTrainedTokenizer) is narrower than
        # what AutoTokenizer.from_pretrained is actually typed to return in this
        # transformers version; grpo_trainer.py is itself in pyproject.toml's mypy
        # debt list, so the mismatch only surfaces at this (non-exempt) call site.
        tokenizer=tokenizer,  # type: ignore[arg-type]
        dataset=train_dataset,
        reward_fn=reward_func_callable,
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

    if output_dir and local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        policy_model.save_pretrained(output_dir)  # type: ignore[operator]
        tokenizer.save_pretrained(output_dir)
        typer.echo(f"Model saved to {output_dir}")


def main():
    app()


if __name__ == "__main__":
    main()
