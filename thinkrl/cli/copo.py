"""
ThinkRL COPO CLI
================

`thinkrl copo` -- Count-based Online Preference Optimization.

COPO had no CLI command at all before this: it was registered in `ALGORITHMS`,
shipped a loss, a config, a `create_copo` factory and a coin-flipping network,
and nothing in the package could run it. Its own module and its own trainer
(`training/copo_trainer.py`) follow the shape #214 established and #216-#219,
#223, #227 and #228 followed (#124).

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


LOSS_TYPES = ("sigmoid", "hinge", "ipo")

app = typer.Typer(
    name="copo",
    help="ThinkRL COPO (Count-based Online Preference Optimization) Command",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(name="copo")
def copo(
    model: Annotated[str, Option("--model", "-m", help="Model name or path")],
    dataset: Annotated[str, Option("--dataset", "-d", help="Prompt dataset name or path")],
    reward_model: Annotated[str, Option("--reward-model", help="Reward model used to score generations (required)")],
    ref_model: Annotated[
        str | None, Option("--ref-model", help="Reference model (defaults to a frozen copy of --model)")
    ] = None,
    prompt_column: Annotated[str, Option("--prompt-column", "-pc", help="Column name for prompts")] = "prompt",
    source: Annotated[str, Option("--source", "-s", help="Dataset source: 'hf', 'local', 'json', 'csv'")] = "hf",
    dataset_split: Annotated[str, Option("--dataset-split", help="Dataset split to load")] = "train",
    dataset_config: Annotated[str | None, Option("--dataset-config", help="Dataset config name")] = None,
    max_samples: Annotated[int | None, Option("--max-samples", help="Max samples to load")] = None,
    output_dir: Annotated[Path, Option("--output-dir", "-o", help="Output directory")] = Path("./copo_output"),
    alpha: Annotated[float, Option("--alpha", help="Strength of the count-based exploration bonus")] = 0.1,
    beta: Annotated[float, Option("--beta", help="Preference-loss temperature (the DPO term)")] = 0.1,
    loss_type: Annotated[str, Option("--loss-type", help="Preference loss: sigmoid, hinge, ipo")] = "sigmoid",
    learning_rate: Annotated[float, Option("--learning-rate", "--lr", help="Policy learning rate")] = 5e-7,
    iterations: Annotated[int, Option("--iterations", help="Number of COPO iterations")] = 10,
    prompts_per_iteration: Annotated[
        int, Option("--prompts-per-iteration", help="Prompts generated from per iteration")
    ] = 8,
    num_generations: Annotated[int, Option("--num-generations", "-g", help="Completions sampled per prompt")] = 4,
    max_new_tokens: Annotated[int, Option("--max-new-tokens", help="Generation budget per completion")] = 512,
    temperature: Annotated[float, Option("--temperature", help="Sampling temperature; exploration needs > 0")] = 1.0,
    batch_size: Annotated[int, Option("--batch-size", "-b", help="Policy batch size over generated pairs")] = 8,
    n_epochs: Annotated[int, Option("--n-epochs", help="Passes over the pairs built in one iteration")] = 1,
    cfn_learning_rate: Annotated[float, Option("--cfn-lr", help="Coin-flipping network learning rate")] = 1e-4,
    cfn_output_dim: Annotated[int, Option("--cfn-output-dim", help="CFN output width (the 'd' in 1/sqrt(N))")] = 20,
    cfn_buffer_size: Annotated[int, Option("--cfn-buffer-size", help="Visited-state replay buffer capacity")] = 20000,
    update_reference_every: Annotated[
        int, Option("--update-reference-every", help="Iterations between reference-model refreshes")
    ] = 1,
    lora_r: Annotated[int | None, Option("--lora-r", help="LoRA rank (enables LoRA if set)")] = None,
    bf16: Annotated[bool, Option("--bf16/--no-bf16", help="Use bfloat16 precision")] = True,
    trust_remote_code: Annotated[bool, Option("--trust-remote-code/--no-trust-remote-code")] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="Initialize and validate, but do not train")] = False,
):
    """
    Count-based Online Preference Optimization (COPO).

    Online preference learning with an exploration bonus. Each iteration samples
    several completions per prompt, scores them with a reward model, pairs the
    best against the worst and takes DPO-shaped steps over those pairs -- plus a
    bonus term that rewards visiting states the policy has seen rarely. The
    visitation count is estimated by a coin-flipping network trained to regress
    random {-1, 1} vectors, whose output magnitude falls off as 1/sqrt(N).

    Needs a reward model: COPO builds its own preference pairs, so there is no
    chosen/rejected column to read them from.

    Example:
        thinkrl copo -m meta-llama/Llama-3.2-1B -d gsm8k --reward-model OpenAssistant/reward-model-deberta-v3-base
        thinkrl copo -m Qwen/Qwen2.5-1.5B -d gsm8k --reward-model my-rm --alpha 0.2 -g 8
    """
    if loss_type not in LOSS_TYPES:
        typer.echo(f"Error: --loss-type={loss_type!r} is not a COPO preference loss.")
        typer.echo(f"Choose one of: {', '.join(LOSS_TYPES)}.")
        raise typer.Exit(1)
    if num_generations < 2:
        typer.echo(
            f"Error: --num-generations={num_generations} cannot produce a preference pair. "
            "COPO pairs the best-scoring completion against the worst, so at least 2 are needed."
        )
        raise typer.Exit(1)
    if temperature <= 0:
        typer.echo(
            f"Error: --temperature={temperature} leaves nothing to explore. COPO's bonus rewards "
            "visiting rare states, which requires sampling, not greedy decoding."
        )
        raise typer.Exit(1)

    typer.echo("=" * 60)
    typer.echo("ThinkRL Count-based Online Preference Optimization (COPO)")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model}")
    typer.echo(f"Reward model: {reward_model}")
    typer.echo(f"Dataset: {dataset} (source={source}, split={dataset_split})")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Alpha (exploration): {alpha}")
    typer.echo(f"Beta / loss type: {beta} / {loss_type}")
    typer.echo(f"Iterations: {iterations} x {prompts_per_iteration} prompts x {num_generations} generations")
    typer.echo(f"LoRA rank: {lora_r if lora_r else 'Disabled'}")
    typer.echo(f"BF16: {bf16}")
    typer.echo()

    import copy

    import torch
    from transformers import AutoTokenizer

    from thinkrl.algorithms.copo import COPOConfig
    from thinkrl.data.datasets import RLHFDataset
    from thinkrl.models.loader import get_model
    from thinkrl.training.copo_trainer import COPOTrainer, COPOTrainerConfig
    from thinkrl.utils.distributed_util import get_local_rank

    device_map = {"": get_local_rank()} if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo(f"Loading dataset: {dataset}")
    train_dataset = RLHFDataset(
        dataset_name_or_path=dataset,
        tokenizer=tokenizer,
        prompt_column=prompt_column,
        source=source,
        split=dataset_split,
        dataset_config=dataset_config,
        max_samples=max_samples,
    )

    typer.echo("Loading policy model...")
    policy_model = get_model(
        model,
        model_type="actor",
        bf16=bf16,
        lora_rank=lora_r or 0,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )

    typer.echo(f"Loading reward model: {reward_model}")
    reward_model_inst = get_model(
        reward_model,
        model_type="reward",
        bf16=bf16,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )

    if ref_model is not None:
        typer.echo(f"Loading reference model: {ref_model}")
        ref_model_inst = get_model(
            ref_model,
            model_type="ref",
            bf16=bf16,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    else:
        typer.echo("No --ref-model given, cloning --model as a frozen reference.")
        ref_model_inst = copy.deepcopy(policy_model)
        ref_model_inst.eval()
        for param in ref_model_inst.parameters():
            param.requires_grad = False

    trainer = COPOTrainer(
        model=policy_model,
        reference_model=ref_model_inst,
        reward_model=reward_model_inst,
        tokenizer=tokenizer,
        dataset=train_dataset,
        config=COPOConfig(
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta,
            loss_type=loss_type,  # type: ignore[arg-type]  # validated against LOSS_TYPES above
            num_generations_per_prompt=num_generations,
            generation_temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            n_epochs=n_epochs,
            cfn_learning_rate=cfn_learning_rate,
            cfn_output_dim=cfn_output_dim,
            cfn_buffer_size=cfn_buffer_size,
            update_reference_every=update_reference_every,
        ),
        args=COPOTrainerConfig(
            num_iterations=iterations,
            prompts_per_iteration=prompts_per_iteration,
            output_dir=str(output_dir),
        ),
    )
    # Reported after construction, not before: COPOTrainer takes the policy's real
    # hidden size over COPOConfig's 4096 default, and the number it settled on is
    # what the coin-flipping network was actually built with.
    typer.echo(f"CFN input width: {trainer.config.hidden_size}")

    if dry_run:
        typer.echo("Dry run: exiting before training.")
        raise typer.Exit(0)

    typer.echo("Starting COPO iterations...")
    trainer.train()
    trainer.save_model()
    typer.echo("Training complete.")


def main():
    app()


if __name__ == "__main__":
    main()
