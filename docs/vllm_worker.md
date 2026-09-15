# vLLM worker

ThinkRL offloads rollout generation to vLLM through a standalone server rather than an
in-process engine. `thinkrl/generation/vllm_engine.py` is an empty placeholder; the working
path is the worker plus client described here.

## Shape

- `thinkrl/integration/vllm_worker.py` holds the vLLM engine behind a FastAPI server.
- `thinkrl/integration/vllm_client.py` is what the trainers use: it submits prompts and
  pushes updated policy weights to the worker over NCCL between rollouts.
- `GRPOTrainer` and `ReinforcePPTrainer` take `use_vllm=True`, `vllm_group_port`,
  `vllm_url` and `vllm_sync_world_size`.

Keeping the engine out of process means the training job and the generation job do not
compete for the same CUDA context, and weights move over NCCL rather than through a
serialization round trip.

## Starting a worker

A worker must already be running before any training command that passes `--use-vllm`.
Nothing starts one for you, and the client will simply fail to connect.

```bash
thinkrl-vllm-worker \
    --model HuggingFaceTB/SmolLM2-135M \
    --host 127.0.0.1 \
    --port 8000 \
    --group-port 51216
```

`python -m thinkrl.integration.vllm_worker` with the same arguments is equivalent.

Then from the CLI:

```bash
thinkrl grpo \
    --model HuggingFaceTB/SmolLM2-135M \
    --ref-model HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --use-vllm true \
    --vllm-url http://127.0.0.1:8000 \
    --vllm-group-port 51216
```

or from Python:

```python
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    reward_fn=reward_fn,
    use_vllm=True,
    vllm_url="http://127.0.0.1:8000",
    vllm_group_port=51216,
    vllm_sync_world_size=2,
)
```

`--vllm-url` matters for anything other than a worker on the same host: the client
defaulted to localhost and the trainer never forwarded an alternative, so a remote worker
was unreachable regardless of what the client supported.

## Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check; the client polls this before submitting work |
| `/generate` | POST | Generate completions for a batch of prompts |
| `/update_weights` | POST | Pull updated policy weights over the NCCL group |
| `/metrics` | GET | Engine counters |
| `/shutdown` | POST | Terminate the worker |

## Security

**The worker has no authentication, and `--host` defaults to `0.0.0.0`,** which binds every
interface. `/shutdown` and `/update_weights` are both unauthenticated, so anyone who can
reach the port can stop your run or replace the weights the policy generates from.

Bind it to loopback (`--host 127.0.0.1`) unless the training process is on another machine,
and when it must be reachable, put it behind a firewall rule that admits only the training
host. Do not expose it to a shared network or the internet.

## Requirements

vLLM is an optional dependency and is not installed by default:

```bash
pip install vllm
```

It needs a CUDA GPU. There is no macOS build, so the worker cannot be run on Apple silicon.
