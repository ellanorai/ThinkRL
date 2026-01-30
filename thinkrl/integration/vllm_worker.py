"""
ThinkRL VLLM Worker
===================
Runs a VLLM engine that can receive weight updates from a training process via NCCL.
"""
import argparse
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
import uvicorn
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from thinkrl.utils.logging import get_logger


logger = get_logger(__name__)

# Global Engine
engine: Optional[AsyncLLMEngine] = None
communicator: Optional[PyNcclCommunicator] = None

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model path")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--group-port", type=int, default=51216)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
parser.add_argument("--dtype", type=str, default="auto")

args, _ = parser.parse_known_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting VLLM Engine...")
    engine_args = AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        disable_log_requests=True,
        enforce_eager=True,  # Required for weight updates often
    )
    global engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Initialize NCCL
    # We are Rank 1 in the bridge group
    logger.info("Initializing NCCL Bridge...")
    try:
        pg = StatelessProcessGroup.create(
            host="127.0.0.1",  # Localhost for now, flexible later
            port=args.group_port,
            rank=1,
            world_size=2,
        )
        global communicator
        communicator = PyNcclCommunicator(pg, device=torch.device("cuda:0"))
    except Exception as e:
        logger.error(f"Failed to init NCCL: {e}")

    yield
    # Shutdown
    if communicator:
        pass  # Cleanup if needed


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: Request):
    """Generate completion."""
    data = await request.json()
    prompts = data.get("prompts", [])
    sampling_params_dict = {k: v for k, v in data.items() if k not in ["prompts"]}

    sampling_params = SamplingParams(**sampling_params_dict)

    results_generator = engine.generate(prompts, sampling_params, request_id=f"req_{id(data)}")

    # We wait for the final result
    final_output = []
    async for request_output in results_generator:
        final_output.append(request_output)

    # Process output similar to vllm.entrypoints.api_server
    # Simply returning text and logprobs
    texts = []
    token_ids = []
    log_probs = []

    for output in final_output:
        # Assuming n=1
        generated_text = output.outputs[0].text
        texts.append(generated_text)

        if output.outputs[0].token_ids:
            token_ids.append(output.outputs[0].token_ids)

        if output.outputs[0].logprobs:
            # VLLM structure: list of dicts {token_id: logprob}
            # We want just the logprob of the chosen token
            seq_logprobs = []
            for step_logprobs in output.outputs[0].logprobs:
                # Top-1
                if step_logprobs:
                    top_token = list(step_logprobs.keys())[0]
                    seq_logprobs.append(step_logprobs[top_token].logprob)
            log_probs.append(seq_logprobs)

    return JSONResponse({"text": texts, "token_ids": token_ids, "log_probs": log_probs})


@app.post("/update_weights")
async def update_weights():
    """Trigger weight update from trainer."""
    if not communicator:
        return JSONResponse({"error": "NCCL not initialized"}, status_code=500)

    logger.info("Receiving new weights...")

    # We need to perform the NCCL recv on the model parameters
    # This must match the order/structure of the sender
    # We assume the engine model is available via engine.engine.model_executor.driver_worker.model_runner.model
    # This acts on the *local* GPU model.

    # Accessing the underlying model in vLLM is tricky and depends on version.
    # For single-GPU vLLM, it's straightforward.
    try:
        # HACK: Access internal model. This is fragile.
        # vLLM > 0.4.0 structure
        model_executor = engine.engine.model_executor
        model_runner = model_executor.driver_worker.model_runner
        model = model_runner.model

        for _, param in model.named_parameters():
            communicator.broadcast(param.data, src=0)

        logger.info("Weights updated successfully.")
        return JSONResponse({"status": "ok"})

    except Exception as e:
        logger.error(f"Update failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
