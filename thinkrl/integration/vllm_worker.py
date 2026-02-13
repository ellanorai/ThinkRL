"""
ThinkRL vLLM Worker
===================

Runs a vLLM engine that can receive weight updates from a training process
via NCCL. Designed as a standalone FastAPI server.

Key design decisions:
- app.state instead of global mutable state
- argparse in main() to avoid import-time side effects
- asyncio.gather for multi-prompt generation (one engine.generate per prompt)
- run_in_executor for blocking NCCL operations (avoids blocking the event loop)
- Version-aware vLLM model access with multiple fallback paths
- Flattened parameter buffers grouped by dtype for efficient weight transfer
- /health and /metrics endpoints for observability
- /shutdown endpoint for graceful termination
- Configurable NCCL host via --nccl-host flag or VLLM_NCCL_HOST env var
"""

import argparse
import asyncio
import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from thinkrl.utils.logging import get_logger

logger = get_logger(__name__)

# Thread pool for blocking NCCL operations (single worker to serialize syncs)
_executor = ThreadPoolExecutor(max_workers=1)


def _get_vllm_model(engine: AsyncLLMEngine):
    """
    Extract the underlying model from vLLM engine.

    Tries multiple known internal access paths to handle version differences.
    Supported: vLLM >= 0.4.0, < 1.0.0
    """
    access_paths = [
        # vLLM >= 0.6.x
        lambda e: e.engine.model_executor.driver_worker.model_runner.model,
        # vLLM >= 0.5.x (alternative structure)
        lambda e: e.engine.model_executor.model_runner.model,
        # vLLM >= 0.4.x
        lambda e: e.engine.driver_worker.model_runner.model,
        # vLLM with GPU executor (some configs)
        lambda e: e.engine.model_executor.driver_worker.worker.model_runner.model,
    ]

    for accessor in access_paths:
        try:
            model = accessor(engine)
            if model is not None:
                return model
        except AttributeError:
            continue

    raise RuntimeError(
        "Could not access vLLM internal model. "
        "This may be due to an incompatible vLLM version. "
        "Supported: vLLM >= 0.4.0, < 1.0.0"
    )


def _receive_weights(app: FastAPI) -> None:
    """
    Receive weight update from trainer via NCCL (runs in background thread).

    Uses flattened parameter buffers grouped by dtype for efficient transfer.
    Must match the client's broadcast order exactly (same model architecture,
    same named_parameters() iteration order, same dtype grouping).
    """
    communicator = app.state.communicator
    nccl_stream = app.state.nccl_stream
    lock = app.state.weight_sync_lock

    if not lock.acquire(timeout=600):
        logger.error("Weight sync lock timeout - another sync may be stuck")
        return

    try:
        model = _get_vllm_model(app.state.engine)

        stream_ctx = (
            torch.cuda.stream(nccl_stream)
            if nccl_stream is not None
            else contextlib.nullcontext()
        )

        with stream_ctx:
            # Group parameters by dtype (same ordering as client)
            dtype_groups: dict[torch.dtype, list[torch.nn.Parameter]] = {}
            for _, param in model.named_parameters():
                dt = param.data.dtype
                if dt not in dtype_groups:
                    dtype_groups[dt] = []
                dtype_groups[dt].append(param)

            for dtype, params in dtype_groups.items():
                # Allocate flat receive buffer
                total_size = sum(p.data.numel() for p in params)
                flat = torch.empty(total_size, device=app.state.device, dtype=dtype)

                # Receive from trainer (rank 0)
                communicator.broadcast(flat, src=0)

                # Unflatten into model parameters
                offset = 0
                for p in params:
                    numel = p.data.numel()
                    p.data.copy_(flat[offset : offset + numel].reshape(p.data.shape))
                    offset += numel

        if nccl_stream is not None:
            nccl_stream.synchronize()

        logger.info("Weights updated successfully.")

    except Exception as e:
        logger.error(f"Weight update failed: {e}")
    finally:
        lock.release()


def create_app(args: argparse.Namespace) -> FastAPI:
    """Create the FastAPI application with the given configuration."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Starting vLLM Engine...")
        engine_args = AsyncEngineArgs(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            disable_log_requests=True,
            enforce_eager=True,  # Required for weight updates (disables CUDA graphs)
        )
        app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Initialize NCCL - we are Rank 1 in the bridge group
        nccl_host = os.environ.get("VLLM_NCCL_HOST", args.nccl_host)
        logger.info(f"Initializing NCCL Bridge (host={nccl_host}, port={args.group_port})...")
        try:
            pg = StatelessProcessGroup.create(
                host=nccl_host,
                port=args.group_port,
                rank=1,
                world_size=2,
            )
            device = torch.device(f"cuda:{args.gpu_id}")
            app.state.communicator = PyNcclCommunicator(pg, device=device)
            app.state.nccl_stream = torch.cuda.Stream(device=device)
            app.state.device = device
            logger.info("NCCL bridge initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to init NCCL: {e}")
            app.state.communicator = None
            app.state.nccl_stream = None
            app.state.device = torch.device(f"cuda:{args.gpu_id}")

        app.state.weight_sync_lock = threading.Lock()
        app.state.request_counter = 0

        yield

        # Shutdown
        logger.info("Shutting down vLLM worker...")
        _executor.shutdown(wait=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        if not hasattr(app.state, "engine") or app.state.engine is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics():
        """Basic metrics endpoint."""
        return JSONResponse({
            "requests_served": getattr(app.state, "request_counter", 0),
        })

    @app.post("/generate")
    async def generate(request: Request):
        """
        Generate completions for one or more prompts.

        Submits each prompt as a separate vLLM request and gathers results
        concurrently via asyncio.gather.
        """
        engine = app.state.engine
        data = await request.json()
        prompts = data.pop("prompts", [])

        # Build SamplingParams from remaining keys (filter None values)
        sampling_params_dict = {k: v for k, v in data.items() if v is not None}
        sampling_params = SamplingParams(**sampling_params_dict)

        # Submit each prompt as a separate request and gather concurrently
        async def _generate_single(prompt: str, req_id: str):
            final = None
            async for output in engine.generate(prompt, sampling_params, request_id=req_id):
                final = output
            return final

        tasks = [
            _generate_single(prompt, f"req_{app.state.request_counter}_{i}")
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)

        app.state.request_counter += len(prompts)

        texts = []
        token_ids_list = []
        log_probs_list = []

        for output in results:
            if output is None:
                texts.append("")
                continue

            # Assuming n=1 (first completion)
            completion = output.outputs[0]
            texts.append(completion.text)

            if completion.token_ids:
                token_ids_list.append(list(completion.token_ids))

            if completion.logprobs:
                # Extract log probability of the chosen token at each step
                seq_logprobs = []
                for step_logprobs in completion.logprobs:
                    if step_logprobs:
                        top_token = next(iter(step_logprobs))
                        seq_logprobs.append(step_logprobs[top_token].logprob)
                log_probs_list.append(seq_logprobs)

        return JSONResponse({
            "text": texts,
            "token_ids": token_ids_list,
            "log_probs": log_probs_list,
        })

    @app.post("/update_weights")
    async def update_weights():
        """
        Trigger weight update from trainer.

        Returns immediately after spawning a background thread for NCCL
        receives. The trainer then broadcasts weights, and NCCL synchronizes
        the sender and receiver internally. This avoids a deadlock where the
        client waits for the HTTP response while the server waits for NCCL data.
        """
        communicator = app.state.communicator
        if not communicator:
            return JSONResponse({"error": "NCCL not initialized"}, status_code=500)

        logger.info("Weight update triggered, spawning receive task...")

        # Launch the blocking NCCL receive in a background thread.
        # run_in_executor submits to the ThreadPoolExecutor immediately;
        # the thread starts independently of the event loop.
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_executor, _receive_weights, app)

        # Return immediately so client can proceed to broadcast
        return JSONResponse({"status": "receiving"})

    @app.post("/shutdown")
    async def shutdown():
        """Graceful shutdown endpoint."""
        logger.info("Shutdown requested")
        loop = asyncio.get_event_loop()
        loop.call_later(1.0, lambda: os._exit(0))
        return JSONResponse({"status": "shutting_down"})

    return app


def main():
    parser = argparse.ArgumentParser(description="ThinkRL vLLM Worker")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--group-port", type=int, default=51216, help="Port for NCCL bridge")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for this worker")
    parser.add_argument(
        "--nccl-host",
        type=str,
        default="127.0.0.1",
        help="Host for NCCL bridge (overridden by VLLM_NCCL_HOST env var)",
    )

    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
