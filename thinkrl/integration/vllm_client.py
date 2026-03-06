"""
ThinkRL vLLM Client
===================

Client for interacting with vLLM server with distributed training support.
Handles Multi-GPU (DDP/FSDP) training environments correctly.

Weight synchronization uses NCCL broadcast with flattened parameter buffers
for efficient transfer. The protocol is:

1. Client sends HTTP POST to /update_weights (server returns immediately,
   spawns a background thread for NCCL receives)
2. Client broadcasts flattened weight buffers grouped by dtype
3. NCCL internally synchronizes sender/receiver

This avoids the deadlock where the client waits for an HTTP response while
the server waits for NCCL data.
"""

import contextlib
import importlib.util
import os
from typing import Any

import requests
import torch
import torch.distributed as dist

from thinkrl.utils.logging import get_logger

logger = get_logger(__name__)

try:
    if importlib.util.find_spec("vllm"):
        pass  # Imports verified inside methods to be safe
except (ImportError, ValueError) as exc:
    logger.debug("vLLM module detection failed; treating 'vllm' as unavailable: %s", exc)

# Default HTTP timeouts (seconds)
_DEFAULT_GENERATE_TIMEOUT = 300  # 5 min for generation (large batches)
_DEFAULT_GENERATE_TIMEOUT = 300  # 5 min for generation (large batches)
_DEFAULT_CONTROL_TIMEOUT = 30  # 30s for control endpoints


class VLLMClient:
    def __init__(
        self,
        url: str = "http://localhost:8000",
        group_port: int = 51216,
        sync_world_size: int = 2,
        generate_timeout: float = _DEFAULT_GENERATE_TIMEOUT,
        control_timeout: float = _DEFAULT_CONTROL_TIMEOUT,
    ):
        self.url = url
        self.group_port = group_port
        self.communicator = None
        self._nccl_stream: torch.cuda.Stream | None = None

        # Bridge topology: Rank 0 = Client (Trainer), Rank 1 = vLLM Server
        self.client_rank = 0
        self.server_rank = 1
        self.sync_world_size = sync_world_size

        # HTTP timeouts
        self.generate_timeout = generate_timeout
        self.control_timeout = control_timeout

        # Distributed training info
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.is_main_process = True
        self.rank = 0

        if self.is_distributed:
            self.rank = dist.get_rank()
            # Only Rank 0 of the training cluster should talk to vLLM
            self.is_main_process = self.rank == 0

    def health_check(self) -> bool:
        """Check if vLLM server is reachable and healthy."""
        if not self.is_main_process:
            return True  # Non-main processes don't interact with server

        try:
            resp = requests.get(f"{self.url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate(
        self,
        prompts: list[str],
        params: dict[str, Any],
        return_logprobs: bool = True,
    ) -> dict[str, Any]:
        """
        Generate completions via HTTP request.

        Args:
            prompts: List of prompt strings
            params: Generation parameters (max_tokens, temperature, etc.)
            return_logprobs: If True, request and return token log probabilities

        Returns:
            Dict containing:
                - "text": List of generated text strings
                - "token_ids": List of token ID lists (if return_logprobs=True)
                - "log_probs": List of log probability lists (if return_logprobs=True)

        Note:
            When return_logprobs=True, this avoids the need for a separate forward
            pass to compute old_log_probs, effectively doubling training throughput.
        """
        if not self.is_main_process:
            return {"text": [], "token_ids": [], "log_probs": []}

        try:
            request_params = {**params}
            if return_logprobs:
                request_params["logprobs"] = 1
                request_params["prompt_logprobs"] = None

            resp = requests.post(
                f"{self.url}/generate",
                json={
                    "prompts": prompts,
                    **request_params,
                },
                timeout=self.generate_timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            output = {"text": result.get("text", [])}

            if return_logprobs and "token_ids" in result:
                output["token_ids"] = result["token_ids"]
                output["log_probs"] = result["log_probs"]
            elif return_logprobs:
                logger.warning(
                    "vLLM server did not return log_probs. "
                    "Ensure server supports logprobs=1 parameter. "
                    "Falling back to separate forward pass."
                )
                output["token_ids"] = []
                output["log_probs"] = []

            return output

        except requests.RequestException as e:
            logger.error(f"Generation request failed: {e}")
            raise RuntimeError(f"vLLM Generation failed: {e}") from e

    def init_weight_sync(self, device: torch.device) -> None:
        """
        Initialize the NCCL bridge for weight synchronization.
        Only the main training process initializes the communicator.

        Creates a dedicated CUDA stream for NCCL operations to avoid
        interfering with computation on the default stream.
        """
        if not self.is_main_process:
            return

        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup
        except ImportError as e:
            raise ImportError(
                "vLLM is required for weight sync. Install with `pip install vllm`."
            ) from e

        logger.info(f"Initializing NCCL bridge on device {device} (Client Rank {self.client_rank})")

        host = os.environ.get("VLLM_NCCL_HOST", "127.0.0.1")

        try:
            pg = StatelessProcessGroup.create(
                host=host, port=self.group_port, rank=self.client_rank, world_size=self.sync_world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
        except Exception as e:
            logger.error(f"Failed to initialize NCCL bridge: {e}")
            raise RuntimeError(f"NCCL init failed. Ensure you are on Linux and 'pynccl' is working: {e}") from e

        self.communicator = PyNcclCommunicator(pg, device=device)

        # Dedicated CUDA stream for NCCL weight transfer
        if isinstance(device, torch.device) and device.type == "cuda":
            self._nccl_stream = torch.cuda.Stream(device=device)
        elif isinstance(device, str) and "cuda" in device:
            self._nccl_stream = torch.cuda.Stream(device=torch.device(device))

        logger.info("NCCL bridge initialized successfully.")

    def check_weights(self, model: torch.nn.Module) -> None:
        """
        Verifies that the local model structure matches the remote vLLM model.
        """
        if not self.is_main_process:
            return

        logger.info("Verifying model structure with vLLM server...")
        params = {n: list(p.shape) for n, p in model.named_parameters()}

        try:
            resp = requests.post(
                f"{self.url}/check_weights",
                json={"params": params},
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") != "ok":
                details = result.get("details", [])
                logger.error(f"Model structure mismatch! vLLM sync will fail. details={details}")
                raise RuntimeError(f"vLLM model mismatch: {details[:3]}...")

            logger.info("Model structure verification passed.")

        except requests.RequestException as e:
            logger.error(f"Failed to verify weights: {e}")
            # We might want to warn instead of crash if the server doesn't support this endpoint yet
            logger.warning("Could not verify weights (server might be old). Proceeding with caution.")

    def update_model_weights(self, model: torch.nn.Module) -> None:
        """
        Broadcast model weights to the vLLM server.
        Handles FSDP/DDP contexts by ensuring we operate on the main process.

        Protocol:
        1. Send HTTP POST to /update_weights. The server returns immediately
           after spawning a background thread that enters the NCCL receive loop.
        2. Broadcast flattened weight buffers grouped by dtype. NCCL internally
           synchronizes: the sender-side broadcast blocks until the receiver
           has also called broadcast, so no explicit handshake is needed.
        """
        if not self.is_main_process:
            return

        if self.communicator is None:
            logger.error("Attempted to update weights without initialized communicator.")
            raise RuntimeError("Communicator not initialized. Call init_weight_sync() first.")

        # 1. Trigger server to enter receive mode.
        #    The server endpoint returns immediately after spawning a background
        #    NCCL receive task, breaking the deadlock where both sides wait.
        try:
            resp = requests.post(
                f"{self.url}/update_weights",
                timeout=self.control_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to trigger weight update on server: {e}")
            raise RuntimeError(f"Weight update trigger failed: {e}") from e

        # 2. Broadcast weights using flattened buffers
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp_ctx = contextlib.nullcontext()
        if isinstance(model, FSDP):
            fsdp_ctx = FSDP.summon_full_params(model, writeback=False, rank0_only=True)

        stream_ctx = (
            torch.cuda.stream(self._nccl_stream)
            if self._nccl_stream is not None
            else contextlib.nullcontext()
        )

        with fsdp_ctx, stream_ctx:
            self._broadcast_weights_flat(model)

        # Synchronize NCCL stream before returning
        if self._nccl_stream is not None:
            self._nccl_stream.synchronize()

    def _broadcast_weights_flat(self, model: torch.nn.Module) -> None:
        """
        Broadcast all model weights using flattened per-dtype buffers.

        Groups parameters by dtype (preserving iteration order), flattens each
        group into a single contiguous tensor, and broadcasts once per group.
        This is significantly faster than per-parameter broadcast for models
        with hundreds of parameters.

        Both client and server must iterate model.named_parameters() in the same
        order, which is guaranteed for the same model architecture.
        """
        # Group parameters by dtype, preserving insertion order
        dtype_groups: dict[torch.dtype, list[torch.Tensor]] = {}
        for _, param in model.named_parameters():
            dt = param.data.dtype
            if dt not in dtype_groups:
                dtype_groups[dt] = []
            dtype_groups[dt].append(param.data)

        for _, params in dtype_groups.items():
            flat = torch.cat([p.reshape(-1) for p in params])
            self.communicator.broadcast(flat, src=self.client_rank)

    def shutdown_server(self) -> None:
        """Send a graceful shutdown signal to the vLLM server."""
        if not self.is_main_process:
            return
        try:
            requests.post(f"{self.url}/shutdown", timeout=5)
        except requests.RequestException:
            pass  # Server may already be down
