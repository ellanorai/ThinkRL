"""
ThinkRL vLLM Client (Improved)
==============================
Robust client for interacting with vLLM server with dynamic distributed support.
Handles Multi-GPU (DDP/FSDP) training environments correctly.
"""
import contextlib
import os
from typing import Any

import requests
import torch
import torch.distributed as dist

from thinkrl.utils.logging import get_logger


try:
    import vllm
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
except ImportError:
    # We only raise this if the user *tries* to instantiate the client.
    # But for module level, we can just set a flag or let it fail later.
    # However, since the user explicitly wants "Linux only" and "Fix it",
    # we should probably just let it be a hard dependency for this module.
    # If this module is imported, vllm should be there.
    # But to be safe for the rest of the codebase (e.g. strict type checking imports), we keep it safe.
    vllm = None


logger = get_logger(__name__)


class VLLMClient:
    def __init__(self, url: str = "http://localhost:8000", group_port: int = 51216, sync_world_size: int = 2):
        self.url = url
        self.group_port = group_port
        self.communicator = None

        # We define the "Bridge" topology strictly
        # Rank 0 = This Client (Trainer)
        # Rank 1 = The vLLM Server
        self.client_rank = 0
        self.server_rank = 1
        self.sync_world_size = sync_world_size

        # Check if we are in a distributed training setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.is_main_process = True
        self.rank = 0

        if self.is_distributed:
            self.rank = dist.get_rank()
            # Only Rank 0 of the training cluster should talk to vLLM
            self.is_main_process = self.rank == 0

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
            # Request log_probs from vLLM to avoid double forward pass
            request_params = {**params}
            if return_logprobs:
                # vLLM parameter: logprobs=N returns top N logprobs. We usually just need the performed action.
                # However, vLLM returns a list of dicts. We set N=1 for efficiency.
                request_params["logprobs"] = 1
                request_params["prompt_logprobs"] = None  # Don't need prompt logprobs usually

            resp = requests.post(
                f"{self.url}/generate",
                json={
                    "prompts": prompts,
                    **request_params,
                },
            )
            resp.raise_for_status()
            result = resp.json()

            output = {"text": result.get("text", [])}

            if return_logprobs and "token_ids" in result:
                output["token_ids"] = result["token_ids"]
                output["log_probs"] = result["log_probs"]
            elif return_logprobs:
                # Fallback: server didn't return logprobs, caller will need to compute
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
        Initializes the NCCL bridge.
        Only the main training process initializes the communicator.
        """
        if not self.is_main_process:
            return

        if vllm is None:
            raise ImportError("vLLM is not installed. Please install vllm to use VLLMClient.")

        logger.info(f"Initializing NCCL bridge on device {device} (Client Rank {self.client_rank})")

        # Get server IP from URL if needed, usually passed separately or env var
        # For now assume localhost or env var config for PyNccl
        # PyNcclCommunicator mostly relies on 'host' being resolvable

        # Create the Stateless Group for the Bridge
        # This isolates the training traffic from the inference sync traffic
        # Note: vLLM worker uses `socket.gethostbyname(socket.gethostname())` usually
        host = os.environ.get("VLLM_NCCL_HOST", "127.0.0.1")

        try:
            pg = StatelessProcessGroup.create(
                host=host, port=self.group_port, rank=self.client_rank, world_size=self.sync_world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
        except Exception as e:
            logger.error(f"Failed to initialize NCCL bridge: {e}")
            raise RuntimeError(f"NCCL init failed. Ensure you are on Linux and 'pynccl' is working: {e}") from e

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
        Broadcasts model weights to the vLLM server.
        Handles FSDP/DDP contexts by ensuring we operate on the main process.
        """
        if not self.is_main_process:
            return

        if self.communicator is None:
            logger.error("Attempted to update weights without initialized communicator.")
            raise RuntimeError("Communicator not initialized. Call init_weight_sync() first.")

        # 1. Trigger Server Sync
        # We MUST tell the server to enter the recv loop.
        try:
            resp = requests.post(f"{self.url}/update_weights")
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to trigger weight update on server: {e}")
            raise RuntimeError(f"Weight update trigger failed: {e}") from e

        # 2. Broadcast Weights
        # logger.debug("Broadcasting model weights to vLLM server...")

        # Handle FSDP: Gather full params on Rank 0
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        context = contextlib.nullcontext()
        if isinstance(model, FSDP):
            # Writeback=False because we just want to read for broadcast, not persist the gathering
            context = FSDP.summon_full_params(model, writeback=False, rank0_only=True)

        with context:
            # Flatten parameters for efficient single-tensor broadcast
            # This requires the server to know the structure to unflatten,
            # OR we broadcast layer-by-layer.
            # Layer-by-layer is slower but robust to architecture mismatches if handled by name.
            # Flattening is much faster. Let's do flattening if possible, but for stability now,
            # we will iterate. *Optimization: Bucket buffers.*

            # Simple iteration for v1 safety
            for _, param in model.named_parameters():
                # We strictly broadcast from Client (0) to Server (1)
                # using the 'src' rank relative to the bridge group.
                self.communicator.broadcast(param.data, src=self.client_rank)

        return
