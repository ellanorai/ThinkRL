"""
ThinkRL vLLM Client (Improved)
==============================
Robust client for interacting with vLLM server with dynamic distributed support.
Handles Multi-GPU (DDP/FSDP) training environments correctly.
"""
import requests
import torch
import torch.distributed as dist
from typing import Any

from thinkrl.utils.logging import get_logger

try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
except ImportError:
    raise ImportError("vLLM is required. Install with `pip install vllm`.")

logger = get_logger(__name__)

class VLLMClient:
    def __init__(self, url: str = "http://localhost:8000", group_port: int = 51216):
        self.url = url
        self.group_port = group_port
        self.communicator = None
        
        # We define the "Bridge" topology strictly
        # Rank 0 = This Client (Trainer)
        # Rank 1 = The vLLM Server
        self.client_rank = 0 
        self.server_rank = 1
        self.sync_world_size = 2
        
        # Check if we are in a distributed training setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.is_main_process = True
        
        if self.is_distributed:
            # Only Rank 0 of the training cluster should talk to vLLM
            self.is_main_process = (dist.get_rank() == 0)

    def generate(self, prompts: list[str], params: dict[str, Any]) -> list[str]:
        """Generates completions via HTTP request."""
        # In a multi-GPU setup, usually only main process triggers generation, 
        # or prompts are scattered. Assuming centralized request for now.
        if not self.is_main_process:
            return []

        try:
            resp = requests.post(f"{self.url}/generate", json={
                "prompts": prompts,
                **params
            })
            resp.raise_for_status()
            return resp.json()["text"]
        except requests.RequestException as e:
            logger.error(f"Generation request failed: {e}")
            raise RuntimeError(f"vLLM Generation failed: {e}")

    def init_weight_sync(self, device: torch.device) -> None:
        """
        Initializes the NCCL bridge.
        Only the main training process initializes the communicator.
        """
        if not self.is_main_process:
            return

        logger.info(f"Initializing NCCL bridge on device {device} (Client Rank {self.client_rank})")

        # Create the Stateless Group for the Bridge (Size = 2)
        # This isolates the training traffic from the inference sync traffic
        pg = StatelessProcessGroup.create(
            host="127.0.0.1",
            port=self.group_port,
            rank=self.client_rank, 
            world_size=self.sync_world_size
        )
        
        self.communicator = PyNcclCommunicator(pg, device=device)
        logger.info("NCCL bridge initialized successfully.")

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

        # OPTIONAL: If using FSDP, you might need a context manager here 
        # to gather full state dict to rank 0 before broadcasting.
        # e.g., with FSDP.summons_params(model, writeback=False): ...
        
        # logger.debug("Broadcasting model weights to vLLM server...")
        for name, param in model.named_parameters():
            # We strictly broadcast from Client (0) to Server (1)
            # using the 'src' rank relative to the bridge group.
            self.communicator.broadcast(param.data, src=self.client_rank)