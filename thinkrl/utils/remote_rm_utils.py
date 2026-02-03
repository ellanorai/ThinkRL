"""
ThinkRL Remote Reward Model Utilities
======================================

Utilities for distributed reward model evaluation.
Aligned with OpenRLHF patterns for remote RM serving.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import os
import time
from typing import Any


logger = logging.getLogger(__name__)


# Optional imports
try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    requests = None

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False
    ray = None


def request_api_wrapper(
    url: str,
    data: dict[str, Any],
    try_max_times: int = 5,
    timeout: int = 180,
    retry_delay: float = 1.0,
) -> dict[str, Any] | None:
    """
    Execute HTTP POST request with retry logic.

    Args:
        url: API endpoint URL
        data: JSON data to send
        try_max_times: Maximum retry attempts
        timeout: Request timeout in seconds
        retry_delay: Delay between retries in seconds

    Returns:
        JSON response or None if all retries fail

    Example:
        ```python
        result = request_api_wrapper(
            "http://localhost:8000/reward",
            {"query": "text to evaluate"},
        )
        if result is not None:
            reward = result["reward"]
        ```
    """
    if not _REQUESTS_AVAILABLE:
        raise RuntimeError("requests library is required. Install with: pip install requests")

    headers = {"Content-Type": "application/json"}

    for attempt in range(try_max_times):
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{try_max_times}): {e}")
            if attempt < try_max_times - 1:
                time.sleep(retry_delay)

    logger.error(f"API request failed after {try_max_times} attempts")
    return None


if _RAY_AVAILABLE:

    @ray.remote
    def remote_rm_fn_ray(
        api_url: str,
        queries: list[str],
        prompts: list[str] | None = None,
        labels: list[Any] | None = None,
    ) -> list[float] | None:
        """
        Ray remote function for reward model API calls.

        Args:
            api_url: Reward model API endpoint
            queries: List of queries to evaluate
            prompts: Optional prompts
            labels: Optional labels

        Returns:
            List of reward scores or None on failure
        """
        data = {"queries": queries}
        if prompts is not None:
            data["prompts"] = prompts
        if labels is not None:
            data["labels"] = labels

        result = request_api_wrapper(api_url, data)
        if result is None:
            return None

        return result.get("rewards", result.get("scores", []))


class RemoteRewardModel:
    """
    Client for distributed reward model evaluation.

    Supports both remote API servers and custom reward functions.

    Example:
        ```python
        # Using remote API
        rm = RemoteRewardModel(
            remote_urls=["http://localhost:8000", "http://localhost:8001"],
        )
        rewards = rm.get_rewards(queries=["text1", "text2"])

        # Using custom reward function
        rm = RemoteRewardModel(
            reward_fn_path="./rewards/custom_reward.py",
            reward_fn_name="compute_reward",
        )
        rewards = rm.get_rewards(queries=["text1", "text2"])
        ```
    """

    def __init__(
        self,
        remote_urls: list[str] | str | None = None,
        reward_fn_path: str | None = None,
        reward_fn_name: str = "compute_reward",
        micro_batch_size: int = 64,
        use_ray: bool = True,
    ):
        """
        Initialize the remote reward model client.

        Args:
            remote_urls: URL(s) of remote reward model server(s)
            reward_fn_path: Path to Python file with custom reward function
            reward_fn_name: Name of reward function in the file
            micro_batch_size: Batch size for processing
            use_ray: Whether to use Ray for distributed calls
        """
        # Handle remote URLs
        if isinstance(remote_urls, str):
            self.remote_urls = [remote_urls]
        else:
            self.remote_urls = remote_urls or []

        self.micro_batch_size = micro_batch_size
        self.use_ray = use_ray and _RAY_AVAILABLE

        # Load custom reward function if specified
        self.reward_fn: Callable | None = None
        if reward_fn_path is not None:
            self.reward_fn = self._load_reward_fn(reward_fn_path, reward_fn_name)

        if not self.remote_urls and self.reward_fn is None:
            logger.warning("No remote URLs or reward function specified. " "RemoteRewardModel will return zeros.")

    def _load_reward_fn(
        self,
        module_path: str,
        fn_name: str,
    ) -> Callable:
        """Load reward function from Python file."""
        import importlib.util

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Reward function module not found: {module_path}")

        spec = importlib.util.spec_from_file_location("reward_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, fn_name):
            raise AttributeError(f"Function '{fn_name}' not found in {module_path}")

        fn = getattr(module, fn_name)
        logger.info(f"Loaded reward function '{fn_name}' from {module_path}")
        return fn

    def get_rewards(
        self,
        queries: list[str],
        prompts: list[str] | None = None,
        labels: list[Any] | None = None,
    ) -> list[float]:
        """
        Get reward scores for queries.

        Args:
            queries: List of texts to evaluate
            prompts: Optional prompts (for prompt-conditioned rewards)
            labels: Optional labels (for supervised rewards)

        Returns:
            List of reward scores

        Example:
            ```python
            rm = RemoteRewardModel(remote_urls=["http://localhost:8000"])
            rewards = rm.get_rewards(
                queries=["Response 1", "Response 2"],
                prompts=["Question 1", "Question 2"],
            )
            ```
        """
        if not queries:
            return []

        # Use custom reward function if available
        if self.reward_fn is not None:
            return self._get_rewards_from_fn(queries, prompts, labels)

        # Use remote servers if available
        if self.remote_urls:
            return self._get_rewards_from_servers(queries, prompts, labels)

        # Fallback: return zeros
        logger.warning("No reward source available, returning zeros")
        return [0.0] * len(queries)

    def _get_rewards_from_fn(
        self,
        queries: list[str],
        prompts: list[str] | None,
        labels: list[Any] | None,
    ) -> list[float]:
        """Get rewards using custom function."""
        all_rewards = []

        # Process in micro-batches
        for i in range(0, len(queries), self.micro_batch_size):
            batch_queries = queries[i : i + self.micro_batch_size]
            batch_prompts = prompts[i : i + self.micro_batch_size] if prompts else None
            batch_labels = labels[i : i + self.micro_batch_size] if labels else None

            if self.use_ray:
                # Wrap function call with Ray
                @ray.remote
                def compute_rewards(fn, q, p, l):
                    return fn(q, prompts=p, labels=l)

                future = compute_rewards.remote(self.reward_fn, batch_queries, batch_prompts, batch_labels)
                rewards = ray.get(future)
            else:
                rewards = self.reward_fn(
                    batch_queries,
                    prompts=batch_prompts,
                    labels=batch_labels,
                )

            all_rewards.extend(rewards)

        return all_rewards

    def _get_rewards_from_servers(
        self,
        queries: list[str],
        prompts: list[str] | None,
        labels: list[Any] | None,
    ) -> list[float]:
        """Get rewards from remote servers."""
        num_servers = len(self.remote_urls)

        if self.use_ray:
            # Distribute across servers using Ray
            futures = []
            batch_size = len(queries) // num_servers + 1

            for i, url in enumerate(self.remote_urls):
                start = i * batch_size
                end = min(start + batch_size, len(queries))

                if start >= len(queries):
                    break

                batch_queries = queries[start:end]
                batch_prompts = prompts[start:end] if prompts else None
                batch_labels = labels[start:end] if labels else None

                future = remote_rm_fn_ray.remote(url, batch_queries, batch_prompts, batch_labels)
                futures.append(future)

            results = ray.get(futures)

            all_rewards = []
            for result in results:
                if result is not None:
                    all_rewards.extend(result)
                else:
                    # Handle failed requests
                    all_rewards.extend([0.0] * batch_size)

            return all_rewards[: len(queries)]

        else:
            # Sequential processing with round-robin
            all_rewards = []

            for i in range(0, len(queries), self.micro_batch_size):
                batch_queries = queries[i : i + self.micro_batch_size]
                batch_prompts = prompts[i : i + self.micro_batch_size] if prompts else None
                batch_labels = labels[i : i + self.micro_batch_size] if labels else None

                # Round-robin server selection
                server_idx = (i // self.micro_batch_size) % num_servers
                url = self.remote_urls[server_idx]

                data = {"queries": batch_queries}
                if batch_prompts is not None:
                    data["prompts"] = batch_prompts
                if batch_labels is not None:
                    data["labels"] = batch_labels

                result = request_api_wrapper(f"{url}/reward", data)

                if result is not None:
                    rewards = result.get("rewards", result.get("scores", []))
                    all_rewards.extend(rewards)
                else:
                    all_rewards.extend([0.0] * len(batch_queries))

            return all_rewards


def create_reward_server_handler(
    reward_fn: Callable[[list[str], list[str] | None, list[Any] | None], list[float]],
) -> Callable:
    """
    Create a request handler for reward model server.

    Args:
        reward_fn: Reward function to wrap

    Returns:
        Handler function for web framework

    Example:
        ```python
        from flask import Flask, request, jsonify

        def my_reward_fn(queries, prompts=None, labels=None):
            return [len(q) / 100 for q in queries]  # Dummy reward

        app = Flask(__name__)
        handler = create_reward_server_handler(my_reward_fn)

        @app.route("/reward", methods=["POST"])
        def reward_endpoint():
            return handler(request.json)
        ```
    """

    def handler(data: dict[str, Any]) -> dict[str, Any]:
        queries = data.get("queries", [])
        prompts = data.get("prompts")
        labels = data.get("labels")

        try:
            rewards = reward_fn(queries, prompts=prompts, labels=labels)
            return {"rewards": rewards, "status": "success"}
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return {"rewards": [0.0] * len(queries), "status": "error", "error": str(e)}

    return handler


# Public API
__all__ = [
    "request_api_wrapper",
    "RemoteRewardModel",
    "create_reward_server_handler",
]

# Conditional exports
if _RAY_AVAILABLE:
    __all__.append("remote_rm_fn_ray")
