"""
ThinkRL Agent Utilities
========================

Agent utilities for agentic RLHF training.
Aligned with OpenRLHF patterns for agent-based reinforcement learning.

Author: Archit Sood @ EllanorAI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


# Optional Ray import
try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False
    ray = None


# Optional vLLM import
try:
    from vllm.inputs import TokensPrompt

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    TokensPrompt = None


@dataclass
class AgentState:
    """
    State container for agent execution.

    Tracks the current state of an agent during multi-step interactions.
    """

    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    action_positions: list[tuple[int, int]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    done: bool = False
    step_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentInstanceBase(ABC):
    """
    Abstract base class for agent instances.

    Defines the interface for agents that interact with environments
    during RLHF training. Subclass this to implement custom agents.

    Example:
        ```python
        class MyAgent(AgentInstanceBase):
            def __init__(self, env_config):
                self.env = Environment(**env_config)

            def reset(self, states: dict, **kwargs) -> dict:
                self.env.reset()
                return states

            def step(self, state_dict: dict, **kwargs) -> dict:
                action = state_dict.get("action", "")
                obs, reward, done, info = self.env.step(action)
                return {
                    "observation": obs,
                    "reward": reward,
                    "done": done,
                    **info,
                }
        ```
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the agent instance."""
        pass

    def reset(self, states: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Reset the agent state.

        Args:
            states: Current state dictionary
            **kwargs: Additional reset arguments

        Returns:
            Modified state dictionary
        """
        return states

    @abstractmethod
    def step(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Execute one step of the agent.

        Args:
            state_dict: Current state dictionary containing action info
            **kwargs: Additional step arguments

        Returns:
            Updated state dictionary with observation, reward, done status
        """
        pass


class AgentExecutorBase:
    """
    Base class for managing concurrent agent execution.

    Orchestrates multiple agent instances, handles LLM generation,
    and manages the interaction loop between agents and environments.

    Example:
        ```python
        executor = AgentExecutorBase(
            llm_engine=my_vllm_engine,
            tokenizer=my_tokenizer,
            agent_class=MyAgent,
            max_steps=10,
            max_length=2048,
        )

        # Execute agents concurrently
        await executor.run_batch(prompts, sampling_params)
        ```
    """

    def __init__(
        self,
        llm_engine: Any,
        tokenizer: Any,
        agent_class: type[AgentInstanceBase] | None = None,
        max_steps: int = 10,
        max_length: int = 2048,
        max_concurrent: int = 128,
        compute_log_probs: bool = True,
    ):
        """
        Initialize the agent executor.

        Args:
            llm_engine: LLM engine for generation (e.g., vLLM engine)
            tokenizer: Tokenizer for encoding/decoding
            agent_class: Agent class to instantiate for each prompt
            max_steps: Maximum steps per episode
            max_length: Maximum token length
            max_concurrent: Maximum concurrent agent tasks
            compute_log_probs: Whether to compute log probabilities
        """
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.agent_class = agent_class
        self.max_steps = max_steps
        self.max_length = max_length
        self.compute_log_probs = compute_log_probs

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._result_queue: asyncio.Queue = asyncio.Queue()

    async def generate(
        self,
        prompt_tokens: list[int],
        sampling_params: Any,
    ) -> dict[str, Any]:
        """
        Generate text from prompt tokens.

        Args:
            prompt_tokens: Input token IDs
            sampling_params: vLLM sampling parameters

        Returns:
            Generation result with output tokens and metadata
        """
        if not _VLLM_AVAILABLE:
            raise RuntimeError("vLLM is required for agent generation")

        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)

        async with self._semaphore:
            request = await self.llm_engine.add_request(
                request_id=f"agent_{id(prompt_tokens)}",
                inputs=prompt,
                params=sampling_params,
            )

            outputs = []
            async for output in request:
                outputs.append(output)

            final_output = outputs[-1] if outputs else None

            if final_output is None:
                return {"output_ids": [], "text": "", "log_probs": []}

            return {
                "output_ids": final_output.outputs[0].token_ids,
                "text": final_output.outputs[0].text,
                "log_probs": (final_output.outputs[0].logprobs if self.compute_log_probs else []),
            }

    async def execute(
        self,
        prompt: str | list[int],
        sampling_params: Any,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> AgentState:
        """
        Execute a single agent episode.

        Args:
            prompt: Initial prompt (text or token IDs)
            sampling_params: vLLM sampling parameters
            agent_kwargs: Arguments for agent initialization

        Returns:
            Final agent state with trajectory
        """
        agent_kwargs = agent_kwargs or {}

        # Initialize agent
        if self.agent_class is not None:
            if _RAY_AVAILABLE:
                agent = ray.remote(self.agent_class).remote(**agent_kwargs)
            else:
                agent = self.agent_class(**agent_kwargs)
        else:
            agent = None

        # Initialize state
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = list(prompt)

        state = AgentState(input_ids=input_ids)

        # Reset agent
        if agent is not None:
            initial_state = {"input_ids": input_ids}
            if _RAY_AVAILABLE:
                initial_state = await asyncio.to_thread(ray.get, agent.reset.remote(initial_state))
            else:
                initial_state = agent.reset(initial_state)
            state.metadata.update(initial_state)

        # Interaction loop
        for step in range(self.max_steps):
            if state.done:
                break

            if len(state.input_ids) >= self.max_length:
                logger.warning(f"Reached max length {self.max_length}")
                break

            # Generate action
            action_start = len(state.input_ids)
            result = await self.generate(state.input_ids, sampling_params)

            action_end = action_start + len(result["output_ids"])
            state.input_ids.extend(result["output_ids"])
            state.actions.append(result["text"])
            state.action_positions.append((action_start, action_end))

            if result["log_probs"]:
                state.log_probs.extend(result["log_probs"])

            # Execute agent step
            if agent is not None:
                step_input = {
                    "action": result["text"],
                    "action_tokens": result["output_ids"],
                    "input_ids": state.input_ids,
                    "step": step,
                }

                if _RAY_AVAILABLE:
                    step_result = await asyncio.to_thread(ray.get, agent.step.remote(step_input))
                else:
                    step_result = agent.step(step_input)

                observation = step_result.get("observation", "")
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", False)

                state.observations.append(observation)
                state.rewards.append(reward)
                state.done = done

                # Append observation to context
                if observation:
                    obs_tokens = self.tokenizer.encode(observation)
                    state.input_ids.extend(obs_tokens)

            state.step_count = step + 1

        # Cleanup Ray actor
        if agent is not None and _RAY_AVAILABLE:
            ray.kill(agent)

        await self._result_queue.put(state)
        return state

    async def run_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: Any,
        agent_kwargs_list: list[dict[str, Any]] | None = None,
    ) -> list[AgentState]:
        """
        Run multiple agent episodes concurrently.

        Args:
            prompts: List of prompts
            sampling_params: Sampling parameters (shared)
            agent_kwargs_list: Per-prompt agent kwargs

        Returns:
            List of final agent states
        """
        if agent_kwargs_list is None:
            agent_kwargs_list = [{}] * len(prompts)

        tasks = [self.execute(prompt, sampling_params, kwargs) for prompt, kwargs in zip(prompts, agent_kwargs_list)]

        results = await asyncio.gather(*tasks)
        return list(results)


def load_agent_class(
    module_path: str,
    class_name: str = "Agent",
) -> type[AgentInstanceBase]:
    """
    Dynamically load an agent class from a Python file.

    Args:
        module_path: Path to Python file containing agent class
        class_name: Name of the agent class

    Returns:
        Loaded agent class

    Example:
        ```python
        AgentClass = load_agent_class("./agents/my_agent.py", "MyAgent")
        agent = AgentClass(env_config)
        ```
    """
    import importlib.util

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Agent module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("agent_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_path}")

    agent_class = getattr(module, class_name)

    if not issubclass(agent_class, AgentInstanceBase):
        logger.warning(
            f"{class_name} does not inherit from AgentInstanceBase. " "Consider using the base class for consistency."
        )

    return agent_class


def create_agent_remote(
    agent_class: type[AgentInstanceBase],
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
) -> Any:
    """
    Create a Ray remote version of an agent class.

    Args:
        agent_class: Agent class to wrap
        num_cpus: CPU resources per agent
        num_gpus: GPU resources per agent

    Returns:
        Ray remote actor class

    Example:
        ```python
        RemoteAgent = create_agent_remote(MyAgent, num_cpus=1)
        agent = RemoteAgent.remote(env_config)
        ```
    """
    if not _RAY_AVAILABLE:
        raise RuntimeError("Ray is required for remote agents")

    return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(agent_class)


# Public API
__all__ = [
    # Base classes
    "AgentState",
    "AgentInstanceBase",
    "AgentExecutorBase",
    # Utilities
    "load_agent_class",
    "create_agent_remote",
]
