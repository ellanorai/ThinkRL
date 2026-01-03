"""
Test Suite for ThinkRL Agent Utilities
======================================

Tests for:
- thinkrl.utils.agent (AgentState, AgentInstanceBase, AgentExecutorBase, etc.)
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from thinkrl.utils.agent import (
    AgentExecutorBase,
    AgentInstanceBase,
    AgentState,
    load_agent_class,
    create_agent_remote,
    _RAY_AVAILABLE,
    _VLLM_AVAILABLE,
)


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_default_initialization(self):
        """Test default values of AgentState."""
        state = AgentState()
        assert state.input_ids == []
        assert state.attention_mask == []
        assert state.action_positions == []
        assert state.observations == []
        assert state.actions == []
        assert state.rewards == []
        assert state.log_probs == []
        assert state.done is False
        assert state.step_count == 0
        assert state.metadata == {}

    def test_custom_initialization(self):
        """Test AgentState with custom values."""
        state = AgentState(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            action_positions=[(0, 3)],
            observations=["obs1"],
            actions=["action1"],
            rewards=[0.5],
            log_probs=[-0.1],
            done=True,
            step_count=1,
            metadata={"key": "value"},
        )
        assert state.input_ids == [1, 2, 3]
        assert state.done is True
        assert state.step_count == 1
        assert state.metadata["key"] == "value"

    def test_state_mutability(self):
        """Test that AgentState can be mutated."""
        state = AgentState()
        state.input_ids.append(1)
        state.done = True
        state.step_count = 5

        assert state.input_ids == [1]
        assert state.done is True
        assert state.step_count == 5


class ConcreteAgent(AgentInstanceBase):
    """Concrete implementation for testing."""

    def __init__(self, env_name: str = "test"):
        super().__init__()
        self.env_name = env_name
        self.step_count = 0

    def step(self, state_dict):
        self.step_count += 1
        return {
            "observation": f"obs_{self.step_count}",
            "reward": 0.5,
            "done": self.step_count >= 3,
        }

    def reset(self, states, **kwargs):
        self.step_count = 0
        states["reset"] = True
        return states


class TestAgentInstanceBase:
    """Tests for AgentInstanceBase abstract class."""

    def test_concrete_agent_init(self):
        """Test initialization of concrete agent."""
        agent = ConcreteAgent(env_name="my_env")
        assert agent.env_name == "my_env"

    def test_concrete_agent_step(self):
        """Test step method of concrete agent."""
        agent = ConcreteAgent()
        result = agent.step({"input_ids": [1, 2, 3]})

        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert result["reward"] == 0.5
        assert agent.step_count == 1

    def test_concrete_agent_reset(self):
        """Test reset method of concrete agent."""
        agent = ConcreteAgent()
        agent.step({})
        agent.step({})

        assert agent.step_count == 2

        states = {"input_ids": [1]}
        result = agent.reset(states)

        assert agent.step_count == 0
        assert result["reset"] is True

    def test_base_reset_default(self):
        """Test that base class reset returns states unchanged."""
        # Create a minimal subclass that doesn't override reset
        class MinimalAgent(AgentInstanceBase):
            def __init__(self):
                super().__init__()
            def step(self, state_dict, **kwargs):
                return {}

        agent = MinimalAgent()
        states = {"test": "value"}
        result = agent.reset(states)
        assert result == states


class TestAgentExecutorBase:
    """Tests for AgentExecutorBase."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "decoded text"
        return tokenizer

    @pytest.fixture
    def mock_llm_engine(self):
        """Create a mock LLM engine."""
        return MagicMock()

    def test_executor_init(self, mock_llm_engine, mock_tokenizer):
        """Test executor initialization."""
        executor = AgentExecutorBase(
            llm_engine=mock_llm_engine,
            tokenizer=mock_tokenizer,
            agent_class=ConcreteAgent,
            max_steps=5,
            max_length=1024,
            max_concurrent=64,
            compute_log_probs=True,
        )

        assert executor.llm_engine is mock_llm_engine
        assert executor.tokenizer is mock_tokenizer
        assert executor.agent_class is ConcreteAgent
        assert executor.max_steps == 5
        assert executor.max_length == 1024
        assert executor.compute_log_probs is True

    def test_executor_init_defaults(self, mock_llm_engine, mock_tokenizer):
        """Test executor initialization with defaults."""
        executor = AgentExecutorBase(
            llm_engine=mock_llm_engine,
            tokenizer=mock_tokenizer,
        )

        assert executor.agent_class is None
        assert executor.max_steps == 10
        assert executor.max_length == 2048

    @pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vLLM not available")
    @pytest.mark.asyncio
    async def test_generate_requires_vllm(self, mock_llm_engine, mock_tokenizer):
        """Test that generate works when vLLM is available."""
        executor = AgentExecutorBase(
            llm_engine=mock_llm_engine,
            tokenizer=mock_tokenizer,
        )
        # This test would require vLLM to be installed

    @pytest.mark.asyncio
    async def test_generate_without_vllm(self, mock_llm_engine, mock_tokenizer):
        """Test that generate raises error without vLLM."""
        with patch("thinkrl.utils.agent._VLLM_AVAILABLE", False):
            executor = AgentExecutorBase(
                llm_engine=mock_llm_engine,
                tokenizer=mock_tokenizer,
            )

            with pytest.raises(RuntimeError, match="vLLM is required"):
                await executor.generate([1, 2, 3], MagicMock())

    @pytest.mark.asyncio
    async def test_execute_with_string_prompt(self, mock_llm_engine, mock_tokenizer):
        """Test execute with string prompt."""
        with patch("thinkrl.utils.agent._VLLM_AVAILABLE", True):
            with patch.object(AgentExecutorBase, "generate") as mock_generate:
                mock_generate.return_value = {
                    "output_ids": [4, 5],
                    "text": "response",
                    "log_probs": [],
                }

                executor = AgentExecutorBase(
                    llm_engine=mock_llm_engine,
                    tokenizer=mock_tokenizer,
                    max_steps=1,
                )

                state = await executor.execute("test prompt", MagicMock())

                assert isinstance(state, AgentState)
                mock_tokenizer.encode.assert_called_with("test prompt")

    @pytest.mark.asyncio
    async def test_execute_with_token_prompt(self, mock_llm_engine, mock_tokenizer):
        """Test execute with token list prompt."""
        with patch("thinkrl.utils.agent._VLLM_AVAILABLE", True):
            with patch.object(AgentExecutorBase, "generate") as mock_generate:
                mock_generate.return_value = {
                    "output_ids": [4, 5],
                    "text": "response",
                    "log_probs": [-0.1, -0.2],
                }

                executor = AgentExecutorBase(
                    llm_engine=mock_llm_engine,
                    tokenizer=mock_tokenizer,
                    max_steps=1,
                    compute_log_probs=True,
                )

                state = await executor.execute([1, 2, 3], MagicMock())

                assert isinstance(state, AgentState)
                assert state.input_ids == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_execute_with_agent_no_ray(self, mock_llm_engine, mock_tokenizer):
        """Test execute with agent class without Ray."""
        with patch("thinkrl.utils.agent._VLLM_AVAILABLE", True):
            with patch("thinkrl.utils.agent._RAY_AVAILABLE", False):
                with patch.object(AgentExecutorBase, "generate") as mock_generate:
                    mock_generate.return_value = {
                        "output_ids": [4],
                        "text": "action",
                        "log_probs": [],
                    }

                    executor = AgentExecutorBase(
                        llm_engine=mock_llm_engine,
                        tokenizer=mock_tokenizer,
                        agent_class=ConcreteAgent,
                        max_steps=2,
                    )

                    state = await executor.execute([1, 2, 3], MagicMock())

                    assert isinstance(state, AgentState)
                    assert len(state.observations) > 0

    @pytest.mark.asyncio
    async def test_run_batch(self, mock_llm_engine, mock_tokenizer):
        """Test batch execution."""
        with patch.object(AgentExecutorBase, "execute") as mock_execute:
            mock_execute.return_value = AgentState()

            executor = AgentExecutorBase(
                llm_engine=mock_llm_engine,
                tokenizer=mock_tokenizer,
            )

            results = await executor.run_batch(
                prompts=["prompt1", "prompt2"],
                sampling_params=MagicMock(),
            )

            assert len(results) == 2
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_run_batch_with_kwargs(self, mock_llm_engine, mock_tokenizer):
        """Test batch execution with agent kwargs."""
        with patch.object(AgentExecutorBase, "execute") as mock_execute:
            mock_execute.return_value = AgentState()

            executor = AgentExecutorBase(
                llm_engine=mock_llm_engine,
                tokenizer=mock_tokenizer,
            )

            results = await executor.run_batch(
                prompts=["prompt1", "prompt2"],
                sampling_params=MagicMock(),
                agent_kwargs_list=[{"env": "env1"}, {"env": "env2"}],
            )

            assert len(results) == 2


class TestLoadAgentClass:
    """Tests for load_agent_class function."""

    @pytest.fixture
    def temp_agent_file(self):
        """Create a temporary Python file with an agent class."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from thinkrl.utils.agent import AgentInstanceBase

class TestAgentFromFile(AgentInstanceBase):
    def __init__(self, config=None):
        self.config = config

    def step(self, state_dict, **kwargs):
        return {"observation": "obs", "reward": 1.0, "done": True}
""")
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_agent_file_not_subclass(self):
        """Create a temp file with a non-subclass agent."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
class NotAnAgent:
    def __init__(self):
        pass

    def step(self, state_dict):
        return {}
""")
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_load_agent_class_success(self, temp_agent_file):
        """Test loading agent class from file."""
        AgentClass = load_agent_class(temp_agent_file, "TestAgentFromFile")

        assert AgentClass is not None
        assert AgentClass.__name__ == "TestAgentFromFile"

        # Test instantiation
        agent = AgentClass(config={"key": "value"})
        assert agent.config == {"key": "value"}

    def test_load_agent_class_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_agent_class("/nonexistent/path.py", "Agent")

    def test_load_agent_class_class_not_found(self, temp_agent_file):
        """Test error when class doesn't exist in file."""
        with pytest.raises(AttributeError, match="not found"):
            load_agent_class(temp_agent_file, "NonexistentClass")

    def test_load_agent_class_not_subclass_warning(self, temp_agent_file_not_subclass, caplog):
        """Test warning when class is not a subclass of AgentInstanceBase."""
        with caplog.at_level(logging.WARNING):
            AgentClass = load_agent_class(temp_agent_file_not_subclass, "NotAnAgent")

        assert AgentClass is not None
        assert "does not inherit from AgentInstanceBase" in caplog.text


class TestCreateAgentRemote:
    """Tests for create_agent_remote function."""

    def test_create_agent_remote_without_ray(self):
        """Test that create_agent_remote raises error without Ray."""
        with patch("thinkrl.utils.agent._RAY_AVAILABLE", False):
            # Need to reload or patch at module level
            with pytest.raises(RuntimeError, match="Ray is required"):
                create_agent_remote(ConcreteAgent)

    @pytest.mark.skipif(not _RAY_AVAILABLE, reason="Ray not available")
    def test_create_agent_remote_with_ray(self):
        """Test creating remote agent with Ray."""
        RemoteAgent = create_agent_remote(ConcreteAgent, num_cpus=0.5, num_gpus=0.0)
        assert RemoteAgent is not None
