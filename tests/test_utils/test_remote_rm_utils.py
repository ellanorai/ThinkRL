"""
Test Suite for ThinkRL Remote Reward Model Utilities
=====================================================

Tests for:
- thinkrl.utils.remote_rm_utils
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from thinkrl.utils.remote_rm_utils import (
    RemoteRewardModel,
    create_reward_server_handler,
    request_api_wrapper,
    _REQUESTS_AVAILABLE,
    _RAY_AVAILABLE,
)


class TestRequestAPIWrapper:
    """Tests for request_api_wrapper function."""

    def test_without_requests_library(self):
        """Test that missing requests library raises error."""
        with patch("thinkrl.utils.remote_rm_utils._REQUESTS_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="requests library is required"):
                request_api_wrapper("http://localhost:8000", {"data": "test"})

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed")
    def test_successful_request(self):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"rewards": [0.5, 0.7]}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = request_api_wrapper(
                "http://localhost:8000/reward",
                {"queries": ["text1", "text2"]},
            )

            assert result == {"rewards": [0.5, 0.7]}

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed")
    def test_request_retry_on_failure(self):
        """Test request retries on failure."""
        import requests

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.RequestException("Connection failed")
            response = MagicMock()
            response.json.return_value = {"rewards": [0.5]}
            response.raise_for_status = MagicMock()
            return response

        with patch("requests.post", side_effect=mock_post):
            with patch("time.sleep"):  # Skip sleep for faster tests
                result = request_api_wrapper(
                    "http://localhost:8000/reward",
                    {"queries": ["text"]},
                    try_max_times=5,
                    retry_delay=0.1,
                )

        assert result == {"rewards": [0.5]}
        assert call_count == 3

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed")
    def test_request_all_retries_fail(self):
        """Test that all retries failing returns None."""
        import requests

        with patch("requests.post", side_effect=requests.exceptions.RequestException("Failed")):
            with patch("time.sleep"):
                result = request_api_wrapper(
                    "http://localhost:8000/reward",
                    {"queries": ["text"]},
                    try_max_times=3,
                )

        assert result is None


class TestRemoteRewardModel:
    """Tests for RemoteRewardModel class."""

    def test_init_with_single_url(self):
        """Test initialization with single URL."""
        rm = RemoteRewardModel(remote_urls="http://localhost:8000")

        assert rm.remote_urls == ["http://localhost:8000"]
        assert rm.reward_fn is None

    def test_init_with_multiple_urls(self):
        """Test initialization with multiple URLs."""
        urls = ["http://localhost:8000", "http://localhost:8001"]
        rm = RemoteRewardModel(remote_urls=urls)

        assert rm.remote_urls == urls

    def test_init_no_source_warning(self, caplog):
        """Test warning when no source is provided."""
        import logging
        with caplog.at_level(logging.WARNING):
            rm = RemoteRewardModel()

        assert "No remote URLs or reward function specified" in caplog.text

    def test_get_rewards_empty_queries(self):
        """Test get_rewards with empty queries list."""
        rm = RemoteRewardModel(remote_urls=["http://localhost:8000"])
        result = rm.get_rewards([])
        assert result == []

    def test_get_rewards_no_source(self, caplog):
        """Test get_rewards with no source returns zeros."""
        import logging
        rm = RemoteRewardModel()

        with caplog.at_level(logging.WARNING):
            result = rm.get_rewards(["query1", "query2"])

        assert result == [0.0, 0.0]
        assert "No reward source available" in caplog.text

    @pytest.fixture
    def temp_reward_fn_file(self):
        """Create a temporary file with reward function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def compute_reward(queries, prompts=None, labels=None):
    return [len(q) / 100.0 for q in queries]
""")
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_init_with_reward_fn(self, temp_reward_fn_file):
        """Test initialization with custom reward function."""
        rm = RemoteRewardModel(
            reward_fn_path=temp_reward_fn_file,
            reward_fn_name="compute_reward",
        )

        assert rm.reward_fn is not None

    def test_get_rewards_with_custom_fn(self, temp_reward_fn_file):
        """Test get_rewards with custom function."""
        rm = RemoteRewardModel(
            reward_fn_path=temp_reward_fn_file,
            reward_fn_name="compute_reward",
            use_ray=False,
        )

        queries = ["short", "medium length text"]
        result = rm.get_rewards(queries)

        assert len(result) == 2
        assert result[0] == 5 / 100.0  # "short" = 5 chars
        assert result[1] == 18 / 100.0  # "medium length text" = 18 chars

    def test_load_reward_fn_file_not_found(self):
        """Test loading reward function from non-existent file."""
        with pytest.raises(FileNotFoundError):
            RemoteRewardModel(
                reward_fn_path="/nonexistent/path.py",
                reward_fn_name="compute_reward",
            )

    def test_load_reward_fn_function_not_found(self, temp_reward_fn_file):
        """Test loading non-existent function from file."""
        with pytest.raises(AttributeError, match="not found"):
            RemoteRewardModel(
                reward_fn_path=temp_reward_fn_file,
                reward_fn_name="nonexistent_function",
            )

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_rewards_from_servers_no_ray(self):
        """Test get_rewards from remote servers without Ray."""
        rm = RemoteRewardModel(
            remote_urls=["http://localhost:8000"],
            use_ray=False,
        )

        mock_response = {"rewards": [0.5, 0.7]}

        with patch("thinkrl.utils.remote_rm_utils.request_api_wrapper", return_value=mock_response):
            result = rm.get_rewards(["query1", "query2"])

        assert result == [0.5, 0.7]

    @pytest.mark.skipif(not _REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_rewards_from_servers_request_failed(self):
        """Test get_rewards handles failed requests gracefully."""
        rm = RemoteRewardModel(
            remote_urls=["http://localhost:8000"],
            use_ray=False,
        )

        with patch("thinkrl.utils.remote_rm_utils.request_api_wrapper", return_value=None):
            result = rm.get_rewards(["query1", "query2"])

        assert result == [0.0, 0.0]


class TestCreateRewardServerHandler:
    """Tests for create_reward_server_handler function."""

    def test_create_handler(self):
        """Test creating a handler from reward function."""

        def my_reward_fn(queries, prompts=None, labels=None):
            return [len(q) / 10.0 for q in queries]

        handler = create_reward_server_handler(my_reward_fn)

        assert callable(handler)

    def test_handler_success(self):
        """Test handler processes request successfully."""

        def my_reward_fn(queries, prompts=None, labels=None):
            return [0.5, 0.7]

        handler = create_reward_server_handler(my_reward_fn)

        data = {"queries": ["query1", "query2"]}
        result = handler(data)

        assert result["status"] == "success"
        assert result["rewards"] == [0.5, 0.7]

    def test_handler_with_prompts_and_labels(self):
        """Test handler passes prompts and labels."""

        def my_reward_fn(queries, prompts=None, labels=None):
            assert prompts is not None
            assert labels is not None
            return [0.5]

        handler = create_reward_server_handler(my_reward_fn)

        data = {
            "queries": ["query1"],
            "prompts": ["prompt1"],
            "labels": ["label1"],
        }
        result = handler(data)

        assert result["status"] == "success"

    def test_handler_error(self):
        """Test handler handles errors gracefully."""

        def failing_reward_fn(queries, prompts=None, labels=None):
            raise ValueError("Reward computation failed")

        handler = create_reward_server_handler(failing_reward_fn)

        data = {"queries": ["query1"]}
        result = handler(data)

        assert result["status"] == "error"
        assert "error" in result
        assert result["rewards"] == [0.0]


class TestRemoteRewardModelMicroBatching:
    """Tests for micro-batching in RemoteRewardModel."""

    @pytest.fixture
    def temp_reward_fn_file(self):
        """Create a temporary file with reward function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def compute_reward(queries, prompts=None, labels=None):
    return [1.0 for q in queries]
""")
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_micro_batching(self, temp_reward_fn_file):
        """Test that large inputs are processed in micro-batches."""
        rm = RemoteRewardModel(
            reward_fn_path=temp_reward_fn_file,
            reward_fn_name="compute_reward",
            micro_batch_size=10,
            use_ray=False,
        )

        # Create 25 queries (should be processed in 3 batches: 10, 10, 5)
        queries = [f"query_{i}" for i in range(25)]

        result = rm.get_rewards(queries)

        assert len(result) == 25
        assert all(r == 1.0 for r in result)
