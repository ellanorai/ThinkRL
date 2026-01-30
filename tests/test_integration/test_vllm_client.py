import unittest
from unittest.mock import MagicMock, patch

import pytest
import requests
import torch


# Define explicit mocks for vLLM modules BEFORE importing the client
# This is crucial because vllm might not be installed in the test environment
mock_vllm = MagicMock()
mock_pynccl = MagicMock()
mock_utils = MagicMock()
mock_vllm.distributed.device_communicators.pynccl.PyNcclCommunicator = mock_pynccl
mock_vllm.distributed.utils.StatelessProcessGroup = mock_utils

# Use patch.dict to map these mocks to sys.modules
with patch.dict(
    "sys.modules",
    {
        "vllm": mock_vllm,
        "vllm.distributed": mock_vllm.distributed,
        "vllm.distributed.device_communicators": mock_vllm.distributed.device_communicators,
        "vllm.distributed.device_communicators.pynccl": mock_vllm.distributed.device_communicators.pynccl,
        "vllm.distributed.utils": mock_vllm.distributed.utils,
    },
):
    from thinkrl.integration.vllm_client import VLLMClient

# Reference the mocked classes for assertions in tests
PyNcclCommunicator = mock_pynccl
StatelessProcessGroup = mock_utils


class TestVLLMClient:
    @pytest.fixture
    def client(self):
        # Mock torch.distributed to look like we are on rank 0 (main process)
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_rank", return_value=0):
            return VLLMClient(url="http://mock-vllm:8000")

    def test_init_basic(self):
        # Test default init
        with patch("torch.distributed.is_available", return_value=False):
            client = VLLMClient()
            assert client.is_main_process is True
            assert client.url == "http://localhost:8000"

    def test_init_distributed_rank1(self):
        # Test init on non-main process
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_rank", return_value=1):
            client = VLLMClient()
            assert client.is_main_process is False

    @patch("requests.post")
    def test_generate_success(self, mock_post, client):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": ["outputs"],
            "token_ids": [[1, 2, 3]],
            "log_probs": [[-0.1, -0.2, -0.3]],
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Call generate
        result = client.generate(["input"], params={"max_tokens": 10})

        # Verify
        assert result["text"] == ["outputs"]
        assert result["token_ids"] == [[1, 2, 3]]

        # Check args
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["prompts"] == ["input"]
        assert kwargs["json"]["logprobs"] == 1

    @patch("requests.post")
    def test_generate_error(self, mock_post, client):
        mock_post.side_effect = requests.RequestException("Connection refused")

        with pytest.raises(RuntimeError, match="vLLM Generation failed"):
            client.generate(["input"], params={})

    def test_generate_skipped_rank(self):
        # Test generate on rank 1 returns empty
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_rank", return_value=1):
            client = VLLMClient()
            result = client.generate(["input"], params={})
            assert result["text"] == []
            assert result["token_ids"] == []

    @pytest.mark.skip(reason="Torch LIBRARY registration conflicts in test environment")
    def test_init_weight_sync(self, client):
        device = torch.device("cpu")
        client.init_weight_sync(device)

        # Verify StatelessProcessGroup.create called
        StatelessProcessGroup.create.assert_called_once()
        call_kwargs = StatelessProcessGroup.create.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["rank"] == 0  # client rank

        PyNcclCommunicator.assert_called_once()

    @pytest.mark.skip(reason="Torch LIBRARY registration conflicts in test environment")
    @patch("requests.post")
    def test_update_model_weights(self, mock_post, client):
        # Mock HTTP call
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Setup communicator mock
        mock_comm_instance = MagicMock()
        client.communicator = mock_comm_instance

        # Create a simple model
        model = torch.nn.Linear(1, 1)

        client.update_model_weights(model)

        # Should broadcast twice (weight and bias)
        assert mock_comm_instance.broadcast.call_count == 2

    def test_update_model_weights_no_init(self, client):
        client.communicator = None
        with pytest.raises(RuntimeError, match="Communicator not initialized"):
            client.update_model_weights(torch.nn.Linear(1, 1))

    @pytest.mark.skip(reason="Torch LIBRARY registration conflicts in test environment")
    @patch("requests.post")
    def test_update_model_weights_fsdp(self, mock_post, client):
        """Test weight update triggers server sync and handles FSDP context."""
        # Mock server response for /update_weights
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Setup communicator mock
        mock_comm_instance = MagicMock()
        client.communicator = mock_comm_instance

        # Create a simple model (not actually FSDP, but tests the path)
        model = torch.nn.Linear(1, 1)

        client.update_model_weights(model)

        # Verify HTTP call to trigger server sync
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "/update_weights" in args[0]

        # Verify broadcast called for both params
        assert mock_comm_instance.broadcast.call_count == 2

    def test_generate_no_logprobs(self, client):
        """Test generate when server doesn't return logprobs."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": ["output"]}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = client.generate(["input"], params={}, return_logprobs=True)

            assert result["text"] == ["output"]
            assert result["token_ids"] == []  # Fallback
            assert result["log_probs"] == []
