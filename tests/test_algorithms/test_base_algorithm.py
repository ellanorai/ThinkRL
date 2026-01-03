from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.base import BaseRLHFAlgorithm


# Concrete implementation for testing abstract base class
class ConcreteAlgorithm(BaseRLHFAlgorithm):
    def compute_loss(self, batch):
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def training_step(self, batch):
        return {"loss": 0.0}

    def _generate_with_policy_model(self, prompts, **kwargs):
        return ["generated" for _ in prompts]


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, input_ids, attention_mask=None):
        # Mock output suitable for get_log_probs
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 100)  # vocab size 100
        return {"logits": logits}


@pytest.fixture
def policy_model():
    return SimpleModel()


@pytest.fixture
def algo(policy_model):
    return ConcreteAlgorithm(policy_model)


class TestBaseRLHFAlgorithm:
    def test_init_defaults(self, policy_model):
        algo = ConcreteAlgorithm(policy_model)
        assert algo.learning_rate == 1e-5
        assert algo.use_vllm is False
        assert algo.optimizer is not None

    def test_init_vllm_not_available(self, policy_model):
        with patch("thinkrl.algorithms.base._VLLM_AVAILABLE", False):
            with pytest.raises(ImportError, match="vLLM is required"):
                ConcreteAlgorithm(policy_model, use_vllm=True)

    def test_init_vllm_client(self, policy_model):
        with patch("thinkrl.algorithms.base._VLLM_AVAILABLE", True):
            with patch("thinkrl.algorithms.base.VLLMClient") as MockClient:
                algo = ConcreteAlgorithm(policy_model, use_vllm=True, vllm_url="http://test")
                assert algo.vllm_client is not None
                MockClient.assert_called_with(url="http://test", group_port=51216)

    def test_compute_kl_penalty_no_ref(self, algo):
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }
        kl = algo.compute_kl_penalty(batch)
        assert kl.item() == 0.0

    def test_compute_kl_penalty_with_ref(self, policy_model):
        ref_model = SimpleModel()
        algo = ConcreteAlgorithm(policy_model, ref_model=ref_model)

        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }

        # Should call ref model and compute KL
        # We can't easily assert value without controlling random seeds/weights,
        # but we can check it runs and returns a tensor
        kl = algo.compute_kl_penalty(batch)
        assert isinstance(kl, torch.Tensor)

    def test_compute_kl_penalty_precomputed(self, algo):
        batch = {
            "ref_log_probs": torch.randn(2, 10),
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.randint(0, 100, (2, 10)),
        }
        kl = algo.compute_kl_penalty(batch)
        assert isinstance(kl, torch.Tensor)

    def test_process_rewards(self, algo):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        # Default normalize=True
        processed = algo.process_rewards(rewards)
        # Should be normalized (mean 0, std 1 ideally)
        assert torch.abs(processed.mean()) < 1e-5

        # Explicit normalize=False
        processed_raw = algo.process_rewards(rewards, normalize=False)
        assert torch.equal(processed_raw, rewards)

    def test_generate_rollouts_policy_fallback(self, algo):
        prompts = ["a", "b"]
        result = algo.generate_rollouts(prompts)
        assert result["text"] == ["generated", "generated"]
        assert result["token_ids"] == []
        assert result["log_probs"] == []

    def test_generate_rollouts_vllm(self, policy_model):
        with patch("thinkrl.algorithms.base._VLLM_AVAILABLE", True):
            with patch("thinkrl.algorithms.base.VLLMClient") as MockClient:
                algo = ConcreteAlgorithm(policy_model, use_vllm=True)
                algo.vllm_client.generate.return_value = {"text": ["vllm"], "token_ids": [[1]], "log_probs": [[-0.1]]}

                result = algo.generate_rollouts(["test"])
                assert result["text"] == ["vllm"]
                algo.vllm_client.generate.assert_called_once()

    def test_sync_vllm_weights(self, policy_model):
        with patch("thinkrl.algorithms.base._VLLM_AVAILABLE", True):
            with patch("thinkrl.algorithms.base.VLLMClient") as MockClient:
                algo = ConcreteAlgorithm(policy_model, use_vllm=True)
                algo.sync_vllm_weights()
                algo.vllm_client.update_model_weights.assert_called_with(algo.policy_model)

    def test_init_vllm_weight_sync(self, policy_model):
        with patch("thinkrl.algorithms.base._VLLM_AVAILABLE", True):
            with patch("thinkrl.algorithms.base.VLLMClient") as MockClient:
                algo = ConcreteAlgorithm(policy_model, use_vllm=True)
                device = torch.device("cpu")
                algo.init_vllm_weight_sync(device)
                algo.vllm_client.init_weight_sync.assert_called_with(device)

    def test_save_load_state_dict(self, policy_model):
        algo = ConcreteAlgorithm(policy_model, learning_rate=0.1)
        state = algo.state_dict()

        assert "policy_model" in state
        assert "optimizer" in state
        assert "config" in state
        assert state["config"]["learning_rate"] == 0.1

        # Create new algo
        new_algo = ConcreteAlgorithm(SimpleModel(), learning_rate=0.01)
        new_algo.load_state_dict(state)

        assert new_algo.learning_rate == 0.1
        # Check model weights loaded (random weights match)
        for p1, p2 in zip(algo.policy_model.parameters(), new_algo.policy_model.parameters()):
            assert torch.equal(p1, p2)

    def test_get_metrics(self, algo):
        algo.metrics_tracker.update_dict({"loss": 0.5})
        metrics = algo.get_metrics()
        assert metrics["loss"] == 0.5

        algo.reset_metrics()
        # get_metrics returns average, if reset, it might return empty dict or 0 depending on implementation
        # The implementation returns {} if metrics is int
        metrics_reset = algo.get_metrics()
        assert metrics_reset == {}
