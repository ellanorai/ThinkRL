import pytest
import torch
import torch.nn.functional as F

from thinkrl.models.loss import (
    DPOLoss,
    EntropyLoss,
    GPTLMLoss,
    KTOLoss,
    LogExpLoss,
    PairWiseLoss,
    PolicyLoss,
    SFTLoss,
    ValueLoss,
)


class TestGPTLMLoss:
    def test_forward(self):
        loss_fn = GPTLMLoss()
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_ignore_index(self):
        loss_fn = GPTLMLoss()
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100)
        loss = loss_fn(logits, labels)
        assert loss.item() == 0.0


class TestSFTLoss:
    def test_forward(self):
        loss_fn = SFTLoss()
        log_probs = torch.randn(2, 10)
        labels = torch.randint(0, 100, (2, 10))
        loss = loss_fn(log_probs, labels)
        # SFT loss is negative log sum. My log_probs are random floats (could be positive).
        # SFTLoss expects log_probs? Yes.
        # Logic: masked_log_probs = log_probs * mask. Loss = -sum / count.
        assert isinstance(loss, torch.Tensor)

    def test_attention_mask(self):
        loss_fn = SFTLoss()
        log_probs = torch.randn(2, 10)
        labels = torch.randint(0, 100, (2, 10))
        mask = torch.zeros(2, 10)
        loss = loss_fn(log_probs, labels, attention_mask=mask)
        assert loss.item() == 0.0  # mostly, depends on clamp


class TestPolicyLoss:
    def test_forward(self):
        loss_fn = PolicyLoss(clip_eps=0.2)
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages)
        assert isinstance(loss, torch.Tensor)
        assert "policy_loss" in metrics

    def test_dual_clip(self):
        loss_fn = PolicyLoss(dual_clip=0.3)
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs - 1.0  # ratio becomes small
        advantages = -torch.ones(2, 10)  # negative advantage triggers dual clip

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages)
        assert isinstance(loss, torch.Tensor)

    def test_action_mask(self):
        loss_fn = PolicyLoss()
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)
        mask = torch.zeros(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask)
        assert loss.item() == 0.0


class TestValueLoss:
    def test_forward(self):
        loss_fn = ValueLoss()
        values = torch.randn(2, 10)
        old_values = values.clone()
        returns = torch.randn(2, 10)

        loss = loss_fn(values, old_values, returns)
        assert loss.item() >= 0

    def test_clipping(self):
        loss_fn = ValueLoss(clip_eps=0.2)
        values = torch.randn(2, 10)
        old_values = values.clone()
        returns = torch.randn(2, 10)

        loss = loss_fn(values, old_values, returns)
        assert loss.item() >= 0


class TestPairWiseLoss:
    def test_forward(self):
        loss_fn = PairWiseLoss()
        chosen = torch.randn(10)
        rejected = torch.randn(10)
        loss, metrics = loss_fn(chosen, rejected)
        assert isinstance(loss, torch.Tensor)


class TestLogExpLoss:
    def test_forward(self):
        loss_fn = LogExpLoss()
        chosen = torch.randn(10)
        rejected = torch.randn(10)
        loss, metrics = loss_fn(chosen, rejected)
        assert isinstance(loss, torch.Tensor)


class TestDPOLoss:
    def test_forward(self):
        loss_fn = DPOLoss()
        policy_chosen = torch.randn(10)
        policy_rejected = torch.randn(10)
        ref_chosen = torch.randn(10)
        ref_rejected = torch.randn(10)

        loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert isinstance(loss, torch.Tensor)

    def test_ipo(self):
        loss_fn = DPOLoss(ipo=True)
        policy_chosen = torch.randn(10)
        policy_rejected = torch.randn(10)
        ref_chosen = torch.randn(10)
        ref_rejected = torch.randn(10)

        loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert isinstance(loss, torch.Tensor)

    def test_label_smoothing(self):
        loss_fn = DPOLoss(label_smoothing=0.1)
        policy_chosen = torch.randn(10)
        policy_rejected = torch.randn(10)
        ref_chosen = torch.randn(10)
        ref_rejected = torch.randn(10)

        loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert isinstance(loss, torch.Tensor)


class TestKTOLoss:
    def test_forward(self):
        loss_fn = KTOLoss()
        policy = torch.randn(10)
        ref = torch.randn(10)
        labels = torch.randint(0, 2, (10,), dtype=torch.float)  # 0 or 1

        loss, metrics = loss_fn(policy, ref, labels)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


class TestEntropyLoss:
    def test_forward(self):
        loss_fn = EntropyLoss()
        logits = torch.randn(2, 10, 100)
        loss = loss_fn(logits)
        # Entropy is positive, loss returns -coeff * entropy, so usually negative
        assert isinstance(loss, torch.Tensor)

    def test_mask(self):
        loss_fn = EntropyLoss()
        logits = torch.randn(2, 10, 100)
        mask = torch.zeros(2, 10)
        loss = loss_fn(logits, action_mask=mask)
        assert loss.item() == 0.0  # if all masked
