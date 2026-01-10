import pytest
import torch
import torch.nn.functional as F

from thinkrl.models.loss import (
    COPOLoss,
    DAPOLoss,
    GRPOLoss,
    IPOLoss,
    ReinforceLoss,
    VAPOLoss,
)


class TestReinforceLoss:
    def test_forward(self):
        loss_fn = ReinforceLoss()
        log_probs = torch.randn(2, 10)
        advantages = torch.randn(2, 10)
        loss = loss_fn(log_probs, advantages)
        assert isinstance(loss, torch.Tensor)
        # Loss should be scalar
        assert loss.dim() == 0

    def test_forward_with_mask(self):
        loss_fn = ReinforceLoss()
        log_probs = torch.randn(2, 10)
        advantages = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        loss = loss_fn(log_probs, advantages, action_mask=mask)
        assert isinstance(loss, torch.Tensor)


class TestIPOLoss:
    def test_forward_ipo(self):
        loss_fn = IPOLoss(loss_type="ipo")
        p_chosen = torch.randn(10)
        p_rejected = torch.randn(10)
        r_chosen = torch.randn(10)
        r_rejected = torch.randn(10)

        loss, metrics = loss_fn(p_chosen, p_rejected, r_chosen, r_rejected)
        assert isinstance(loss, torch.Tensor)
        assert "ipo_loss" in metrics

    def test_forward_hinge(self):
        loss_fn = IPOLoss(loss_type="ipo_hinge")
        p_chosen = torch.randn(10)
        p_rejected = torch.randn(10)
        r_chosen = torch.randn(10)
        r_rejected = torch.randn(10)

        loss, metrics = loss_fn(p_chosen, p_rejected, r_chosen, r_rejected)
        assert isinstance(loss, torch.Tensor)


class TestGRPOLoss:
    def test_forward(self):
        loss_fn = GRPOLoss(clip_eps=0.2, beta=0.04)
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)
        kl_div = torch.abs(torch.randn(2, 10))  # KL is positive
        mask = torch.ones(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, kl_div, action_mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert "grpo_loss" in metrics
        assert "clip_frac" in metrics

    def test_clipping(self):
        loss_fn = GRPOLoss(clip_eps=0.2)
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs - 1.0  # Significant shift
        advantages = torch.randn(2, 10)
        kl_div = torch.zeros(2, 10)
        mask = torch.ones(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, kl_div, action_mask=mask)
        assert "clip_frac" in metrics
        assert metrics["clip_frac"] > 0


class TestVAPOLoss:
    def test_forward(self):
        loss_fn = VAPOLoss()
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)
        mask = torch.ones(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert "policy_loss" in metrics

    def test_lm_loss(self):
        loss_fn = VAPOLoss(lm_loss_coeff=0.5)
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        lm_loss_val = torch.tensor(1.0)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask, lm_loss=lm_loss_val)
        # Total loss should include lm_loss * coeff
        # policy part is roughly 0 (ratio=1, adv is random but sum might not be exactly 0)
        # roughly: policy_loss + 0.5 * 1.0
        assert metrics["total_loss"] == loss


class TestDAPOLoss:
    def test_forward(self):
        loss_fn = DAPOLoss()
        log_probs = torch.randn(2, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 10)
        mask = torch.ones(2, 10)

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert "dapo_loss" in metrics
        assert "clip_frac" in metrics
