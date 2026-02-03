import math

import pytest

from thinkrl.training.kl_controller import (
    AdaptiveKLController,
    FixedKLController,
    KLController,
    KLControllerConfig,
    KLControllerType,
)


def test_fixed_kl_controller():
    """Test fixed KL controller."""
    ctl = FixedKLController(kl_coef=0.5)
    assert ctl.get_kl_coef() == 0.5

    # Update should not change coef
    new_coef = ctl.update(current_kl=100.0)
    assert new_coef == 0.5
    assert ctl.step == 1


def test_adaptive_kl_controller_increase():
    """Test adaptive controller increases coef when KL is too high."""
    ctl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, kl_lr=0.1)

    # Current KL (0.02) > Target (0.01) -> should increase
    ctl.update(current_kl=0.02)
    assert ctl.get_kl_coef() > 0.1


def test_adaptive_kl_controller_decrease():
    """Test adaptive controller decreases coef when KL is too low."""
    ctl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, kl_lr=0.1)

    # Current KL (0.005) < Target (0.01) -> should decrease
    ctl.update(current_kl=0.005)
    assert ctl.get_kl_coef() < 0.1


def test_adaptive_kl_bounds():
    """Test min/max clipping."""
    ctl = AdaptiveKLController(init_kl_coef=1.0, target_kl=1.0, min_kl_coef=0.5, max_kl_coef=2.0)

    # Force decrease below min
    ctl.kl_coef = 0.4
    ctl.update(current_kl=1.0)  # no error error=0
    # Actually need to trigger update logic
    # but manually setting it is easier to test implementation detail if we mock _adaptive_update
    # Let's test via public API

    # Set high LR to force rapid change
    ctl.kl_lr = 100.0

    # Try to go WAY up
    ctl.update(current_kl=100.0)
    assert ctl.get_kl_coef() <= 2.0

    # Try to go WAY down
    ctl.kl_coef = 1.0
    ctl.target_kl = 100.0
    ctl.update(current_kl=0.0)
    assert ctl.get_kl_coef() >= 0.5


def test_linear_schedule():
    """Test linear schedule."""
    ctl = KLController(init_kl_coef=0.1, final_kl_coef=0.0, total_steps=10, controller_type="linear")

    ctl.update(0.0)
    # Step 1/10 -> progress 0.1
    # 0.1 + 0.1 * (0.0 - 0.1) = 0.1 - 0.01 = 0.09
    assert ctl.get_kl_coef() == pytest.approx(0.09)


def test_state_dict():
    """Test checkpointing."""
    ctl = AdaptiveKLController(init_kl_coef=0.1)
    ctl.update(0.05)

    state = ctl.state_dict()
    assert state["step"] == 1
    assert len(state["kl_history"]) == 1

    ctl2 = AdaptiveKLController(init_kl_coef=0.5)
    ctl2.load_state_dict(state)
    assert ctl2.step == 1
    assert ctl2.get_kl_coef() == ctl.get_kl_coef()
