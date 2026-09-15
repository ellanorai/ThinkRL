"""Regression test: the RL trainers must produce a held-out signal, not only training reward.

Training reward is the quantity being optimized, so it rises whether or not the policy
improves, and `CheckpointManager`'s mode="max" selection had no metric to rank by. Written
as a contract over all three trainers because the defect was that none of them had it, and
a fix applied to one is the shape this repo has already regressed into once. See #135.
"""

import inspect

import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer
import thinkrl.training.star_trainer as star_trainer


MODULES = (grpo_trainer, reinforce_pp_trainer, star_trainer)
TRAINERS = (grpo_trainer.GRPOTrainer, reinforce_pp_trainer.ReinforcePPTrainer, star_trainer.STaRTrainer)


def test_every_rl_trainer_accepts_an_eval_dataset():
    """The regression: none of these took an evaluation argument at all."""
    for trainer in TRAINERS:
        params = inspect.signature(trainer.train).parameters
        assert "eval_dataset" in params, f"{trainer.__name__}.train has no eval_dataset"
        assert "eval_every" in params, f"{trainer.__name__}.train has no eval_every"


def test_evaluation_is_off_by_default():
    """Existing callers must be unaffected, so the feature has to be opt-in."""
    for trainer in TRAINERS:
        params = inspect.signature(trainer.train).parameters
        assert params["eval_dataset"].default is None
        assert params["eval_every"].default == 0


def test_every_rl_trainer_builds_the_hook():
    for module in MODULES:
        source = inspect.getsource(module)
        assert "build_periodic_evaluator" in source, f"{module.__name__} never builds an evaluator"


def test_the_held_out_metric_drives_best_checkpoint_selection():
    """CheckpointManager supports mode="max" and had nothing to rank by, which is half
    of why #104 found nothing wired."""
    for module in MODULES:
        source = inspect.getsource(module)
        assert 'metric_name="eval/reward_mean"' in source, (
            f"{module.__name__} does not hand the held-out metric to CheckpointManager"
        )
        assert 'mode="max"' in source, f"{module.__name__} does not select on maximum"


def test_training_and_eval_reward_are_logged_under_separate_keys():
    """The two diverging is the entire signal, so they must not share a key."""
    from thinkrl.evaluation.periodic import PeriodicEvaluator

    source = inspect.getsource(PeriodicEvaluator.evaluate)
    assert 'f"eval/{name}"' in source
