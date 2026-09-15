"""#128: the RL trainers had no distributed path at all.

`grep -rn "deepspeed\\|DistributedDataParallel" thinkrl/training/` matched sft_trainer.py
only, while the README recommends GRPO for reasoning work and GRPO was the algorithm with
none.

The synchronisation check runs two real processes over gloo on CPU, so the claim is
verified rather than asserted and needs no GPU budget. It runs as a subprocess with a
timeout: torch.multiprocessing.spawn inside a pytest process deadlocks on macOS, and a
hanging test is worse than no test.
"""

import inspect
import pathlib
import subprocess
import sys

import pytest
import torch.nn as nn

from thinkrl.training.distributed import is_distributed, unwrap_model, wrap_policy
import thinkrl.training.grpo_trainer as grpo_trainer
import thinkrl.training.reinforce_pp_trainer as reinforce_pp_trainer


WORKER = pathlib.Path(__file__).with_name("_ddp_worker.py")


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


def test_unwrap_is_a_no_op_on_a_plain_module():
    """Call sites should not have to know whether the run is distributed."""
    model = _Tiny()

    assert unwrap_model(model) is model


def test_wrap_is_a_no_op_outside_a_process_group():
    """A single-process run must take exactly the path it took before."""
    model = _Tiny()

    assert wrap_policy(model) is model
    assert is_distributed() is False


def test_gradients_are_synchronised_across_two_processes():
    """The real check: two gloo ranks, different data, one averaged gradient.

    Also asserts DDP does not expose `generate`, which is the reason the trainers unwrap
    before rolling out rather than calling through the wrapper.
    """
    completed = subprocess.run(
        [sys.executable, str(WORKER)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    assert completed.stdout.strip().endswith("OK"), completed.stdout


@pytest.mark.parametrize("module", [grpo_trainer, reinforce_pp_trainer])
def test_every_rl_trainer_wraps_its_policy(module):
    source = inspect.getsource(module)

    assert "wrap_policy" in source, f"{module.__name__} never wraps its policy"


@pytest.mark.parametrize("module", [grpo_trainer, reinforce_pp_trainer])
def test_generation_and_weight_reads_unwrap(module):
    """Generation, checkpointing and the vLLM weight push all read the real module, since
    DDP proxies forward and nothing else."""
    source = inspect.getsource(module)

    assert "unwrap_model(self.algorithm.policy_model)" in source
    assert "self.algorithm.policy_model.generate(" not in source
