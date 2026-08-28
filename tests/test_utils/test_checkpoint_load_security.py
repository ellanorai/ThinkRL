"""Checkpoint loading must not unpickle arbitrary objects.

Every torch.load call in the loading path used weights_only=False, so a crafted
model.pt executed its pickle __reduce__ payload as soon as the checkpoint was
loaded. These tests drive the real CheckpointManager API, not torch.load.
"""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from thinkrl.utils.checkpoint import CheckpointManager, load_checkpoint


class _Payload:
    """Innocuous-looking object whose unpickling runs a shell command."""

    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _write_malicious_checkpoint(directory: Path, marker: Path) -> Path:
    model_path = directory / "model.pt"
    torch.save({"payload": _Payload(str(marker))}, model_path)
    return model_path


def test_manager_load_does_not_execute_pickle_payload(tmp_path):
    marker = tmp_path / "PWNED"
    ckpt_dir = tmp_path / "checkpoint-1"
    ckpt_dir.mkdir()
    _write_malicious_checkpoint(ckpt_dir, marker)

    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    with pytest.raises(Exception):
        manager.load_checkpoint(ckpt_dir, model=nn.Linear(2, 2))

    assert not marker.exists(), "pickle payload executed during load_checkpoint"


def test_module_level_load_does_not_execute_pickle_payload(tmp_path):
    marker = tmp_path / "PWNED"
    model_path = _write_malicious_checkpoint(tmp_path, marker)

    with pytest.raises(Exception):
        load_checkpoint(model_path, model=nn.Linear(2, 2))

    assert not marker.exists(), "pickle payload executed during load_checkpoint()"


def test_optimizer_state_load_does_not_execute_pickle_payload(tmp_path):
    """The optimizer path is a separate torch.load call and needs its own pin."""
    marker = tmp_path / "PWNED_OPTIM"
    ckpt_dir = tmp_path / "checkpoint-1"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save({"payload": _Payload(str(marker))}, ckpt_dir / "optimizer.pt")

    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(Exception):
        manager.load_checkpoint(ckpt_dir, model=model, optimizer=optimizer)

    assert not marker.exists(), "pickle payload executed while loading optimizer state"


def test_no_load_site_reintroduces_weights_only_false():
    for filename in (
        "thinkrl/utils/checkpoint.py",
        "thinkrl/models/reward_model.py",
        "thinkrl/models/critic.py",
    ):
        assert "weights_only=False" not in Path(filename).read_text(), filename


def test_round_trip_still_works(tmp_path):
    """weights_only=True must not break ordinary checkpoints."""
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    path = manager.save_checkpoint(model=model, optimizer=optimizer, epoch=1, step=10)

    restored = nn.Linear(4, 3)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    manager.load_checkpoint(path, model=restored, optimizer=restored_optimizer)

    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
