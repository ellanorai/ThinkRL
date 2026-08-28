"""Checkpoint cleanup must never delete a path outside checkpoint_dir.

`_load_metadata` restores each entry's path verbatim from
checkpoint_registry.json, and `_cleanup_checkpoints` shutil.rmtree's the
entries that fall outside max_checkpoints. A registry that names any other
directory therefore deletes it recursively on the next save.
"""

import json

import torch
from torch import nn

from thinkrl.utils.checkpoint import CheckpointManager


def _write_registry(ckpt_dir, entries):
    (ckpt_dir / "checkpoint_registry.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "path": str(path),
                        "name": name,
                        "epoch": 0,
                        "step": step,
                        "metrics": {},
                        "timestamp": timestamp,
                    }
                    for path, name, step, timestamp in entries
                ],
                "best_checkpoint": None,
            }
        )
    )


def test_save_does_not_delete_directory_named_by_registry(tmp_path):
    """The full attack path: hostile registry on disk, then an ordinary save."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("keep me")

    _write_registry(ckpt_dir, [(victim, "evil", 0, "2000-01-01T00:00:00")])

    manager = CheckpointManager(checkpoint_dir=ckpt_dir, max_checkpoints=1)
    assert manager.checkpoints, "registry was not loaded, test would pass vacuously"

    manager.save_checkpoint(model=nn.Linear(2, 2), epoch=1, step=2)

    assert victim.exists(), "directory outside checkpoint_dir was deleted"
    assert (victim / "important.txt").read_text() == "keep me"


def test_cleanup_still_removes_real_old_checkpoints(tmp_path):
    """The guard must not stop ordinary rotation."""
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir, max_checkpoints=1)
    model = nn.Linear(2, 2)

    first = manager.save_checkpoint(model=model, epoch=1, step=1)
    manager.save_checkpoint(model=model, epoch=2, step=2)

    assert not first.exists(), "old checkpoint inside checkpoint_dir was not rotated out"
    assert len(manager.checkpoints) == 1


def test_relative_parent_escape_is_refused(tmp_path):
    """A path that only escapes after resolution must also be refused."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()

    escaping = ckpt_dir / ".." / "victim"
    _write_registry(ckpt_dir, [(escaping, "evil", 0, "2000-01-01T00:00:00")])

    manager = CheckpointManager(checkpoint_dir=ckpt_dir, max_checkpoints=1)
    manager.save_checkpoint(model=nn.Linear(2, 2), epoch=1, step=2)

    assert victim.exists(), "path escaping via .. was deleted"
