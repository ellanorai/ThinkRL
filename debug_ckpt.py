import logging
from pathlib import Path
import shutil
import tempfile

import torch.nn as nn

from thinkrl.utils.checkpoint import load_checkpoint, save_checkpoint


def test_debug():
    logging.basicConfig(level=logging.INFO)
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = nn.Linear(10, 1)
        path = temp_dir / "extra_state.pt"

        save_checkpoint(path, model, epoch=1, custom_key="custom_value", custom_step=100)

        loaded = load_checkpoint(path, model)

        assert loaded.get("custom_key") == "custom_value", f"custom_key missing, got {loaded.get('custom_key')}"
        assert loaded.get("custom_step") == 100

    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_debug()
