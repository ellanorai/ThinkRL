"""CFG-2/OBS-2: from_dict must not crash with AttributeError on empty/null-section configs."""
import pytest
from thinkrl.config.base import ThinkRLConfig


def test_empty_config_raises_clear_error():
    with pytest.raises(ValueError, match="empty"):
        ThinkRLConfig.from_dict(None)


def test_null_sections_default_to_empty():
    # `model:` / `distributed:` present but null must not crash
    cfg = ThinkRLConfig.from_dict({"model": None, "distributed": None, "algorithm": None})
    assert cfg.model is not None
    assert cfg.distributed is not None
