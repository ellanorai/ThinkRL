"""DIST-1: thinkrl.utils.get_strategy must not silently return None."""
import pytest
from thinkrl.utils import get_strategy


def test_get_strategy_fails_loud_not_silent_none():
    with pytest.raises(NotImplementedError, match="thinkrl.distributed.get_strategy"):
        get_strategy(None)
