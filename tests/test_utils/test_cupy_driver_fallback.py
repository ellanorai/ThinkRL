"""cupy imports cleanly on a host that has the package but no usable driver.

The failure only appears on the call that actually reaches the CUDA runtime, which used
to be somewhere inside a metric, so CI saw `CUDARuntimeError: cudaErrorInsufficientDriver`
out of `compute_perplexity` rather than a tidy fallback. `metrics.py` probes with a real
one-element ufunc at import; this pins that, since no machine running this suite has cupy
installed and the branch would otherwise never execute.

Parametrised over where the driver error surfaces because guessing that narrowly was the
bug twice over: `getDeviceCount()` returns cleanly on the runner, and probing it alone
shipped a fix that changed nothing.
"""

import importlib
import sys
import types

import pytest


class _CUDARuntimeError(Exception):
    pass


def _no_driver(*args, **kwargs):
    raise _CUDARuntimeError("cudaErrorInsufficientDriver: CUDA driver version is insufficient")


def _fake_cupy(*, failing: str) -> types.ModuleType:
    """A cupy usable except for one operation, so each probe is pinned independently."""
    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0))
    fake.zeros = lambda n: [0.0] * n
    fake.exp = lambda a: a

    if failing in ("getDeviceCount", "getDevice"):
        setattr(fake.cuda.runtime, failing, _no_driver)
    else:
        setattr(fake, failing, _no_driver)
    return fake


@pytest.mark.parametrize("failing", ["getDeviceCount", "getDevice", "zeros", "exp"])
def test_cupy_without_a_usable_driver_falls_back(monkeypatch, failing):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(failing=failing))

    import thinkrl.utils.metrics as metrics

    reloaded = importlib.reload(metrics)
    try:
        assert reloaded._CUPY_AVAILABLE is False
        assert reloaded._CUPY_SCIPY_AVAILABLE is False
        assert reloaded.cp is None
    finally:
        # Put the real module back before any later test imports it.
        monkeypatch.undo()
        importlib.reload(metrics)


def test_the_probe_does_not_break_the_ordinary_import():
    """Guards against the probe itself becoming the failure it was added to prevent."""
    import thinkrl.utils.metrics as metrics

    assert metrics.compute_perplexity is not None


def test_the_test_suite_reads_availability_from_the_library():
    """tests/test_utils/test_metrics.py kept a private cupy import that only caught
    (ImportError, OSError), so it selected cupy while the library had fallen back to
    numpy. One source of truth, or the two disagree again."""
    import tests.test_utils.test_metrics as test_metrics
    from thinkrl.utils import metrics

    assert test_metrics._CUPY_AVAILABLE is metrics._CUPY_AVAILABLE
    assert test_metrics.cp is metrics.cp
