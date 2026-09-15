"""cupy imports cleanly on a host that has the package but no usable driver.

The failure only appears on the first call that touches the CUDA runtime, which used to
be somewhere inside a metric, so CI saw `CUDARuntimeError: cudaErrorInsufficientDriver`
out of `compute_perplexity` rather than a tidy fallback. `metrics.py` now probes the
runtime at import time; this pins that, since the machines that run this suite do not
have cupy installed and would otherwise never exercise the branch.
"""

import importlib
import sys
import types


def test_cupy_without_a_usable_driver_falls_back(monkeypatch):
    class _CUDARuntimeError(Exception):
        pass

    def _no_driver():
        raise _CUDARuntimeError("cudaErrorInsufficientDriver: CUDA driver version is insufficient")

    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=_no_driver))
    monkeypatch.setitem(sys.modules, "cupy", fake)

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
