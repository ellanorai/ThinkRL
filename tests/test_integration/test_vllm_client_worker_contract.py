"""Regression test: every endpoint the vLLM client calls must exist on the worker.

`VLLMClient.check_weights` posted to `/check_weights`, which the worker never defined, so
every REINFORCE++ run with `use_vllm=True` died at trainer setup with a 404. The two files
cannot be imported without vllm and uvicorn installed, so the contract is checked against
the sources, which is also what makes the test runnable in CI without a GPU.
"""

import re
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[2] / "thinkrl" / "integration"
CLIENT = (PACKAGE / "vllm_client.py").read_text()
WORKER = (PACKAGE / "vllm_worker.py").read_text()

# f"{self.url}/health" and friends
REQUESTED = sorted(set(re.findall(r'\{self\.url\}(/[a-z_]+)"', CLIENT)))
REGISTERED = sorted(set(re.findall(r'@app\.(?:get|post)\("(/[a-z_]+)"\)', WORKER)))


def test_the_sources_were_actually_parsed():
    assert REQUESTED, "no client requests found; the extraction pattern has drifted"
    assert REGISTERED, "no worker routes found; the extraction pattern has drifted"


@pytest.mark.parametrize("path", REQUESTED)
def test_every_requested_path_is_registered(path):
    assert path in REGISTERED, f"vllm_client.py posts to {path}, which vllm_worker.py does not define"


def test_check_weights_specifically_is_registered():
    """The endpoint this test file exists for."""
    assert "/check_weights" in REGISTERED


def test_check_weights_returns_the_shape_the_client_parses():
    """The client reads result['status'] and result['details']."""
    route = WORKER[WORKER.index('@app.post("/check_weights")') : WORKER.index('@app.post("/update_weights")')]

    assert '"status": "ok"' in route
    assert '"status": "mismatch"' in route
    assert '"details"' in route
