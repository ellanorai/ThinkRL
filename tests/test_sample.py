import pytest
import torch
import cupy as cp
import numpy as np
from thinkrl import __version__

def test_version():
    assert __version__ == "0.1.0"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    assert tensor.device.type == "cuda"

@pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() == 0, reason="No GPU available")
def test_cupy_array():
    array = cp.array([1, 2, 3])
    assert cp.sum(array).get() == 6

def test_cpu_fallback():
    array = np.array([1, 2, 3])
    assert np.sum(array) == 6