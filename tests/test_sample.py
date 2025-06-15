import numpy as np
import pytest
import torch

from thinkrl import __version__


def _has_cupy_gpu():
    """Helper function to safely check for CuPy GPU availability."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except (ImportError, Exception):
        return False


def test_version():
    """Test that version is correctly set."""
    assert __version__ == "0.1.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda():
    """Test PyTorch CUDA functionality when available."""
    tensor = torch.tensor([1, 2, 3], device="cuda")
    assert tensor.device.type == "cuda"
    assert tensor.sum().item() == 6


def test_torch_cpu():
    """Test PyTorch CPU functionality (always available)."""
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6


def test_numpy_operations():
    """Test NumPy operations (CPU fallback)."""
    array = np.array([1, 2, 3])
    assert np.sum(array) == 6


# Optional CuPy test - only run if CuPy is available and has GPU
@pytest.mark.skipif(not _has_cupy_gpu(), reason="CuPy not available or no GPU detected")
def test_cupy_array():
    """Test CuPy operations when GPU is available."""
    import cupy as cp

    array = cp.array([1, 2, 3])
    assert cp.sum(array).get() == 6


def test_basic_imports():
    """Test that basic ThinkRL imports work."""
    try:
        import thinkrl

        assert hasattr(thinkrl, "__version__")
    except ImportError as e:
        pytest.fail(f"Failed to import thinkrl: {e}")


@pytest.mark.parametrize("device", ["cpu"])
def test_tensor_operations(device):
    """Test tensor operations on available devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    result = tensor * 2
    expected = torch.tensor([2.0, 4.0, 6.0], device=device)
    assert torch.allclose(result, expected)
