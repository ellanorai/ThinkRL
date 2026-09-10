"""Regression test: float16 weights must not be handed to an Adam-family optimizer.

Adam keeps its moments in the parameter dtype. In float16 the second moment underflows and
eps=1e-8 is exactly 0.0, so one step turns every weight into nan while the reported loss
stays finite. The failure only surfaces later, as `probability tensor contains inf, nan`
out of generate().
"""

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.grpo import GRPOAlgorithm
from thinkrl.training.mixed_precision import check_optimizable_dtype


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.head(self.emb(input_ids))}


def test_the_underlying_failure_is_real():
    """Pins the mechanism the guard exists for, independent of any ThinkRL code."""
    param = nn.Parameter(torch.randn(4, 4, dtype=torch.float16))
    param.grad = torch.full_like(param, 1e-3)

    torch.optim.AdamW([param], lr=1e-6, eps=1e-8).step()

    assert not torch.isfinite(param).all(), "float16 AdamW no longer breaks; the guard can go"
    assert float(torch.tensor(1e-8, dtype=torch.float16)) == 0.0


def test_float32_and_bfloat16_pass():
    for dtype in (torch.float32, torch.bfloat16):
        check_optimizable_dtype(_TinyLM().to(dtype))


def test_float16_is_rejected():
    with pytest.raises(ValueError, match="float16"):
        check_optimizable_dtype(_TinyLM().half())


def test_the_error_says_how_to_fix_it():
    with pytest.raises(ValueError) as excinfo:
        check_optimizable_dtype(_TinyLM().half())

    message = str(excinfo.value)
    assert "float32 or bfloat16" in message
    assert "0.0 in float16" in message
    assert "nan" in message


def test_frozen_float16_parameters_are_ignored():
    """A frozen fp16 reference model is fine; the optimizer never touches it."""
    model = _TinyLM().half()
    for param in model.parameters():
        param.requires_grad = False

    check_optimizable_dtype(model)


def test_algorithm_construction_rejects_a_float16_policy():
    with pytest.raises(ValueError, match="float16"):
        GRPOAlgorithm(policy_model=_TinyLM().half())


def test_algorithm_construction_accepts_float32():
    assert GRPOAlgorithm(policy_model=_TinyLM()) is not None


def test_a_caller_supplied_optimizer_is_left_alone():
    """The guard covers the optimizer the library builds, not one the caller brought."""
    model = _TinyLM().half()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    assert GRPOAlgorithm(policy_model=model, optimizer=optimizer) is not None
