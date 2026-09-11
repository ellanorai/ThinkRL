"""Every registered algorithm that can be constructed should have a working factory."""

import torch.nn as nn

import thinkrl.algorithms as algorithms
from thinkrl.algorithms import create_vapo
from thinkrl.algorithms.vapo import VAPOAlgorithm


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.vhead = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.emb(input_ids)
        return {"logits": self.head(hidden), "values": self.vhead(hidden).squeeze(-1)}


def test_create_vapo_returns_a_configured_algorithm():
    algorithm = create_vapo(policy_model=_TinyLM(), learning_rate=5e-6, n_epochs=3)

    assert isinstance(algorithm, VAPOAlgorithm)
    assert algorithm.config.learning_rate == 5e-6
    assert algorithm.config.n_epochs == 3


def test_create_vapo_forwards_config_kwargs():
    algorithm = create_vapo(policy_model=_TinyLM(), adaptive_gae_alpha=0.1)

    assert algorithm.config.adaptive_gae_alpha == 0.1


def test_create_vapo_is_exported():
    assert "create_vapo" in algorithms.__all__
    assert algorithms.create_vapo is create_vapo


def test_every_non_stub_registered_algorithm_has_a_factory():
    """Matches on the class the factory builds, so registry aliases do not create false gaps."""
    # KTO, ORPO and RLOO raise NotImplementedError from __init__ (see #76); they are
    # excluded here rather than silently passing.
    stubs = {"kto", "orpo", "rloo"}

    # Some modules use postponed annotations, so the return type may be a string.
    built = {
        getattr(returns, "__name__", returns)
        for name in dir(algorithms)
        if name.startswith("create_")
        for returns in [getattr(algorithms, name).__annotations__.get("return")]
    }

    registered = {cls.__name__ for name, cls in algorithms.ALGORITHMS.items() if name not in stubs}
    assert registered - built == set()
