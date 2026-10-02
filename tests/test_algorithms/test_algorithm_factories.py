"""Every registered algorithm that can be constructed should have a working factory."""

import torch.nn as nn

import thinkrl.algorithms as algorithms
from thinkrl.algorithms import (
    create_dapo,
    create_dpo,
    create_grpo,
    create_ipo,
    create_ppo,
    create_prime,
    create_reinforce,
    create_reinforce_pp,
    create_star,
    create_vapo,
)
from thinkrl.algorithms.dapo import DAPOAlgorithm, DAPOConfig
from thinkrl.algorithms.dpo import DPOAlgorithm, DPOConfig
from thinkrl.algorithms.grpo import GRPOAlgorithm, GRPOConfig
from thinkrl.algorithms.ipo import IPOAlgorithm, IPOConfig
from thinkrl.algorithms.ppo import PPOAlgorithm, PPOConfig
from thinkrl.algorithms.prime import PRIMEAlgorithm, PRIMEConfig
from thinkrl.algorithms.reinforce import REINFORCEAlgorithm, REINFORCEConfig
from thinkrl.algorithms.reinforce_pp import REINFORCEPPAlgorithm, REINFORCEPPConfig
from thinkrl.algorithms.star import STaRAlgorithm, STaRConfig
from thinkrl.algorithms.vapo import VAPOAlgorithm, VAPOConfig


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


def test_create_ppo_accepts_a_prebuilt_config():
    config = PPOConfig(learning_rate=9e-5)
    algorithm = create_ppo(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, PPOAlgorithm)
    assert algorithm.config is config


def test_create_dapo_accepts_ref_model_and_a_prebuilt_config():
    policy = _TinyLM()
    ref = _TinyLM()
    config = DAPOConfig(learning_rate=2e-6)
    algorithm = create_dapo(policy_model=policy, ref_model=ref, config=config)

    assert isinstance(algorithm, DAPOAlgorithm)
    assert algorithm.ref_model is ref
    assert algorithm.config is config


def test_create_vapo_accepts_a_prebuilt_config():
    config = VAPOConfig(learning_rate=4e-6)
    algorithm = create_vapo(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, VAPOAlgorithm)
    assert algorithm.config is config


def test_create_grpo_accepts_a_prebuilt_config():
    config = GRPOConfig(learning_rate=1e-6)
    algorithm = create_grpo(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, GRPOAlgorithm)
    assert algorithm.config is config


def test_create_prime_accepts_a_prebuilt_config():
    config = PRIMEConfig(beta=0.1)
    algorithm = create_prime(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, PRIMEAlgorithm)
    assert algorithm.config is config


def test_create_dpo_accepts_a_prebuilt_config():
    config = DPOConfig(learning_rate=3e-6)
    algorithm = create_dpo(policy_model=_TinyLM(), ref_model=_TinyLM(), config=config)

    assert isinstance(algorithm, DPOAlgorithm)
    assert algorithm.config is config


def test_create_ipo_accepts_a_prebuilt_config():
    config = IPOConfig(learning_rate=3e-6)
    algorithm = create_ipo(policy_model=_TinyLM(), ref_model=_TinyLM(), config=config)

    assert isinstance(algorithm, IPOAlgorithm)
    assert algorithm.config is config


def test_create_star_accepts_a_prebuilt_config():
    config = STaRConfig()
    algorithm = create_star(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, STaRAlgorithm)
    assert algorithm.config is config


def test_create_reinforce_accepts_a_prebuilt_config():
    config = REINFORCEConfig(learning_rate=2e-5)
    algorithm = create_reinforce(policy_model=_TinyLM(), config=config)

    assert isinstance(algorithm, REINFORCEAlgorithm)
    assert algorithm.config is config


def test_create_reinforce_pp_accepts_a_prebuilt_config():
    config = REINFORCEPPConfig(learning_rate=2e-6)
    algorithm = create_reinforce_pp(policy_model=_TinyLM(), ref_model=_TinyLM(), config=config)

    assert isinstance(algorithm, REINFORCEPPAlgorithm)
    assert algorithm.config is config


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
