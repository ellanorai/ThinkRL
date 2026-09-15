"""#124: eight registered algorithms shipped with no code path that trains them.

PPO, DAPO, VAPO, PRIME and Dr.GRPO each have a loss, a config dataclass, a factory and a
registry entry, and nothing stepped an optimizer with any of them. All five already
implement `train_on_rollout`, so the cheapest honest route, as the issue put it, is to let
the existing rollout loop drive any of them rather than write four more trainers.

DPO, IPO and COPO are the offline preference family and are a different batch shape; they
are not covered here.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms import ALGORITHMS
from thinkrl.training.grpo_trainer import GRPOTrainer


ROLLOUT_FAMILY = ["ppo", "dapo", "vapo", "prime", "dr_grpo"]


class _TinyPolicy(nn.Module):
    def __init__(self, vocab=32, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, vocab)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        # PPO and VAPO are actor-critic, so the policy has to emit values as well.
        return {"logits": self.linear(hidden), "values": self.value_head(hidden).squeeze(-1)}


@pytest.mark.parametrize("name", ROLLOUT_FAMILY)
def test_the_rollout_family_implements_the_contract_the_loop_needs(name):
    """train_on_rollout is what GRPOTrainer drives. If an algorithm has it, the loop can
    run it, which is the whole basis for generalising rather than duplicating."""
    algorithm = ALGORITHMS[name]

    assert hasattr(algorithm, "train_on_rollout"), f"{name} cannot be driven by the rollout loop"


def test_the_trainer_accepts_a_prebuilt_algorithm():
    params = inspect.signature(GRPOTrainer.__init__).parameters

    assert "algorithm" in params
    assert params["algorithm"].default is None, "must stay opt-in so existing callers are unaffected"


def test_passing_both_an_algorithm_and_a_model_is_refused():
    """Silently ignoring one of them would leave the caller training a model they did not
    pass, which is the kind of thing that only shows up in the loss curve."""
    from thinkrl.algorithms.ppo import create_ppo

    algorithm = create_ppo(policy_model=_TinyPolicy())

    with pytest.raises(ValueError, match="not both"):
        GRPOTrainer(model=_TinyPolicy(), algorithm=algorithm, tokenizer=None, dataset=None, reward_fn=None)


def test_group_size_is_read_from_whichever_config_is_in_play():
    """A non-GRPO algorithm may not define group_size at all, and the old line read
    self.config.group_size unconditionally."""
    source = inspect.getsource(GRPOTrainer.__init__)

    assert 'getattr(source_config, "group_size", None)' in source


@pytest.mark.parametrize("name", ROLLOUT_FAMILY)
def test_each_algorithm_still_constructs(name):
    """Guards the claim that these five are constructible today, since #132 found COPO was
    not and nobody had noticed."""
    algorithm_cls = ALGORITHMS[name]

    algorithm = algorithm_cls(policy_model=_TinyPolicy())

    assert algorithm.policy_model is not None


def test_a_non_grpo_algorithm_reaches_train_on_rollout():
    """The end-to-end claim: hand the loop a PPO algorithm and it trains with it.

    Driven directly rather than through trainer.train(), which needs a tokenizer, a
    dataset and generation; the point here is that the loop's call into the algorithm is
    algorithm-agnostic.
    """
    from thinkrl.algorithms.ppo import create_ppo

    torch.manual_seed(0)
    policy = _TinyPolicy()
    algorithm = create_ppo(policy_model=policy)

    batch = {
        "input_ids": torch.randint(1, 32, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "labels": torch.randint(1, 32, (2, 6)),
        "old_log_probs": torch.randn(2, 6) * 0.1,
        "rewards": torch.randn(2),
        "advantages": torch.randn(2),
    }
    batch["labels"][:, :2] = -100

    before = policy.linear.weight.detach().clone()
    metrics = algorithm.train_on_rollout(batch)

    assert metrics, "train_on_rollout returned nothing"
    assert not torch.allclose(before, policy.linear.weight), "the policy did not move"
