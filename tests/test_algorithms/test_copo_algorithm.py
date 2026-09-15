"""COPO: 655 lines, the largest module in the package with no test file until #132.

It was also unreachable from any entry point until #90 exported it, so nothing had ever
run it. These cover construction, the preference loss path and the CFN exploration
bonus that distinguishes COPO from plain DPO.
"""

import pytest
import torch
import torch.nn as nn

from thinkrl.algorithms.copo import (
    CoinFlippingNetwork,
    COPOAlgorithm,
    COPOConfig,
    ReplayBuffer,
    create_copo,
)


VOCAB = 32
HIDDEN = 16
SEQ = 6


class _Output:
    """COPO reads attributes off the model output and asks for hidden states."""

    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _HiddenStatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.linear = nn.Linear(HIDDEN, VOCAB)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
        hidden = self.embedding(input_ids)
        return _Output(self.linear(hidden), hidden_states=(hidden,))


class _RewardModel(nn.Module):
    """COPO scores its own generations, so a reward model is mandatory."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.head(self.embedding(input_ids).mean(dim=1)).squeeze(-1)


class _Tokenizer:
    """COPO requires a tokenizer for RM encoding and pairing."""

    pad_token_id = 0
    eos_token_id = 1


def _algorithm():
    model = _HiddenStatePolicy()
    algorithm = COPOAlgorithm(
        policy_model=model,
        reference_model=_HiddenStatePolicy(),
        reward_model=_RewardModel(),
        config=COPOConfig(hidden_size=HIDDEN),
        tokenizer=_Tokenizer(),
    )
    return algorithm, model


def _preference_batch(batch_size: int = 2):
    torch.manual_seed(0)

    def pair():
        ids = torch.randint(1, VOCAB, (batch_size, SEQ))
        labels = ids.clone()
        labels[:, :2] = -100
        return ids, torch.ones(batch_size, SEQ, dtype=torch.long), labels

    chosen_ids, chosen_mask, chosen_labels = pair()
    rejected_ids, rejected_mask, rejected_labels = pair()

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
        "rejected_labels": rejected_labels,
    }


def test_compute_loss_returns_a_finite_loss():
    algorithm, _ = _algorithm()

    metrics = algorithm.compute_loss(_preference_batch())

    assert "loss" in metrics, f"compute_loss returned keys {sorted(metrics)}"
    loss = metrics["loss"]
    assert loss.dim() == 0
    assert torch.isfinite(loss), f"loss is {loss.item()}"


def test_the_loss_reaches_the_policy_weights():
    algorithm, model = _algorithm()

    loss = algorithm.compute_loss(_preference_batch())["loss"]
    model.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients on the policy"
    assert any(g.abs().sum() > 0 for g in grads)


def test_a_training_step_runs_and_moves_the_weights():
    """The end-to-end check the other three could not reach. training_step calls
    .backward() on whatever compute_loss returns, and COPOLoss detaches everything it
    reports, so this raised "does not require grad" until compute_loss was fixed to hand
    back the live tensor."""
    algorithm, model = _algorithm()
    before = model.linear.weight.detach().clone()

    metrics = algorithm.training_step(_preference_batch())

    assert "loss" in metrics
    assert not torch.allclose(before, model.linear.weight), "training_step left the policy unchanged"


def test_a_tokenizer_is_required_rather_than_assumed():
    """COPO needs one for RM encoding and pairing, so a missing one has to fail at
    construction rather than several hundred lines later."""
    with pytest.raises(ValueError, match="requires a tokenizer"):
        COPOAlgorithm(
            policy_model=_HiddenStatePolicy(),
            reference_model=_HiddenStatePolicy(),
            reward_model=_RewardModel(),
            config=COPOConfig(hidden_size=HIDDEN),
            tokenizer=None,
        )


def test_coin_flipping_network_maps_features_to_prediction_heads():
    cfn = CoinFlippingNetwork(input_dim=HIDDEN, hidden_dim=8, output_dim=5)

    out = cfn(torch.randn(3, HIDDEN))

    assert out.shape == (3, 5)
    assert torch.isfinite(out).all()


def test_replay_buffer_evicts_oldest_past_capacity():
    """The buffer feeds the pseudo-count estimate; unbounded growth would be a leak in
    a long run, and silently keeping everything would skew the counts."""
    buffer = ReplayBuffer(capacity=2)

    for value in range(4):
        buffer.push(torch.full((1, HIDDEN), float(value)))

    assert len(buffer) == 2


def test_factory_builds_the_algorithm():
    algorithm = create_copo(
        policy_model=_HiddenStatePolicy(),
        reference_model=_HiddenStatePolicy(),
        reward_model=_RewardModel(),
        config=COPOConfig(hidden_size=HIDDEN),
        tokenizer=_Tokenizer(),
    )

    assert isinstance(algorithm, COPOAlgorithm)
