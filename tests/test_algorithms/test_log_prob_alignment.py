"""`get_log_probs` must return log probabilities aligned with `labels`.

Index `t` holds `log p(labels[t])`. That is the contract the docstring always described
("Log probabilities [B, S] with 0.0 at masked positions") and the one every caller
assumes when it masks the result with `(labels != -100)`, which is unshifted.

The old implementation appended the pad column instead of prepending it, so index `t`
held `log p(labels[t+1])`. Masking that with unshifted labels dropped the first
generated token and counted the appended zero in its place. See #84.
"""

import pytest
import torch

from thinkrl.algorithms.base import BaseRLHFAlgorithm


# prompt of two tokens, then four generated tokens
LABELS = torch.tensor([[-100, -100, 2, 3, 4, 1]])


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(1, 6, 8)


def _expected_per_token(logits: torch.Tensor, labels: torch.Tensor) -> dict[int, float]:
    """log p(labels[t]) computed independently of the implementation under test."""
    log_probs = torch.log_softmax(logits, dim=-1)
    out = {}
    for t in range(1, labels.size(1)):
        if labels[0, t].item() != -100:
            # logits[t - 1] is the distribution that predicts the token at position t
            out[t] = log_probs[0, t - 1, labels[0, t]].item()
    return out


def test_index_t_holds_the_log_prob_of_labels_t(logits):
    result = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)

    assert result.shape == LABELS.shape
    for t, expected in _expected_per_token(logits, LABELS).items():
        assert result[0, t].item() == pytest.approx(expected, abs=1e-6), f"position {t} misaligned"


def test_position_zero_is_zero(logits):
    """No logit predicts the first token, so there is nothing to report there."""
    result = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)

    assert result[0, 0].item() == 0.0


def test_unshifted_mask_selects_exactly_the_generated_tokens(logits):
    """The pattern used at 14 call sites across 11 algorithm files."""
    result = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)
    mask = (LABELS != -100).float()

    selected = (result * mask).sum().item()
    expected = sum(_expected_per_token(logits, LABELS).values())

    assert selected == pytest.approx(expected, abs=1e-6)


def test_the_first_generated_token_is_not_dropped(logits):
    """The specific regression: its log prob used to land outside the mask, and it is
    typically the largest magnitude of the completion because it is least predictable."""
    result = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)
    mask = (LABELS != -100).float()

    first_generated = _expected_per_token(logits, LABELS)[2]
    assert (result * mask)[0, 2].item() == pytest.approx(first_generated, abs=1e-6)
    assert result[0, 2].item() != 0.0


def test_the_sum_over_all_positions_is_unchanged(logits):
    """ipo.py sums the whole row rather than masking it, so re-aligning must not move
    the total. Guards the caller that was already correct."""
    result = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)

    expected = sum(_expected_per_token(logits, LABELS).values())
    assert result.sum().item() == pytest.approx(expected, abs=1e-6)


def test_actor_precomputed_tuple_has_the_same_alignment(logits):
    """The Actor path hands back gathered log probs of width S-1 and takes the same
    padding decision, so it has to agree with the logits path."""
    from_logits = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    gathered = torch.gather(log_probs, -1, LABELS[:, 1:].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered * (LABELS[:, 1:] != -100).float()

    from_tuple = BaseRLHFAlgorithm.get_log_probs(None, (gathered,), LABELS)

    assert torch.allclose(from_logits, from_tuple, atol=1e-6)


def test_dapo_agrees_with_the_base_implementation(logits):
    """DAPO carried a private copy of this function with the same defect, so a fix to
    the base alone left it behind."""
    from thinkrl.algorithms.dapo import DAPOAlgorithm

    base = BaseRLHFAlgorithm.get_log_probs(None, logits, LABELS)
    dapo = DAPOAlgorithm.get_log_probs(None, logits, LABELS)

    assert torch.allclose(base, dapo, atol=1e-6)
