"""Shared fixtures for the per-algorithm smoke tests added for #132.

Each uncovered module gets the same treatment: build the config, build the algorithm,
run one `compute_loss` on a fixed batch, and assert the loss is a finite scalar that
carries a gradient. That is deliberately shallow, but it is the depth that catches the
class of defect #100 was, where PRIME silently returned a different quantity for
`gamma != 1.0` and nothing noticed for 443 lines.

Per-algorithm maths assertions belong in the focused files next to this one
(test_prime_advantages.py, test_ppo_returns.py and so on).
"""

from __future__ import annotations

import torch
import torch.nn as nn


VOCAB = 32
SEQ = 6
PROMPT_LEN = 2


class TinyPolicy(nn.Module):
    """Smallest model that still exercises the real forward and backward path."""

    def __init__(self, vocab: int = VOCAB, hidden: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"logits": self.linear(self.embedding(input_ids))}


def make_batch(batch_size: int = 4, seq: int = SEQ, seed: int = 0) -> dict[str, torch.Tensor]:
    """A batch shaped like a rollout: a two-token prompt then generated tokens.

    Labels carry -100 over the prompt, which is the convention every algorithm here
    masks on.
    """
    torch.manual_seed(seed)

    input_ids = torch.randint(1, VOCAB, (batch_size, seq))
    attention_mask = torch.ones(batch_size, seq, dtype=torch.long)

    labels = input_ids.clone()
    labels[:, :PROMPT_LEN] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "rewards": torch.randn(batch_size),
        "old_log_probs": torch.randn(batch_size, seq) * 0.1,
        "advantages": torch.randn(batch_size),
    }


def assert_trainable_loss(loss: torch.Tensor, model: nn.Module) -> None:
    """A loss is only useful if it is finite, scalar, and actually reaches the weights."""
    assert isinstance(loss, torch.Tensor), f"loss is {type(loss).__name__}, not a tensor"
    assert loss.dim() == 0, f"loss has shape {tuple(loss.shape)}, expected a scalar"
    assert torch.isfinite(loss), f"loss is {loss.item()}"
    assert loss.requires_grad, "loss does not require grad, so no training can happen"

    model.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients on any parameter"
    assert any(g.abs().sum() > 0 for g in grads), "every gradient is exactly zero"
    assert all(torch.isfinite(g).all() for g in grads), "a gradient is inf or nan"
