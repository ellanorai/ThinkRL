"""Show what the policy is actually generating during a run.

A reward of 0.0 is ambiguous on its own: the policy may be bad, the reward function may be
broken, the prompts may be malformed, the completions may be empty, or generation may be
producing something the answer parser never matches. Those have different fixes and a
scalar cannot separate them. Printing a few (prompt, completion, reward) triples can.

Renders through ``rich`` when it is installed and the output is a terminal, and falls back
to plain text otherwise, so it costs nothing in a log file or a CI job. ``rich`` arrives
with ``typer`` and is not required.
"""

from __future__ import annotations

from collections.abc import Sequence
import sys
from typing import Any


try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the plain-text path
    Console = None
    Table = None
    _RICH_AVAILABLE = False


def _as_float(value: Any) -> float:
    """Coerce a reward entry, which may be a 0-dim tensor, to a float."""
    item = getattr(value, "item", None)
    return float(item()) if callable(item) else float(value)


def truncate(text: str, max_chars: int) -> str:
    """Collapse newlines and clip to max_chars, marking the clip."""
    flattened = " ".join(str(text).split())
    if max_chars <= 0 or len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1] + "…"


class RolloutInspector:
    """Periodically print a sample of prompts, completions and rewards.

    Args:
        every: Show a sample every N steps. 0 disables the inspector entirely.
        num_samples: How many rollouts to show each time
        max_chars: Truncation width for prompts and completions
        file: Stream to write to; defaults to stdout
        use_rich: Force rich on or off. None auto-detects.
    """

    def __init__(
        self,
        every: int = 0,
        num_samples: int = 3,
        max_chars: int = 200,
        file: Any = None,
        use_rich: bool | None = None,
    ):
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.every = every
        self.num_samples = num_samples
        self.max_chars = max_chars
        self.file = file or sys.stdout
        if use_rich is None:
            use_rich = _RICH_AVAILABLE and getattr(self.file, "isatty", lambda: False)()
        self.use_rich = bool(use_rich) and _RICH_AVAILABLE

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def should_show(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.every == 0

    def maybe_show(
        self,
        step: int,
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Sequence[Any] | None = None,
    ) -> bool:
        """Print a sample if this step is due. Returns whether anything was printed."""
        if not self.should_show(step):
            return False
        self.show(step, prompts, completions, rewards)
        return True

    def show(
        self,
        step: int,
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Sequence[Any] | None = None,
    ) -> None:
        """Print a sample unconditionally."""
        rows = self._rows(prompts, completions, rewards)
        if not rows:
            print(f"[step {step}] rollout sample: nothing generated", file=self.file)
            return
        if self.use_rich:
            self._show_rich(step, rows)
        else:
            self._show_plain(step, rows)

    def _rows(self, prompts, completions, rewards) -> list[tuple[str, str, str]]:
        # Completions are the shortest of the three in the degenerate cases, and zipping to
        # the shortest keeps a mismatch from raising in the middle of a training run.
        count = min(len(prompts), len(completions), self.num_samples)
        rows = []
        for i in range(count):
            if rewards is not None and i < len(rewards):
                reward = f"{_as_float(rewards[i]):+.4f}"
            else:
                reward = "-"
            rows.append((truncate(prompts[i], self.max_chars), truncate(completions[i], self.max_chars), reward))
        return rows

    def _show_rich(self, step: int, rows) -> None:  # pragma: no cover - needs a terminal
        console = Console(file=self.file)
        table = Table(title=f"Rollout sample at step {step}", show_lines=True, title_justify="left")
        table.add_column("Reward", justify="right", style="bold", no_wrap=True)
        table.add_column("Prompt", overflow="fold")
        table.add_column("Completion", overflow="fold")
        for prompt, completion, reward in rows:
            table.add_row(reward, prompt, completion or "[dim](empty)[/dim]")
        console.print(table)

    def _show_plain(self, step: int, rows) -> None:
        print(f"\n--- rollout sample at step {step} ---", file=self.file)
        for prompt, completion, reward in rows:
            print(f"  reward {reward}", file=self.file)
            print(f"    prompt    : {prompt}", file=self.file)
            print(f"    completion: {completion or '(empty)'}", file=self.file)
        print("", file=self.file)


__all__ = ["RolloutInspector", "truncate"]
