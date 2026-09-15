"""Two-process gloo check for #128, run as a subprocess by test_distributed_rl.py.

Kept out of the test module on purpose. torch.multiprocessing.spawn inside a pytest
process deadlocks on macOS, and a hanging test is worse than no test, so pytest launches
this with a timeout instead and reads the verdict from stdout.

Prints OK on success, or a line starting with FAIL.
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from thinkrl.training.distributed import unwrap_model, wrap_policy  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear(x)

    def generate(self, **kwargs):
        return "generated"


def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        wrapped = wrap_policy(_Tiny())

        # Each rank sees different data, so unsynchronised gradients would differ.
        wrapped(torch.full((2, 4), float(rank + 1))).sum().backward()

        grad = unwrap_model(wrapped).linear.weight.grad.clone()
        torch.save(
            {
                "grad": grad,
                "ddp_exposes_generate": hasattr(wrapped, "generate"),
                "unwrapped_generates": unwrap_model(wrapped).generate(),
            },
            os.path.join(out_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def main() -> int:
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker,
            args=(world_size, os.path.join(tmp, "pg"), tmp),
            nprocs=world_size,
            join=True,
        )
        results = [torch.load(os.path.join(tmp, f"rank{r}.pt"), weights_only=False) for r in range(world_size)]

    if not torch.allclose(results[0]["grad"], results[1]["grad"]):
        print("FAIL: gradients differ across ranks, so each process is training its own model")
        return 1

    # Local gradients would be 2.0 and 4.0 per element; DDP averages them to 3.0.
    if not torch.allclose(results[0]["grad"], torch.full((4, 4), 3.0)):
        print(f"FAIL: expected the averaged gradient 3.0, got {results[0]['grad'][0, 0].item()}")
        return 1

    if results[0]["ddp_exposes_generate"]:
        print("FAIL: DDP exposed generate, so the trainers' unwrap is dead code")
        return 1

    if results[0]["unwrapped_generates"] != "generated":
        print("FAIL: unwrapped module did not generate")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
