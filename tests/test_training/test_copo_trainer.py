"""COPO's dedicated trainer.

#124 listed COPO among the algorithms that shipped a loss, a config, a factory
and a registry entry with no code path that stepped an optimizer with them.
COPOAlgorithm.run_iteration already owned the generate -> score -> pair -> CFN
-> policy-step sequence (covered in tests/test_algorithms/test_copo_algorithm.py);
what was missing was anything that fed it prompts. These cover exactly that
loop, plus the hidden-size resolution that keeps the coin-flipping network the
same width as the policy.

The policy/reward/tokenizer fixtures are reused from the algorithm's own test
module so the trainer is driven against the same shapes.
"""

import pytest
import torch
from torch.utils.data import Dataset

from thinkrl.algorithms.copo import COPOAlgorithm, COPOConfig
from thinkrl.training.copo_trainer import COPOTrainer, COPOTrainerConfig

from ..test_algorithms.test_copo_algorithm import (
    HIDDEN,
    SEQ,
    VOCAB,
    _HiddenStatePolicy,
    _RewardModel,
    _Tokenizer,
)


class PromptDataset(Dataset):
    """Mimics RLHFDataset's output shape: tokenized prompt plus the raw text,
    which is what COPO generates from."""

    def __init__(self, n=6):
        self.samples = [
            {
                "input_ids": torch.randint(1, VOCAB, (SEQ,)),
                "attention_mask": torch.ones(SEQ, dtype=torch.long),
                "prompt_text": f"prompt {i}",
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _PolicyWithConfig(_HiddenStatePolicy):
    """A transformers-style model exposes its width as `model.config.hidden_size`;
    that is where COPOTrainer reads the CFN's input width from."""

    class _Config:
        hidden_size = HIDDEN

    def __init__(self):
        super().__init__()
        self.config = self._Config()


def _trainer(dataset=None, config=None, **args_kwargs):
    args_kwargs.setdefault("num_iterations", 2)
    args_kwargs.setdefault("prompts_per_iteration", 2)
    return COPOTrainer(
        model=_PolicyWithConfig(),
        reference_model=_HiddenStatePolicy(),
        reward_model=_RewardModel(),
        tokenizer=_Tokenizer(),
        dataset=dataset if dataset is not None else PromptDataset(),
        config=config if config is not None else COPOConfig(hidden_size=HIDDEN),
        args=COPOTrainerConfig(**args_kwargs),
    )


def test_trainer_builds_its_own_algorithm():
    """Per #124: the trainer owns which algorithm it drives."""
    trainer = _trainer()

    assert isinstance(trainer.algorithm, COPOAlgorithm)


def test_hidden_size_follows_the_policy_not_the_config_default():
    """COPOConfig defaults hidden_size to 4096 (Llama-3-8B). Left alone, the CFN
    is a Linear(4096, ...) fed the policy's real hidden state, which fails inside
    the exploration bonus rather than at configuration time."""
    trainer = _trainer(config=COPOConfig())  # hidden_size left at its 4096 default

    assert trainer.config.hidden_size == HIDDEN
    assert trainer.algorithm.cfn.net[0].in_features == HIDDEN


def test_an_explicitly_wrong_hidden_size_is_corrected_and_warned_about(caplog):
    trainer = _trainer(config=COPOConfig(hidden_size=HIDDEN * 4))

    assert trainer.config.hidden_size == HIDDEN
    assert "does not match the policy model's" in caplog.text


def test_a_model_without_a_config_keeps_the_given_hidden_size():
    """A bare nn.Module has no `.config`, so there is nothing to read and the
    caller's value is the only information available."""
    trainer = COPOTrainer(
        model=_HiddenStatePolicy(),
        reward_model=_RewardModel(),
        tokenizer=_Tokenizer(),
        dataset=PromptDataset(),
        config=COPOConfig(hidden_size=HIDDEN),
    )

    assert trainer.config.hidden_size == HIDDEN


def test_tokenizer_is_required():
    with pytest.raises(ValueError, match="tokenizer"):
        COPOTrainer(model=_PolicyWithConfig(), reward_model=_RewardModel(), tokenizer=None, dataset=PromptDataset())


def test_dataset_is_required():
    with pytest.raises(ValueError, match="prompt dataset"):
        COPOTrainer(model=_PolicyWithConfig(), reward_model=_RewardModel(), tokenizer=_Tokenizer(), dataset=None)


def test_train_runs_one_iteration_per_requested_iteration():
    """run_iteration is stubbed: generation and reward scoring are the
    algorithm's own tested territory, the loop around them is this trainer's."""
    trainer = _trainer(num_iterations=5, prompts_per_iteration=2)
    seen = []

    def fake_run_iteration(prompts):
        seen.append(list(prompts))
        return {"loss": 0.5, "iteration": len(seen)}

    trainer.algorithm.run_iteration = fake_run_iteration
    metrics = trainer.train()

    assert len(seen) == 5
    assert metrics["iteration"] == 5
    # 6 prompts at 2 per iteration is 3 batches, so the loader is re-entered.
    assert all(len(prompts) == 2 for prompts in seen)
    assert all(isinstance(prompt, str) for prompts in seen for prompt in prompts)


def test_train_passes_prompt_text_not_token_ids():
    """COPOAlgorithm.generate_responses takes list[str]; handing it the collated
    input_ids would fail inside generation."""
    trainer = _trainer(num_iterations=1, prompts_per_iteration=3)
    seen = []
    trainer.algorithm.run_iteration = lambda prompts: seen.append(list(prompts)) or {}

    trainer.train()

    # Shuffled, so the identities are checked as a set, not in order.
    assert len(seen[0]) == 3
    assert set(seen[0]) <= {f"prompt {i}" for i in range(6)}
    assert all(isinstance(prompt, str) for prompt in seen[0])


def test_an_empty_dataset_is_a_clear_error():
    trainer = _trainer(dataset=PromptDataset(n=0))
    with pytest.raises(ValueError, match="nothing to generate from"):
        trainer.train()


def test_save_model_writes_the_policy_and_the_cfn(tmp_path):
    """The CFN carries the visitation statistics the exploration bonus is derived
    from, so a checkpoint without it restarts exploration from zero."""
    trainer = _trainer()
    out = trainer.save_model(str(tmp_path / "copo"))

    assert (tmp_path / "copo" / "model.pt").exists()
    assert (tmp_path / "copo" / "cfn.pt").exists()
    assert out.endswith("copo")
