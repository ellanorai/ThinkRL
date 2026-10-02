"""`thinkrl sft` used to echo its flags and then call _not_implemented (#193):
every flag below reached nothing. These assert each one actually reaches the
trainer/dataset/model call, not just that the command exits zero."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from thinkrl.cli.main import app


runner = CliRunner()

BASE = ["sft", "--model", "gpt2", "--dataset", "fake_dataset"]


@pytest.fixture
def mock_trainer():
    with patch("thinkrl.training.sft_trainer.SFTTrainer") as mock:
        mock.return_value.train.return_value = {"global_step": 1}
        yield mock


@pytest.fixture
def mock_get_model():
    with patch("thinkrl.models.loader.get_model") as mock:
        yield mock


@pytest.fixture
def mock_dataset():
    with patch("thinkrl.data.datasets.RLHFDataset") as mock:
        mock.return_value.__len__.return_value = 100
        yield mock


@pytest.fixture
def mock_tokenizer():
    with patch("transformers.AutoTokenizer") as mock:
        mock.from_pretrained.return_value.pad_token = "<eos>"
        mock.from_pretrained.return_value.pad_token_id = 0
        yield mock


def test_sft_reaches_the_trainer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, BASE)

    assert result.exit_code == 0, result.output
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_sft_builds_the_dataset_with_the_given_columns(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--prompt-column", "instruction", "--response-column", "output"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_dataset.call_args
    assert kwargs["prompt_column"] == "instruction"
    assert kwargs["response_column"] == "output"


def test_sft_passes_lora_rank_to_get_model(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--lora-r", "8"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_get_model.call_args
    assert kwargs["lora_rank"] == 8


def test_sft_resume_reaches_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--resume", "./sft_output/checkpoint-10"])

    assert result.exit_code == 0, result.output
    _, kwargs = mock_trainer.return_value.train.call_args
    assert kwargs["resume_from_checkpoint"] == "./sft_output/checkpoint-10"


def test_sft_dry_run_does_not_train(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_trainer.return_value.train.assert_not_called()


def test_sft_push_to_hub_pushes_model_and_tokenizer(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    result = runner.invoke(app, [*BASE, "--push-to-hub", "me/my-sft-model"])

    assert result.exit_code == 0, result.output
    mock_get_model.return_value.push_to_hub.assert_called_once_with("me/my-sft-model")
    mock_tokenizer.from_pretrained.return_value.push_to_hub.assert_called_once_with("me/my-sft-model")


def test_sft_collator_masks_the_prompt_out_of_labels(mock_trainer, mock_get_model, mock_dataset, mock_tokenizer):
    """The actual correctness bug this PR exists to avoid: without masking, SFT
    trains on reproducing the prompt as well as the response."""
    import torch

    runner.invoke(app, BASE)

    collator = mock_trainer.call_args.kwargs["data_collator"]
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "prompt_length": 3,
        }
    ]
    out = collator(batch)

    assert out["labels"][0, 0].item() == -100
    assert out["labels"][0, 1].item() == -100
    assert out["labels"][0, 2].item() == -100
    assert out["labels"][0, 3].item() == 4
    assert out["labels"][0, 4].item() == 5
