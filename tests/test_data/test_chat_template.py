"""RL prompts must go through the tokenizer's chat template.

Nothing in thinkrl/cli or thinkrl/training called apply_chat_template, so an
instruct checkpoint was rolled out on bare prompt strings: generation still
worked, rewards were still computed, and the policy was optimized off the
distribution it was aligned to.
"""

import pytest
import torch

from thinkrl.data.datasets import RLHFDataset


SAMPLES = [{"prompt": "What is 2+2?", "answer": "4", "response": "It is 4."}]


class ChatTokenizer:
    """Mock tokenizer that renders a recognisable chat template."""

    chat_template = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = "<EOS>"
        self.eos_token_id = 1
        self.last_add_special_tokens = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered

    def __call__(self, text, max_length=None, padding=False, truncation=False, return_tensors=None, **kwargs):
        self.last_add_special_tokens = kwargs.get("add_special_tokens")
        ids = [len(tok) for tok in text.split()] or [1]
        if truncation and max_length:
            ids = ids[:max_length]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class PlainTokenizer(ChatTokenizer):
    """Base model: no template defined."""

    chat_template = None


class BrokenTemplateTokenizer(ChatTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("template does not support the system role")


def _dataset(tokenizer, **kwargs):
    return RLHFDataset(dataset_name_or_path=SAMPLES, tokenizer=tokenizer, **kwargs)


def test_prompt_is_rendered_with_generation_prompt():
    sample = _dataset(ChatTokenizer())[0]
    assert sample["prompt_text"] == "<|user|>What is 2+2?<|assistant|>"


def test_system_prompt_becomes_a_system_message():
    sample = _dataset(ChatTokenizer(), system_prompt="Think step by step.")[0]
    assert sample["prompt_text"] == "<|system|>Think step by step.<|user|>What is 2+2?<|assistant|>"
    assert "Think step by step.\n\nWhat is 2+2?" not in sample["prompt_text"]


def test_rendered_text_is_tokenized_without_extra_special_tokens():
    """A rendered template already carries BOS and role markers."""
    tokenizer = ChatTokenizer()
    _dataset(tokenizer)[0]
    assert tokenizer.last_add_special_tokens is False


def test_base_model_tokenizer_is_untouched():
    tokenizer = PlainTokenizer()
    sample = _dataset(tokenizer)[0]
    assert sample["prompt_text"] == "What is 2+2?"
    assert tokenizer.last_add_special_tokens is True


def test_flag_disables_templating():
    sample = _dataset(ChatTokenizer(), apply_chat_template=False)[0]
    assert sample["prompt_text"] == "What is 2+2?"


def test_sft_mode_puts_the_response_in_an_assistant_turn():
    dataset = _dataset(ChatTokenizer(), response_column="response")
    sample = dataset[0]
    _, text, templated = dataset._render("What is 2+2?", "It is 4.")
    assert templated
    assert text == "<|user|>What is 2+2?<|assistant|>It is 4."
    assert sample["prompt_text"] == "<|user|>What is 2+2?<|assistant|>"


def test_failing_template_falls_back_instead_of_crashing(caplog):
    dataset = _dataset(BrokenTemplateTokenizer(), system_prompt="sys")
    sample = dataset[0]
    assert sample["prompt_text"] == "sys\n\nWhat is 2+2?"
    assert dataset._chat_template_failed is True


def test_target_column_survives_templating():
    sample = _dataset(ChatTokenizer())[0]
    assert sample["target"] == "4"
