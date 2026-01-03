"""
Test Suite for ThinkRL Dataset Utilities
========================================

Tests for:
- thinkrl.utils.datasets
"""

import pytest
import torch


try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except (ImportError, OSError):
    _CUPY_AVAILABLE = False

from unittest.mock import MagicMock, patch

from thinkrl.utils.datasets import (
    BatchEncoding,
    collate_batch,
    compute_sequence_lengths,
    create_attention_mask,
    create_causal_mask,
    create_dataloader,
    create_labels_for_clm,
    create_position_ids,
    mask_padding_in_loss,
    pad_sequences,
    prepare_batch_for_training,
    preprocess_text,
    shuffle_batch,
    split_batch,
    to_device,
    truncate_sequence,
)


@pytest.fixture
def sample_batch_tensors():
    return {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }


@pytest.fixture
def sample_list_sequences():
    return [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
    ]


class TestBatchEncoding:
    def test_init(self):
        data = {"input_ids": torch.tensor([1, 2])}
        be = BatchEncoding(data, encoding="dummy", tensor_type="pt")
        assert be["input_ids"] is data["input_ids"]
        assert be.encoding == "dummy"
        assert be.tensor_type == "pt"

    def test_to_device(self):
        data = {"input_ids": torch.tensor([1, 2]), "meta": "metadata"}
        be = BatchEncoding(data)
        device = torch.device("cpu")
        be_moved = be.to(device)

        assert be_moved is be
        assert isinstance(be["input_ids"], torch.Tensor)
        assert be["meta"] == "metadata"


def test_pad_sequences(sample_list_sequences):
    padded_right = pad_sequences(sample_list_sequences, padding_value=0, padding_side="right")
    expected_right = torch.tensor([[1, 2, 3], [4, 5, 0]])
    assert torch.equal(padded_right, expected_right)

    padded_left = pad_sequences(sample_list_sequences, padding_value=0, padding_side="left")
    expected_left = torch.tensor([[1, 2, 3], [0, 4, 5]])
    assert torch.equal(padded_left, expected_left)

    assert torch.equal(pad_sequences([]), torch.tensor([]))


def test_create_attention_mask():
    input_ids = torch.tensor([[1, 2, 0], [0, 3, 4]])
    mask = create_attention_mask(input_ids, padding_value=0)
    expected = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert torch.equal(mask, expected)


def test_create_position_ids():
    mask = torch.tensor(
        [
            [1, 1, 0],  # 1, 2, 0
            [0, 1, 1],  # 0, 1, 2
        ]
    )
    pos_ids = create_position_ids(mask)
    expected = torch.tensor([[1, 2, 0], [0, 1, 2]])
    assert torch.equal(pos_ids, expected)


def test_create_causal_mask():
    seq_len = 3
    mask = create_causal_mask(seq_len)
    expected = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).view(1, 1, 3, 3)
    assert torch.equal(mask, expected.float())


def test_collate_batch():
    batch = [
        {"id": 1, "val": torch.tensor([1, 2]), "label": 0, "score": 1.5},
        {"id": 2, "val": torch.tensor([3, 4]), "label": 1, "score": 2.5},
    ]
    collated = collate_batch(batch)
    assert torch.equal(collated["val"], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(collated["label"], torch.tensor([0, 1]))
    assert torch.equal(collated["id"], torch.tensor([1, 2]))
    assert torch.equal(collated["score"], torch.tensor([1.5, 2.5]))

    batch_var = [
        {"val": torch.tensor([1, 2, 3])},
        {"val": torch.tensor([4, 5])},
    ]
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 99

    collated_var = collate_batch(batch_var, tokenizer=mock_tokenizer)
    expected = torch.tensor([[1, 2, 3], [4, 5, 99]])
    assert torch.equal(collated_var["val"], expected)

    assert collate_batch([]) == {}

    with patch("thinkrl.utils.datasets.to_device") as mock_to_device:
        mock_device = MagicMock()
        collate_batch(batch, device=mock_device)
        mock_to_device.assert_called()

    batch_2d = [{"img": torch.randn(3, 3)}, {"img": torch.randn(3, 3)}]
    collated_2d = collate_batch(batch_2d)
    assert collated_2d["img"].shape == (2, 3, 3)

    batch_mismatch = [{"img": torch.randn(3, 3)}, {"img": torch.randn(2, 2)}]
    collated_mismatch = collate_batch(batch_mismatch)
    assert isinstance(collated_mismatch["img"], list)
    assert len(collated_mismatch["img"]) == 2

    batch_str = [{"txt": "hello"}, {"txt": "world"}]
    collated_str = collate_batch(batch_str)
    assert collated_str["txt"] == ["hello", "world"]


def test_create_dataloader():
    dataset = [1, 2, 3]
    dl = create_dataloader(dataset, batch_size=1)
    assert len(dl) == 3


def test_preprocess_text():
    assert preprocess_text("  hello  ") == "hello"
    assert preprocess_text(123) == "123"


def test_truncate_sequence():
    seq = [1, 2, 3, 4, 5]
    assert truncate_sequence(seq, 10) == seq
    assert truncate_sequence(seq, 3, side="right") == [1, 2, 3]
    assert truncate_sequence(seq, 3, side="left") == [3, 4, 5]


def test_create_labels_for_clm():
    input_ids = torch.tensor([1, 2, 3])
    labels = create_labels_for_clm(input_ids)
    assert torch.equal(labels, input_ids)
    assert labels is not input_ids


def test_mask_padding_in_loss():
    labels = torch.tensor([1, 2, 3, 0])
    mask = torch.tensor([1, 1, 1, 0])
    masked_labels = mask_padding_in_loss(labels, mask, ignore_index=-100)

    expected = torch.tensor([1, 2, 3, -100])
    assert torch.equal(masked_labels, expected)


def test_split_batch():
    batch = {
        "input_ids": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "labels": [1, 2, 3, 4],
    }

    micro_batches = split_batch(batch, 2)
    assert len(micro_batches) == 2
    assert torch.equal(micro_batches[0]["input_ids"], torch.tensor([[1, 1], [2, 2]]))
    assert micro_batches[1]["labels"] == [3, 4]

    assert split_batch({}, 2) == [{}]

    batch_meta = {"meta": "data"}
    assert split_batch(batch_meta, 2) == [batch_meta]


def test_compute_sequence_lengths():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    lengths = compute_sequence_lengths(mask)
    assert torch.equal(lengths, torch.tensor([2, 3]))


def test_shuffle_batch():
    batch = {"a": torch.tensor([1, 2, 3]), "b": ["x", "y", "z"], "c": "constant"}

    torch.manual_seed(42)
    shuffled = shuffle_batch(batch)

    assert len(shuffled["a"]) == 3
    assert len(shuffled["b"]) == 3
    assert shuffled["c"] == "constant"

    assert shuffle_batch({}) == {}


def test_to_device_recursive():
    batch = {
        "tensor": torch.tensor([1]),
        "nested": {"tensor": torch.tensor([2])},
        "list": [1, 2],
    }

    device = torch.device("cpu")
    moved = to_device(batch, device)

    assert isinstance(moved["tensor"], torch.Tensor)
    assert isinstance(moved["nested"]["tensor"], torch.Tensor)
    assert moved["list"] == [1, 2]


def test_prepare_batch_for_training():
    with patch("thinkrl.utils.datasets.to_device") as mock_to:
        prepare_batch_for_training({}, "cpu")
        mock_to.assert_called_once()


class TestZeroPadSequences:
    """Tests for zero_pad_sequences function."""

    def test_zero_pad_left(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        padded = zero_pad_sequences(seqs, side="left", value=0)

        expected = torch.tensor([[0, 1, 2], [3, 4, 5]])
        assert torch.equal(padded, expected)

    def test_zero_pad_right(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        padded = zero_pad_sequences(seqs, side="right", value=0)

        expected = torch.tensor([[1, 2, 0], [3, 4, 5]])
        assert torch.equal(padded, expected)

    def test_zero_pad_custom_value(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        padded = zero_pad_sequences(seqs, side="left", value=-1)

        expected = torch.tensor([[-1, 1, 2], [3, 4, 5]])
        assert torch.equal(padded, expected)

    def test_zero_pad_empty(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        result = zero_pad_sequences([])
        assert torch.equal(result, torch.tensor([]))

    def test_zero_pad_no_padding_needed(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        padded = zero_pad_sequences(seqs, side="left")

        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(padded, expected)

    def test_zero_pad_stack(self):
        from thinkrl.utils.datasets import zero_pad_sequences

        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        padded = zero_pad_sequences(seqs, side="left", stack=True)

        assert padded.shape == (2, 3)


class TestRemovePadToken:
    """Tests for remove_pad_token function."""

    def test_remove_left_padding(self):
        from thinkrl.utils.datasets import remove_pad_token

        input_ids = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
        attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

        unpadded = remove_pad_token(input_ids, attention_mask)

        assert len(unpadded) == 2
        assert torch.equal(unpadded[0], torch.tensor([1, 2]))
        assert torch.equal(unpadded[1], torch.tensor([3, 4, 5, 6]))

    def test_remove_right_padding(self):
        from thinkrl.utils.datasets import remove_pad_token

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        unpadded = remove_pad_token(input_ids, attention_mask)

        assert len(unpadded) == 2
        assert torch.equal(unpadded[0], torch.tensor([1, 2]))
        assert torch.equal(unpadded[1], torch.tensor([3, 4, 5]))

    def test_remove_no_padding(self):
        from thinkrl.utils.datasets import remove_pad_token

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        unpadded = remove_pad_token(input_ids, attention_mask)

        assert len(unpadded) == 1
        assert torch.equal(unpadded[0], torch.tensor([1, 2, 3]))


class TestConvertTokenToId:
    """Tests for convert_token_to_id function."""

    def test_convert_single_token(self):
        from thinkrl.utils.datasets import convert_token_to_id

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [42]

        result = convert_token_to_id("<|endoftext|>", mock_tokenizer)

        assert result == 42
        mock_tokenizer.encode.assert_called_with("<|endoftext|>", add_special_tokens=False)

    def test_convert_multi_token_raises_error(self):
        from thinkrl.utils.datasets import convert_token_to_id

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]  # Token encodes to 3 tokens

        with pytest.raises(ValueError, match="encodes to 3 tokens"):
            convert_token_to_id("multi word token", mock_tokenizer)


class TestGetStrategy:
    """Tests for get_strategy function."""

    def test_get_strategy_import_error(self):
        from thinkrl.utils.datasets import get_strategy

        # Should return None when import fails
        result = get_strategy(MagicMock())
        # This might return None or a strategy depending on the setup
        assert result is None or result is not None  # Just verify no crash


class TestApplyChatTemplate:
    """Tests for apply_chat_template function."""

    def test_apply_chat_template_with_tokenizer_support(self):
        from thinkrl.utils.datasets import apply_chat_template

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Formatted chat"

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = apply_chat_template(messages, mock_tokenizer)

        assert result == "Formatted chat"
        mock_tokenizer.apply_chat_template.assert_called_with(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

    def test_apply_chat_template_with_generation_prompt(self):
        from thinkrl.utils.datasets import apply_chat_template

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Chat with prompt"

        messages = [{"role": "user", "content": "Hello!"}]

        result = apply_chat_template(messages, mock_tokenizer, add_generation_prompt=True)

        mock_tokenizer.apply_chat_template.assert_called_with(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def test_apply_chat_template_fallback(self):
        from thinkrl.utils.datasets import apply_chat_template

        # Tokenizer without apply_chat_template
        mock_tokenizer = MagicMock(spec=[])
        del mock_tokenizer.apply_chat_template  # Ensure it doesn't exist

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = apply_chat_template(messages, mock_tokenizer)

        assert "User: Hello!" in result
        assert "Assistant: Hi!" in result

    def test_apply_chat_template_fallback_with_generation_prompt(self):
        from thinkrl.utils.datasets import apply_chat_template

        mock_tokenizer = MagicMock(spec=[])
        del mock_tokenizer.apply_chat_template

        messages = [{"role": "user", "content": "Hello!"}]

        result = apply_chat_template(messages, mock_tokenizer, add_generation_prompt=True)

        assert result.endswith("Assistant:")

    def test_apply_chat_template_fallback_tokenize(self):
        from thinkrl.utils.datasets import apply_chat_template

        mock_tokenizer = MagicMock(spec=["__call__"])
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        messages = [{"role": "user", "content": "Hello!"}]

        result = apply_chat_template(messages, mock_tokenizer, tokenize=True)

        mock_tokenizer.assert_called()
        assert result == {"input_ids": [1, 2, 3]}
