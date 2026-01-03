"""
Test Suite for ThinkRL Processor Utilities
==========================================

Tests for:
- thinkrl.utils.processor
"""

from unittest.mock import MagicMock, patch

import pytest

from thinkrl.utils.processor import (
    PROCESSORS,
    best_of_n_processor,
    conditional_sft_processor,
    create_pairwise_data,
    filter_by_reward_threshold,
    get_processor,
    iterative_dpo_processor,
    register_processor,
    rejection_sampling_processor,
    reward_normalization,
)


class TestRewardNormalization:
    """Tests for reward_normalization function."""

    def test_basic_normalization(self):
        """Test basic z-score normalization."""
        objs = [
            {"input": "a", "reward": 1.0},
            {"input": "b", "reward": 2.0},
            {"input": "c", "reward": 3.0},
            {"input": "d", "reward": 4.0},
            {"input": "e", "reward": 5.0},
        ]

        result = reward_normalization(objs, reward_key="reward")

        # Mean should be approximately 0
        rewards = [obj["reward"] for obj in result]
        mean = sum(rewards) / len(rewards)
        assert abs(mean) < 1e-6

        # Std should be approximately 1
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5
        assert abs(std - 1.0) < 0.1

    def test_empty_rewards(self):
        """Test with empty rewards list."""
        objs = [{"input": "a"}, {"input": "b"}]
        result = reward_normalization(objs, reward_key="reward")
        assert result == objs

    def test_custom_reward_key(self):
        """Test with custom reward key."""
        objs = [
            {"input": "a", "score": 1.0},
            {"input": "b", "score": 3.0},
        ]

        result = reward_normalization(objs, reward_key="score")
        assert "score" in result[0]


class TestConditionalSFTProcessor:
    """Tests for conditional_sft_processor function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "Question 1", "output": "Answer 1", "reward": 0.8},
            {"input": "Question 2", "output": "Answer 2", "reward": 0.5},
        ]

    def test_basic_processing(self, sample_data):
        """Test basic conditional SFT processing."""
        args = MagicMock()
        result = conditional_sft_processor(
            args,
            sample_data,
            input_key="input",
            output_key="output",
            reward_key="reward",
        )

        assert len(result) == 2
        assert "<rm_score>:" in result[0]["input"]
        assert "0.8" in result[0]["input"]

    def test_custom_template(self, sample_data):
        """Test with custom template."""
        args = MagicMock()
        result = conditional_sft_processor(
            args,
            sample_data,
            input_template="Input: {input} Score: {reward}",
        )

        assert "Input:" in result[0]["input"]
        assert "Score:" in result[0]["input"]

    def test_invalid_template(self, sample_data):
        """Test with invalid template raises error."""
        args = MagicMock()
        with pytest.raises(ValueError, match="must contain"):
            conditional_sft_processor(
                args,
                sample_data,
                input_template="Invalid template",
            )

    def test_with_normalization(self, sample_data):
        """Test with reward normalization."""
        args = MagicMock()
        result = conditional_sft_processor(
            args,
            sample_data,
            normalize_rewards=True,
        )

        assert len(result) == 2


class TestRejectionSamplingProcessor:
    """Tests for rejection_sampling_processor function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "Q1", "output": "A1", "reward": 0.5},
            {"input": "Q1", "output": "A2", "reward": 0.9},  # Best for Q1
            {"input": "Q2", "output": "A3", "reward": 0.7},
            {"input": "Q2", "output": "A4", "reward": 0.3},
        ]

    def test_basic_rejection_sampling(self, sample_data):
        """Test basic rejection sampling."""
        args = MagicMock()
        result = rejection_sampling_processor(args, sample_data)

        assert len(result) == 2

        # Check best outputs are kept
        q1_result = next(r for r in result if r["input"] == "Q1")
        assert q1_result["output"] == "A2"
        assert q1_result["reward"] == 0.9

        q2_result = next(r for r in result if r["input"] == "Q2")
        assert q2_result["output"] == "A3"
        assert q2_result["reward"] == 0.7

    def test_single_output_per_input(self):
        """Test with single output per input."""
        data = [
            {"input": "Q1", "output": "A1", "reward": 0.5},
            {"input": "Q2", "output": "A2", "reward": 0.7},
        ]
        args = MagicMock()
        result = rejection_sampling_processor(args, data)

        assert len(result) == 2


class TestIterativeDPOProcessor:
    """Tests for iterative_dpo_processor function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "Q1", "output": "Good", "reward": 0.9},
            {"input": "Q1", "output": "Bad", "reward": 0.1},
            {"input": "Q2", "output": "Medium", "reward": 0.5},
            {"input": "Q2", "output": "Worst", "reward": 0.2},
        ]

    def test_basic_dpo_processing(self, sample_data):
        """Test basic DPO pair creation."""
        args = MagicMock()
        result = iterative_dpo_processor(args, sample_data)

        assert len(result) == 2

        q1_pair = next(r for r in result if r["input"] == "Q1")
        assert q1_pair["chosen"] == "Good"
        assert q1_pair["rejected"] == "Bad"
        assert q1_pair["chosen_reward"] == 0.9
        assert q1_pair["rejected_reward"] == 0.1

    def test_single_output_skipped(self):
        """Test that inputs with single output are skipped."""
        data = [
            {"input": "Q1", "output": "Only", "reward": 0.5},
            {"input": "Q2", "output": "Good", "reward": 0.9},
            {"input": "Q2", "output": "Bad", "reward": 0.1},
        ]
        args = MagicMock()
        result = iterative_dpo_processor(args, data)

        assert len(result) == 1
        assert result[0]["input"] == "Q2"


class TestBestOfNProcessor:
    """Tests for best_of_n_processor function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "Q1", "output": "A1", "reward": 0.5},
            {"input": "Q1", "output": "A2", "reward": 0.9},
            {"input": "Q1", "output": "A3", "reward": 0.7},
            {"input": "Q1", "output": "A4", "reward": 0.8},
            {"input": "Q2", "output": "B1", "reward": 0.6},
            {"input": "Q2", "output": "B2", "reward": 0.4},
        ]

    def test_best_of_4(self, sample_data):
        """Test best-of-4 selection."""
        args = MagicMock()
        result = best_of_n_processor(args, sample_data, n=4)

        # Only Q1 has >= 4 samples
        assert len(result) == 1
        assert result[0]["input"] == "Q1"
        assert result[0]["output"] == "A2"  # Best
        assert result[0]["reward"] == 0.9

    def test_best_of_2(self, sample_data):
        """Test best-of-2 selection."""
        args = MagicMock()
        result = best_of_n_processor(args, sample_data, n=2)

        # Both Q1 and Q2 have >= 2 samples
        assert len(result) == 2


class TestFilterByRewardThreshold:
    """Tests for filter_by_reward_threshold function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "a", "reward": 0.9},
            {"input": "b", "reward": 0.7},
            {"input": "c", "reward": 0.5},
            {"input": "d", "reward": 0.3},
        ]

    def test_filter_above_threshold(self, sample_data):
        """Test filtering above threshold."""
        result = filter_by_reward_threshold(sample_data, threshold=0.6, keep_above=True)

        assert len(result) == 2
        assert all(r["reward"] >= 0.6 for r in result)

    def test_filter_below_threshold(self, sample_data):
        """Test filtering below threshold."""
        result = filter_by_reward_threshold(sample_data, threshold=0.6, keep_above=False)

        assert len(result) == 2
        assert all(r["reward"] < 0.6 for r in result)


class TestCreatePairwiseData:
    """Tests for create_pairwise_data function."""

    @pytest.fixture
    def sample_data(self):
        return [
            {"input": "Q1", "output": "A", "reward": 0.9},
            {"input": "Q1", "output": "B", "reward": 0.5},
            {"input": "Q1", "output": "C", "reward": 0.3},
        ]

    def test_create_pairs_no_margin(self, sample_data):
        """Test creating pairs without margin."""
        result = create_pairwise_data(sample_data, margin=0.0)

        # For 3 outputs, we get 6 ordered pairs (n * (n-1))
        # But only pairs where chosen > rejected
        # A > B, A > C, B > C = 3 pairs
        assert len(result) == 3

        # Check all pairs have chosen_reward > rejected_reward
        for pair in result:
            assert pair["chosen_reward"] > pair["rejected_reward"]

    def test_create_pairs_with_margin(self, sample_data):
        """Test creating pairs with margin."""
        result = create_pairwise_data(sample_data, margin=0.3)

        # Only pairs with reward diff >= 0.3
        # A-B: 0.4, A-C: 0.6, B-C: 0.2 -> 2 pairs
        assert len(result) == 2


class TestProcessorRegistry:
    """Tests for processor registry functions."""

    def test_get_processor_valid(self):
        """Test getting valid processor."""
        processor = get_processor("rs")
        assert processor is rejection_sampling_processor

        processor = get_processor("rejection_sampling")
        assert processor is rejection_sampling_processor

    def test_get_processor_invalid(self):
        """Test getting invalid processor raises error."""
        with pytest.raises(ValueError, match="Unknown processor"):
            get_processor("invalid_processor")

    def test_register_processor(self):
        """Test registering custom processor."""

        def my_processor(args, objs):
            return objs

        register_processor("my_custom", my_processor)

        assert "my_custom" in PROCESSORS
        assert get_processor("my_custom") is my_processor

        # Cleanup
        del PROCESSORS["my_custom"]

    def test_all_processors_registered(self):
        """Test that all standard processors are registered."""
        expected = [
            "rs",
            "rejection_sampling",
            "csft",
            "conditional_sft",
            "iter_dpo",
            "iterative_dpo",
            "best_of_n",
            "bon",
        ]

        for name in expected:
            assert name in PROCESSORS
