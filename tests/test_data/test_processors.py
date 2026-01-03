"""
Test Suite for ThinkRL Data Processors
======================================

Tests for:
- thinkrl.data.processors.process_image
- thinkrl.data.processors.process_audio

"""

from unittest.mock import MagicMock, patch

import pytest

from thinkrl.data.processors import process_audio, process_image


# Check actual availability in the module we are testing
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import librosa

    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


# --- Image Processor Tests ---


class TestProcessImage:
    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    @patch("thinkrl.data.processors._PIL_AVAILABLE", True)
    @patch("thinkrl.data.processors.Image")
    def test_process_image_success(self, mock_image_class):
        """Test successful image loading."""
        # Setup mock image
        mock_img_instance = MagicMock()
        mock_img_instance.convert.return_value = "rgb_image"

        # Setup open return
        mock_image_class.open.return_value = mock_img_instance

        # Call function
        result = process_image("test.jpg")

        # Verify
        mock_image_class.open.assert_called_with("test.jpg")
        mock_img_instance.convert.assert_called_with("RGB")
        assert result == "rgb_image"

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    @patch("thinkrl.data.processors._PIL_AVAILABLE", True)
    @patch("thinkrl.data.processors.Image")
    def test_process_image_with_transform(self, mock_image_class):
        """Test image processing with a transform function."""
        mock_img_instance = MagicMock()
        mock_img_instance.convert.return_value = "rgb_image"
        mock_image_class.open.return_value = mock_img_instance

        # Mock transform
        mock_transform = MagicMock(return_value="transformed_image")

        result = process_image("test.jpg", transform=mock_transform)

        mock_transform.assert_called_with("rgb_image")
        assert result == "transformed_image"

    def test_process_image_no_pil(self):
        """Test behavior when Pillow is not installed."""
        # Should log warning and return None
        # Force _PIL_AVAILABLE to False for this test case even if it IS installed
        with patch("thinkrl.data.processors._PIL_AVAILABLE", False):
            with patch("thinkrl.data.processors.logger") as mock_logger:
                result = process_image("test.jpg")

                assert result is None
                mock_logger.warning.assert_called_once()
                assert "Pillow not installed" in mock_logger.warning.call_args[0][0]

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    @patch("thinkrl.data.processors._PIL_AVAILABLE", True)
    @patch("thinkrl.data.processors.Image")
    def test_process_image_error(self, mock_image_class):
        """Test handling of exceptions during image processing."""
        mock_image_class.open.side_effect = OSError("File corrupted")

        with patch("thinkrl.data.processors.logger") as mock_logger:
            result = process_image("bad.jpg")

            assert result is None
            mock_logger.error.assert_called_once()
            assert "Error processing image" in mock_logger.error.call_args[0][0]


# --- Audio Processor Tests ---


class TestProcessAudio:
    @pytest.mark.skipif(not _AUDIO_AVAILABLE, reason="Librosa not installed")
    @patch("thinkrl.data.processors._AUDIO_AVAILABLE", True)
    @patch("thinkrl.data.processors.librosa")
    def test_process_audio_success_no_resample(self, mock_librosa):
        """Test audio loading where SR matches (no resampling needed)."""
        # Setup: return audio and matching SR
        mock_librosa.load.return_value = ("audio_data", 16000)

        result = process_audio("test.wav", sr=16000)

        mock_librosa.load.assert_called_with("test.wav", sr=None)
        # Should NOT call resample
        mock_librosa.resample.assert_not_called()
        assert result == "audio_data"

    @pytest.mark.skipif(not _AUDIO_AVAILABLE, reason="Librosa not installed")
    @patch("thinkrl.data.processors._AUDIO_AVAILABLE", True)
    @patch("thinkrl.data.processors.librosa")
    def test_process_audio_resample(self, mock_librosa):
        """Test audio loading with resampling."""
        # Setup: return audio and DIFFERENT SR (44100 vs target 16000)
        mock_librosa.load.return_value = ("audio_data_44k", 44100)
        mock_librosa.resample.return_value = "audio_data_16k"

        result = process_audio("test.wav", sr=16000)

        mock_librosa.resample.assert_called_with("audio_data_44k", orig_sr=44100, target_sr=16000)
        assert result == "audio_data_16k"

    @pytest.mark.skipif(not _AUDIO_AVAILABLE, reason="Librosa not installed")
    @patch("thinkrl.data.processors._AUDIO_AVAILABLE", True)
    @patch("thinkrl.data.processors.librosa")
    def test_process_audio_with_transform(self, mock_librosa):
        """Test audio processing with a transform."""
        mock_librosa.load.return_value = ("raw_audio", 16000)

        mock_transform = MagicMock(return_value="spectrogram")

        result = process_audio("test.wav", sr=16000, transform=mock_transform)

        # Verify transform call signature matches the implementation
        mock_transform.assert_called_with("raw_audio", sampling_rate=16000, return_tensors="pt")
        assert result == "spectrogram"

    def test_process_audio_no_libs(self):
        """Test behavior when librosa/soundfile are not installed."""
        with patch("thinkrl.data.processors._AUDIO_AVAILABLE", False):
            with patch("thinkrl.data.processors.logger") as mock_logger:
                result = process_audio("test.wav")

                assert result is None
                mock_logger.warning.assert_called_once()
                assert "librosa/soundfile not installed" in mock_logger.warning.call_args[0][0]

    @pytest.mark.skipif(not _AUDIO_AVAILABLE, reason="Librosa not installed")
    @patch("thinkrl.data.processors._AUDIO_AVAILABLE", True)
    @patch("thinkrl.data.processors.librosa")
    def test_process_audio_error(self, mock_librosa):
        """Test handling of exceptions during audio processing."""
        mock_librosa.load.side_effect = RuntimeError("Decode error")

        with patch("thinkrl.data.processors.logger") as mock_logger:
            result = process_audio("bad.wav")

            assert result is None
            mock_logger.error.assert_called_once()
            assert "Error processing audio" in mock_logger.error.call_args[0][0]


# --- Processor Class Tests ---

from thinkrl.data.processors import AudioProcessor, DataProcessor, ImageProcessor, TextProcessor, get_data_processor


class TestTextProcessor:
    def test_init(self):
        processor = TextProcessor(max_length=1024)
        assert processor.max_length == 1024
        assert processor.tokenizer is None

    def test_process_no_tokenizer(self):
        processor = TextProcessor()
        data = "hello world"
        result = processor.process(data)
        assert result == {"text": "hello world"}

    def test_process_with_tokenizer(self):
        mock_tokenizer = MagicMock(return_value={"input_ids": [1, 2, 3]})
        processor = TextProcessor(tokenizer=mock_tokenizer, max_length=128)

        data = "hello world"
        result = processor.process(data)

        mock_tokenizer.assert_called_with(data, max_length=128, truncation=True, return_tensors="pt")
        assert result == {"input_ids": [1, 2, 3]}

    def test_batch_process(self):
        # Test inherited batch_process
        processor = TextProcessor()
        batch = ["a", "b"]
        result = processor.batch_process(batch)
        assert result == [{"text": "a"}, {"text": "b"}]


class TestImageProcessor:
    @patch("thinkrl.data.processors.process_image")
    def test_process_calls_fn(self, mock_fn):
        processor = ImageProcessor()
        data = "img.jpg"
        processor.process(data)
        mock_fn.assert_called_with(data, transform=None)

    @patch("thinkrl.data.processors.process_image")
    def test_process_with_transform(self, mock_fn):
        mock_transform = MagicMock()
        processor = ImageProcessor(transform=mock_transform)
        processor.process("img.jpg")
        mock_fn.assert_called_with("img.jpg", transform=mock_transform)


class TestAudioProcessor:
    @patch("thinkrl.data.processors.process_audio")
    def test_process_calls_fn(self, mock_fn):
        processor = AudioProcessor(sample_rate=48000)
        processor.process("audio.wav")
        mock_fn.assert_called_with("audio.wav", sr=48000, transform=None)


class TestGetDataProcessorFactory:
    def test_get_text(self):
        p = get_data_processor("text", max_length=512)
        assert isinstance(p, TextProcessor)
        assert p.max_length == 512

    def test_get_image(self):
        p = get_data_processor("image")
        assert isinstance(p, ImageProcessor)

    def test_get_audio(self):
        p = get_data_processor("audio")
        assert isinstance(p, AudioProcessor)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown processor type"):
            get_data_processor("invalid")
