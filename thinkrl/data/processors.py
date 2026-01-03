"""
ThinkRL Data Processors
=======================

Processors for multimodal data (Image, Audio).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """
    Abstract base class for data processors.

    Provides a standard interface for processing different types of data
    (text, image, audio) in RLHF pipelines.
    """

    def __init__(self, **kwargs):
        """Initialize the processor with optional configuration."""
        self.config = kwargs

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process the input data.

        Args:
            data: Input data to process
            **kwargs: Additional processing arguments

        Returns:
            Processed data
        """
        pass

    def batch_process(self, batch: list[Any], **kwargs) -> list[Any]:
        """
        Process a batch of data.

        Args:
            batch: List of data items to process
            **kwargs: Additional processing arguments

        Returns:
            List of processed data items
        """
        return [self.process(item, **kwargs) for item in batch]


class TextProcessor(DataProcessor):
    """Processor for text data."""

    def __init__(self, tokenizer: Any = None, max_length: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, data: str, **kwargs) -> dict[str, Any]:
        """Process text data."""
        if self.tokenizer is None:
            return {"text": data}

        return self.tokenizer(
            data,
            max_length=kwargs.get("max_length", self.max_length),
            truncation=True,
            return_tensors="pt",
        )


class ImageProcessor(DataProcessor):
    """Processor for image data."""

    def __init__(self, transform: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def process(self, data: str | Any, **kwargs) -> Any:
        """Process image data (path or PIL Image)."""
        return process_image(data, transform=self.transform)


class AudioProcessor(DataProcessor):
    """Processor for audio data."""

    def __init__(self, sample_rate: int = 16000, transform: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.transform = transform

    def process(self, data: str, **kwargs) -> Any:
        """Process audio data (path)."""
        return process_audio(
            data,
            sr=kwargs.get("sample_rate", self.sample_rate),
            transform=self.transform,
        )


def get_data_processor(
    processor_type: str,
    **kwargs,
) -> DataProcessor:
    """
    Factory function to get a data processor by type.

    Args:
        processor_type: Type of processor ("text", "image", "audio")
        **kwargs: Configuration arguments for the processor

    Returns:
        DataProcessor instance

    Raises:
        ValueError: If processor_type is not recognized
    """
    processors = {
        "text": TextProcessor,
        "image": ImageProcessor,
        "audio": AudioProcessor,
    }

    if processor_type not in processors:
        raise ValueError(
            f"Unknown processor type: {processor_type}. "
            f"Available: {list(processors.keys())}"
        )

    return processors[processor_type](**kwargs)

# Optional dependencies
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


def process_image(image_path: str, transform: Any | None = None) -> Any | None:
    """
    Load and process an image file.
    """
    if not _PIL_AVAILABLE:
        logger.warning("Pillow not installed. Cannot process image.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


def process_audio(audio_path: str, sr: int = 16000, transform: Any | None = None) -> Any | None:
    """
    Load and process an audio file.
    """
    if not _AUDIO_AVAILABLE:
        logger.warning("librosa/soundfile not installed. Cannot process audio.")
        return None

    try:
        # Load audio
        audio, orig_sr = librosa.load(audio_path, sr=None)

        # Resample if needed
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        if transform:
            # Transform usually expects tensor or specific format
            audio = transform(audio, sampling_rate=sr, return_tensors="pt")

        return audio
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {e}")
        return None
