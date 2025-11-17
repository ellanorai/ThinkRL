"""
ThinkRL Data Processors
=======================

Processors for multimodal data (Image, Audio).
"""

import logging
from typing import Optional, Any, Union

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


def process_image(
    image_path: str, 
    transform: Optional[Any] = None
) -> Optional[Any]:
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


def process_audio(
    audio_path: str, 
    sr: int = 16000, 
    transform: Optional[Any] = None
) -> Optional[Any]:
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