"""
Automatic License Plate Recognition (ALPR) System.

This package provides a complete ALPR system using YOLOv8 models for:
- License plate detection
- Character detection and recognition
- State classification
- Vehicle detection and classification

The system supports both PyTorch and ONNX model formats and includes
hardware acceleration support for CUDA, MPS (Apple Silicon), and CPU.
"""

from .config import ALPRConfig, load_from_env
from .core import ALPRSystem
from .adapter import ALPRAdapter
from .exceptions import ALPRException, ModelLoadingError, InferenceError, CharacterRecognitionError

__version__ = "1.0.0"
__all__ = [
    "ALPRConfig",
    "load_from_env", 
    "ALPRSystem",
    "ALPRAdapter",
    "ALPRException",
    "ModelLoadingError",
    "InferenceError", 
    "CharacterRecognitionError"
]
