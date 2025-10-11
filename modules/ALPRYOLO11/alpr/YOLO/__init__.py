"""
YOLOv8 models for Automatic License Plate Recognition.

This module contains the YOLOv8-based detectors and classifiers for license plates,
characters, states, and vehicles. Supports both PyTorch and ONNX formats with
centralized session management.
"""

# Export base class and session manager
from .base import YOLOBase
from .session_manager import ONNXSessionManager, SessionConfig, get_session_manager, cleanup_session_manager

# Export detector and classifier classes
from .plate_detector import PlateDetector
from .character_detector import CharacterDetector
from .state_classifier import StateClassifier
from .vehicle_detector import VehicleDetector

# Export new character processing modules
from .char_organizer import CharacterOrganizer
from .char_classifier_manager import CharacterClassifierManager

__all__ = [
    'YOLOBase',
    'ONNXSessionManager', 
    'SessionConfig',
    'get_session_manager',
    'cleanup_session_manager',
    'PlateDetector',
    'CharacterDetector', 
    'StateClassifier',
    'VehicleDetector',
    'CharacterOrganizer',
    'CharacterClassifierManager'
]