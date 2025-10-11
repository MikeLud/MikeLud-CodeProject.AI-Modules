"""
Exception classes for the ALPR system.
Provides specialized exceptions for different error conditions.
"""
from typing import Optional, Any


class ALPRException(Exception):
    """Base exception for all ALPR-related errors"""
    def __init__(self, message: str, *args: Any):
        super().__init__(message, *args)
        self.message = message


class ConfigurationError(ALPRException):
    """Exception raised for configuration-related errors"""
    pass


class ModelLoadingError(ALPRException):
    """Exception raised when a model fails to load"""
    def __init__(self, model_path: str, original_error: Optional[Exception] = None):
        message = f"Failed to load model from {model_path}"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)
        self.model_path = model_path
        self.original_error = original_error


class InferenceError(ALPRException):
    """Exception raised when inference fails"""
    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        message = f"Inference failed for model {model_name}"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)
        self.model_name = model_name
        self.original_error = original_error


class ImageProcessingError(ALPRException):
    """Exception raised for image processing errors"""
    def __init__(self, operation: str, original_error: Optional[Exception] = None):
        message = f"Image processing operation '{operation}' failed"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class LicensePlateError(ALPRException):
    """Exception raised for license plate detection and processing errors"""
    pass


class CharacterRecognitionError(ALPRException):
    """Exception raised for character recognition errors"""
    pass


class VehicleDetectionError(ALPRException):
    """Exception raised for vehicle detection and classification errors"""
    pass
